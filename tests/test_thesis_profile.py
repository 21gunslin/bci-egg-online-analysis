"""Numerical reference and end-to-end checks for the opt-in paper scheme."""

import json
import os
from dataclasses import asdict
from pathlib import Path
import shutil
import sys
import tempfile
import threading
import unittest
import urllib.request
from http.server import ThreadingHTTPServer

import numpy as np
import pandas as pd
from scipy import signal

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
from eeg_feature_engine import AnalysisSettings, make_example_bdf, PIPELINE_VERSION
from eeg_thesis_features import (
    time_features,
    spectral_windows,
    extract_collection,
    network_metrics,
    continuous_blocks,
    compare_profile,
)
from test_events import event_file
import eeg_web_server as server


class ThesisProfileTests(unittest.TestCase):
    def settings(self, **kwargs):
        return AnalysisSettings(feature_profile="thesis_v1", **kwargs)

    def pre(self, seconds=8, scale=1):
        fs = 200
        t = np.arange(seconds * fs) / fs
        x = (
            np.stack(
                [
                    10 * np.sin(2 * np.pi * 10 * t),
                    20 * np.sin(2 * np.pi * 10 * t + 0.4),
                    7 * np.sin(2 * np.pi * 6 * t + 0.8),
                ]
            )
            * scale
        )
        return {
            "data": x,
            "fs": fs,
            "channel_names": ["F3", "F4", "Cz"],
            "kept_epoch_indices": np.arange(seconds // 2),
            "original_epoch_samples": 400,
        }

    def test_exact_moments_crossings_and_sum_line_length(self):
        x = np.array([-2.0, 0.0, 1.0, -1.0])[None, None, :]
        result = time_features(x, 200)
        self.assertAlmostEqual(result["variance"][0], x.var(ddof=1))
        self.assertAlmostEqual(result["std"][0], x.std(ddof=1))
        self.assertAlmostEqual(result["line_length"][0], 5)
        self.assertAlmostEqual(result["zero_crossing_rate"][0], 2 / 3)
        expected = ((x - x.mean()) ** 3).mean() / x.std(ddof=1) ** 3
        self.assertAlmostEqual(result["skewness"][0], expected)
        dx = np.diff(x, axis=-1) * 200
        ddx = np.diff(dx, axis=-1) * 200
        expected_complexity = np.sqrt(ddx.var(ddof=1) / dx.var(ddof=1)) / np.sqrt(
            dx.var(ddof=1) / x.var(ddof=1)
        )
        self.assertAlmostEqual(result["hjorth_complexity"][0], expected_complexity)

    def test_stft_density_equals_scipy_welch(self):
        pre = self.pre()
        f, _, power = spectral_windows([pre["data"]], 200)[0]
        expected_f, expected = signal.welch(
            pre["data"],
            fs=200,
            nperseg=400,
            noverlap=200,
            window="hann",
            detrend="constant",
            scaling="density",
        )
        np.testing.assert_allclose(f, expected_f)
        np.testing.assert_allclose(power.mean(axis=-1), expected, atol=1e-12)

    def test_optional_mne_welch_parity(self):
        from importlib.util import find_spec

        if find_spec("mne") is None:
            self.skipTest("Optional parity check: install requirements-mne.txt")
        # Isolate MNE's user preferences during verification; do not modify them.
        from unittest.mock import patch

        config_dir = tempfile.TemporaryDirectory()
        self.addCleanup(config_dir.cleanup)
        env = patch.dict(os.environ, {"_MNE_FAKE_HOME_DIR": config_dir.name})
        env.start()
        self.addCleanup(env.stop)
        import mne

        pre = self.pre()
        f, _, power = spectral_windows([pre["data"]], 200)[0]
        mne_psd, mf = mne.time_frequency.psd_array_welch(
            pre["data"],
            sfreq=200,
            n_fft=400,
            n_per_seg=400,
            n_overlap=200,
            window="hann",
            remove_dc=True,
            average="mean",
            verbose=False,
        )
        np.testing.assert_allclose(mf, f)
        np.testing.assert_allclose(mne_psd, power.mean(axis=-1), atol=1e-12)

    def test_fixed_schema_power_faa_and_scaling(self):
        pre = self.pre()
        channel, summary, meta = extract_collection([pre], self.settings())
        self.assertEqual(len(summary), 287)
        self.assertEqual(
            [len(meta["feature_categories"][k]) for k in ("时域", "频域", "时频域", "脑网络")],
            [84, 103, 70, 30],
        )
        values = summary.set_index("特征")["数值"]
        self.assertAlmostEqual(channel["频域__alpha_absolute_power"][0], 50, places=9)
        np.testing.assert_allclose(
            sum(channel[f"频域__{b}_relative_power"] for b in meta["bands"]), 1
        )
        self.assertAlmostEqual(values["频域__FAA__F3_F4"], np.log(4), places=9)
        self.assertTrue(np.isnan(values["频域__FAA__Fp1_Fp2"]))
        self.assertTrue(np.isnan(values["时域__std__枕区"]))
        self.assertTrue(any(row["缺失原因"] for row in meta["dictionary"]))
        scaled, _, _ = extract_collection([self.pre(scale=3)], self.settings())
        np.testing.assert_allclose(
            scaled["频域__total_power"], channel["频域__total_power"] * 9, rtol=1e-10
        )
        np.testing.assert_allclose(
            scaled["时频域__alpha_TF_cv"], channel["时频域__alpha_TF_cv"], atol=1e-12
        )

    def test_band_first_tf_and_no_windows_across_gaps(self):
        pre = self.pre(seconds=6)
        pre["data"][:, 400:800] *= 3
        pre["kept_epoch_indices"] = np.array([0, 3, 7])
        blocks = continuous_blocks(pre)
        self.assertEqual([b.shape[1] for b in blocks], [400] * 3)
        channel, _, meta = extract_collection([pre], self.settings())
        self.assertEqual(meta["coverage"]["psd_tf_windows"], 3)
        self.assertEqual(meta["coverage"]["continuous_runs"], 3)
        # Each independently generated window has exact alpha power 50, 450, 50.
        self.assertAlmostEqual(
            channel["时频域__alpha_TF_std"][0], np.std([50, 450, 50], ddof=1), places=8
        )
        self.assertAlmostEqual(
            channel["时频域__alpha_TF_cv"][0],
            np.std([50, 450, 50], ddof=1) / np.mean([50, 450, 50]),
            places=8,
        )
        self.assertEqual(meta["coverage"]["network"]["delta"]["samples"], 0)
        one, _, _ = extract_collection([self.pre(seconds=2)], self.settings())
        self.assertTrue(np.isnan(one["时频域__alpha_TF_std"][0]))

    def test_network_denominator_and_tie_density(self):
        matrix = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0.0]])
        metrics, details = network_metrics(matrix, 1.0)
        self.assertAlmostEqual(metrics["global_efficiency"], 1 / 3)
        self.assertAlmostEqual(details["reachable_fraction"], 1 / 3)
        matrix = np.ones((5, 5)) - np.eye(5)
        metrics, details = network_metrics(matrix, 0.25)
        self.assertEqual(details["edges"], 3)
        self.assertAlmostEqual(metrics["degree_mean"], 1.2)

    def test_end_to_end_single_pair_session_and_downloads(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            previous = server.OUTPUT_ROOT
            server.OUTPUT_ROOT = root
            self.addCleanup(setattr, server, "OUTPUT_ROOT", previous)
            httpd = ThreadingHTTPServer(("127.0.0.1", 0), server.Handler)
            threading.Thread(target=httpd.serve_forever, daemon=True).start()
            self.addCleanup(httpd.server_close)
            self.addCleanup(httpd.shutdown)
            base = "http://127.0.0.1:" + str(httpd.server_port)

            def post(route, body, headers=None):
                req = urllib.request.Request(
                    base + route, data=body, headers=headers or {}, method="POST"
                )
                with urllib.request.urlopen(req, timeout=180) as response:
                    return json.load(response)

            a = make_example_bdf(root / "a.bdf", seconds=8, fs=200)
            settings = {
                "feature_profile": "thesis_v1",
                "target_fs": 200,
                "cache_for_comparison": True,
            }
            result = post(
                "/api/analyze?filename=a.bdf",
                a.read_bytes(),
                {"X-EEG-Settings": json.dumps(settings)},
            )
            self.assertEqual(len(result["rows"]), 287)
            self.assertEqual(len(result["research_appendix"]["rows"]), 287)
            self.assertEqual(
                result["research_appendix"]["counts"],
                {"时域": 84, "频域": 103, "时频域": 70, "网络": 30},
            )
            self.assertTrue(
                all(
                    r["公式或定义"] and r["探索用途"] and r["解释界限"]
                    for r in result["research_appendix"]["rows"]
                )
            )
            self.assertTrue(result["feature_definition"])
            pair = post(
                "/api/compare",
                json.dumps(
                    {"run_a": result["run_id"], "run_b": result["run_id"], "max_seconds": 8}
                ).encode(),
                {"Content-Type": "application/json"},
            )
            self.assertEqual(len(pair["rows"]), 287)
            self.assertEqual(len(pair["research_appendix"]["rows"]), 287)
            self.assertTrue(
                all("记录B" in r and "变化量_B减A" in r for r in pair["research_appendix"]["rows"])
            )
            differences = [r["变化量_B减A"] for r in pair["rows"] if r["变化量_B减A"] is not None]
            np.testing.assert_allclose(differences, 0)
            wide = next(f for f in pair["files"] if "861" in f["name"])
            self.assertEqual(pd.read_csv(root / pair["run_id"] / wide["name"]).shape, (1, 861))
            b = root / "b.bdf"
            shutil.copy2(a, b)
            raw = bytearray(b.read_bytes())
            raw[176:184] = b"08.30.12"
            b.write_bytes(raw)
            evt = event_file(root / "evt.bdf", [b"+2\x14KeyIn:253\x14\0"])

            def upload(path, kind):
                return post(
                    "/api/upload?filename=" + path.name + "&kind=" + kind, path.read_bytes()
                )["upload_id"]

            session = post(
                "/api/session",
                json.dumps(
                    {
                        "items": [{"upload_id": upload(path, "eeg")} for path in (a, b)],
                        "evt_upload": upload(evt, "evt"),
                        "settings": settings,
                    }
                ).encode(),
                {"Content-Type": "application/json"},
            )
            self.assertEqual(len(session["rows"]), 287)
            self.assertEqual(len(session["research_appendix"]["rows"]), 287)
            self.assertEqual(session["metrics"]["continuous_runs"], 2)
            self.assertEqual(session["feature_coverage"]["psd_tf_windows"], 14)
            for info in (result, pair, session):
                for item in info["files"]:
                    with urllib.request.urlopen(base + item["url"]) as response:
                        self.assertEqual(response.status, 200)
                        if item["name"].endswith(".html"):
                            self.assertIn("attachment", response.headers["Content-Disposition"])
                            html = response.read().decode("utf8")
                            self.assertIn("论文探索性分析附录", html)
                            self.assertIn("公式或定义", html)
                            self.assertNotIn("<script", html)


if __name__ == "__main__":
    unittest.main()
