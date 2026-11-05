"""Preservation contracts for the unfiltered workflow, using synthetic BDF only."""

import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
from eeg_raw_bdf import NativeBDF, waveform
from eeg_raw_qc import analyze_raw, alpha_peak, spectrum, condition_spectrum, validate_intervals


def fixture(path, seconds=12):
    specs = [
        ("O1", 256, "uV"),
        ("O2", 256, "uV"),
        ("T1", 256, "uV"),
        ("Status", 256, ""),
        ("PG1", 8, "pcs"),
        ("BDF Annotations", 64, ""),
    ]
    n = len(specs)
    fixed = bytearray(b" " * 256)
    fixed[:8] = b"\xffBIOSEMI"
    for offset, width, text in [
        (168, 8, "01.01.24"),
        (176, 8, "08.30.00"),
        (184, 8, str(256 * (n + 1))),
        (192, 44, "BDF+C"),
        (236, 8, str(seconds)),
        (244, 8, "1"),
        (252, 4, str(n)),
    ]:
        fixed[offset : offset + width] = text.encode().ljust(width)
    fields = [
        ([s[0] for s in specs], 16),
        ([""] * n, 80),
        ([s[2] for s in specs], 8),
        ([-8388608] * n, 8),
        ([8388607] * n, 8),
        ([-8388608] * n, 8),
        ([8388607] * n, 8),
        (["acquisition unknown"] * n, 80),
        ([s[1] for s in specs], 8),
        ([""] * n, 32),
    ]
    header = b"".join(str(v).encode().ljust(width) for values, width in fields for v in values)
    expected = {name: [] for name, _, _ in specs[:-1]}
    payload = []
    rng = np.random.default_rng(9)
    for record in range(seconds):
        for name, fs, unit in specs:
            if name == "BDF Annotations":
                tal = f"+{record}\x14\x14\0".encode()
                if record == 1:
                    tal += "+1\x152\x14标记A\x14\0".encode("utf8")
                if record == 3:
                    tal += b"+3\x151\x14BAD_ACQ_SKIP\x14\0"
                payload.append(tal.ljust(fs * 3, b"\0"))
                continue
            t = record + np.arange(fs) / fs
            if name in ("O1", "O2"):
                values = np.round(
                    (8 if record < 6 else 35) * np.sin(2 * np.pi * 10 * t)
                    + 12 * np.sin(2 * np.pi * 50 * t)
                    + rng.normal(0, 2, fs)
                ).astype("i4")
            elif name == "Status":
                values = np.arange(fs, dtype="i4") + record * fs + 2**20
            elif name == "PG1":
                values = np.arange(fs, dtype="i4") + record * fs
            else:
                values = rng.integers(-50, 50, fs, dtype="i4")
            expected[name].extend(values.tolist())
            unsigned = values & 0xFFFFFF
            payload.append(
                np.stack([unsigned & 255, (unsigned >> 8) & 255, (unsigned >> 16) & 255], axis=1)
                .astype("u1")
                .tobytes()
            )
    path.write_bytes(fixed + header + b"".join(payload))
    return expected


class RawQCTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        os.environ["_MNE_FAKE_HOME_DIR"] = str(cls.root / "mne_config")
        (cls.root / "mne_config").mkdir()
        cls.path = cls.root / "synthetic.bdf"
        cls.expected = fixture(cls.path)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_native_all_channels_and_rates_no_status_bitmask(self):
        source = NativeBDF(self.path)
        self.assertEqual([r["name"] for r in source.signals], list(self.expected))
        self.assertEqual({r["fs"] for r in source.signals}, {256, 8})
        for row in source.signals:
            values = source.read(row)
            expected = np.array(self.expected[row["name"]]) * (1e-6 if row["unit"] == "uV" else 1.0)
            np.testing.assert_allclose(values, expected, atol=1e-18)
        self.assertGreater(source.read(source.signals[3]).min(), 2**17)
        self.assertEqual(
            source.annotations(),
            [
                {"onset": 1.0, "duration": 2.0, "description": "标记A"},
                {"onset": 3.0, "duration": 1.0, "description": "BAD_ACQ_SKIP"},
            ],
        )

    def test_fif_roundtrip_bads_annotations_coordinates_and_no_processing(self):
        import mne
        from mne.io import BaseRaw

        options = {
            "bads": ["O2"],
            "bad_segments": [{"start": 4, "end": 5, "label": "movement"}],
            "eyes": [
                {"start": 0, "end": 6, "label": "open"},
                {"start": 6, "end": 12, "label": "closed"},
            ],
            "welch_seconds": 2,
        }
        forbidden = ["filter", "notch_filter", "resample", "set_eeg_reference", "interpolate_bads"]
        from contextlib import ExitStack

        with ExitStack() as stack:
            for name in forbidden:
                stack.enter_context(
                    patch.object(BaseRaw, name, side_effect=AssertionError(name + " forbidden"))
                )
            result = analyze_raw(self.path, self.root / "qc", options)
        self.assertEqual(len(result["checks"]), 2)
        self.assertEqual(sum(c["通道数"] for c in result["checks"]), 5)
        for path in [p for p in result["files"] if p.endswith(".fif")]:
            raw = mne.io.read_raw_fif(path, preload=False, verbose="ERROR")
            for i, name in enumerate(raw.ch_names):
                expected = np.array(self.expected[name]) * (
                    1e-6 if name in ("O1", "O2", "T1") else 1.0
                )
                np.testing.assert_allclose(raw.get_data(picks=[i])[0], expected, rtol=2e-15, atol=0)
            self.assertIn("BAD_movement", raw.annotations.description)
            self.assertIn("标记A", [e["original_description"] for e in raw.annotations.extras])
            self.assertIn("EYES_closed", raw.annotations.description)
            self.assertIn("BAD_source_acquisition_skip", raw.annotations.description)
            self.assertNotIn("BAD_ACQ_SKIP", raw.annotations.description)
            if "O1" in raw.ch_names:
                self.assertEqual(raw.info["bads"], ["O2"])
                self.assertTrue(np.isfinite(raw.info["chs"][0]["loc"][:3]).all())
                self.assertTrue(
                    np.isnan(raw.info["chs"][raw.ch_names.index("T1")]["loc"][:3]).all()
                )
            raw.close()
        candidates = [r for r in result["quality"] if r["通道"] == "O1"][0]
        self.assertGreater(candidates["50Hz突出_dB"], 10)
        self.assertFalse(candidates["人工坏道"])
        closed = [a for a in result["alpha"] if a["通道"] == "O1" and a["状态"] == "closed"][0]
        self.assertEqual(closed["alpha_peak_hz"], 10.0)
        self.assertEqual(closed["detected"], 1)
        self.assertTrue(all(a["detected"] == 0 for a in result["alpha"] if a["通道"] == "O2"))
        self.assertEqual(len(result["segments"]), 5)

    def test_no_peak_for_flat_and_known_alpha_peak(self):
        f, p, _, _ = spectrum(np.zeros(1024), 256, 2)
        self.assertEqual(alpha_peak(f, p)["detected"], 0)
        t = np.arange(2048) / 256
        f, p, _, _ = spectrum(np.sin(2 * np.pi * 10 * t), 256, 2)
        self.assertEqual(alpha_peak(f, p)["alpha_peak_hz"], 10)

    def test_voltage_samples_match_mne_reader_without_mixed_rate_selection(self):
        import mne

        raw = mne.io.read_raw_bdf(
            self.path,
            include=["O1", "O2", "T1"],
            stim_channel=None,
            infer_types=False,
            preload=True,
            verbose="ERROR",
        )
        source = NativeBDF(self.path)
        for name in raw.ch_names:
            row = next(r for r in source.signals if r["name"] == name)
            np.testing.assert_allclose(
                source.read(row), raw.get_data(picks=[name])[0], atol=2e-15, rtol=1e-10
            )
        raw.close()

    def test_fif_download_allowed_private_bdf_and_json_blocked(self):
        from http.server import ThreadingHTTPServer
        import threading
        import urllib.request
        import urllib.error
        import eeg_web_server as server

        folder = self.root / "http_files"
        run = folder / "raw_test_run"
        run.mkdir(parents=True)
        for name in ("source.bdf", "raw_qc_private.json", "group1_raw.fif"):
            (run / name).write_bytes(b"synthetic")
        previous = server.OUTPUT_ROOT
        server.OUTPUT_ROOT = folder
        self.addCleanup(setattr, server, "OUTPUT_ROOT", previous)
        httpd = ThreadingHTTPServer(("127.0.0.1", 0), server.Handler)
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        self.addCleanup(httpd.server_close)
        self.addCleanup(httpd.shutdown)
        base = f"http://127.0.0.1:{httpd.server_port}"
        with urllib.request.urlopen(base + "/download/raw_test_run/group1_raw.fif") as result:
            self.assertEqual(result.read(), b"synthetic")
        for name in ("source.bdf", "raw_qc_private.json"):
            with self.assertRaises(urllib.error.HTTPError) as caught:
                urllib.request.urlopen(base + "/download/raw_test_run/" + name)
            self.assertEqual(caught.exception.code, 404)

    def test_bad_span_does_not_bridge_welch_windows(self):
        x = np.sin(np.arange(1200))
        f, p, count = condition_spectrum(
            x, 100, 4, {"start": 0, "end": 12}, [{"start": 3, "end": 9}]
        )
        self.assertIsNone(p)
        self.assertEqual(count, 0)
        with self.assertRaises(ValueError):
            validate_intervals(
                [
                    {"start": 0, "end": 6, "label": "open"},
                    {"start": 5, "end": 8, "label": "closed"},
                ],
                12,
                eyes=True,
            )

    def test_truncation_rejected_not_silently_trimmed(self):
        path = self.root / "truncated.bdf"
        path.write_bytes(self.path.read_bytes()[:-1])
        with self.assertRaises(ValueError):
            NativeBDF(path)

    def test_waveform_envelope_preserves_actual_extrema(self):
        view = waveform(self.path, {"channels": ["O1", "Status", "PG1"], "start": 1, "seconds": 10})
        self.assertTrue(view["display_only"])
        for row in view["rows"]:
            fs = int(row["fs"])
            original = self.expected[row["name"]][fs : 11 * fs]
            values = [p[1] for p in row["points"]]
            self.assertAlmostEqual(min(values), min(original))
            self.assertAlmostEqual(max(values), max(original))


if __name__ == "__main__":
    unittest.main()
