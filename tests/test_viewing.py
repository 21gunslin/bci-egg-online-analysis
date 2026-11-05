"""Display-only axis fitting and HTTP operations must leave source/FIF intact."""

import hashlib
import json
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch
import urllib.request
from http.server import ThreadingHTTPServer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
from eeg_raw_bdf import waveform
from eeg_viewing import axis_settings
from eeg_montage import montage_mapping
from test_raw_qc import fixture
import eeg_web_server as server


class ViewingTests(unittest.TestCase):
    def test_auto_fit_large_dc_and_flat_does_not_mutate_input(self):
        rows = [
            {
                "name": "O1",
                "unit": "µV",
                "points": [[0, -716000], [1, -714000]],
                "stats": {"min": -716000, "max": -714000, "median": -715000, "ptp": 2000},
            },
            {
                "name": "T2",
                "unit": "µV",
                "points": [[0, 750000], [1, 750000]],
                "stats": {"min": 750000, "max": 750000, "median": 750000, "ptp": 0},
            },
        ]
        before = json.dumps(rows)
        a = axis_settings(rows)
        self.assertLessEqual(a[0]["纵轴最小"], -716000)
        self.assertGreaterEqual(a[0]["纵轴最大"], -714000)
        self.assertTrue(a[1]["平线"])
        self.assertEqual(a[1]["中心原值"], 750000)
        self.assertEqual(before, json.dumps(rows))

    def test_more_than_60_seconds_uses_all_native_samples_for_stats(self):
        with tempfile.TemporaryDirectory() as temp:
            p = Path(temp) / "long.bdf"
            expected = fixture(p, seconds=80)
            view = waveform(p, {"channels": ["O1"], "start": 0, "seconds": 80})
            row = view["rows"][0]
            self.assertEqual(row["stats"]["n"], 80 * 256)
            self.assertEqual(view["stop"], 80)
            self.assertAlmostEqual(row["stats"]["max"], max(expected["O1"]))
            self.assertLessEqual(len(row["points"]), 2800)

    def test_restore_and_view_export_do_not_call_analysis_or_rewrite_source_or_fif(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            run = root / "raw_view_test"
            run.mkdir()
            path = run / "source.bdf"
            fixture(path)
            fif = run / "existing_raw.fif"
            fif.write_bytes(b"preserved FIF sentinel")
            mapping = montage_mapping(["O1", "O2", "T1", "Status", "PG1"])
            result = {
                "kind": "raw_qc",
                "channels": [{"name": n} for n in ["O1", "O2", "T1", "Status", "PG1"]],
                "montage_info": mapping,
                "files": [str(fif)],
                "options": {"bads": ["O2"]},
                "annotations": [],
            }
            (run / "raw_qc_result.json").write_text(json.dumps(result), encoding="utf8")
            (run / "raw_qc_private.json").write_text("{}")
            digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
            before = (digest(path), digest(fif), fif.stat().st_mtime_ns)
            old = server.OUTPUT_ROOT
            server.OUTPUT_ROOT = root
            httpd = ThreadingHTTPServer(("127.0.0.1", 0), server.Handler)
            threading.Thread(target=httpd.serve_forever, daemon=True).start()
            try:

                def post(route, data):
                    req = urllib.request.Request(
                        f"http://127.0.0.1:{httpd.server_port}" + route,
                        data=json.dumps(data).encode(),
                        headers={"Content-Type": "application/json"},
                        method="POST",
                    )
                    with urllib.request.urlopen(req, timeout=60) as response:
                        return json.load(response)

                with patch(
                    "eeg_raw_qc_api.subprocess.run",
                    side_effect=AssertionError("analysis subprocess forbidden"),
                ):
                    restored = post("/api/raw-qc/restore", {"run_id": "raw_view_test"})
                    self.assertTrue(restored["view_only_restored"])
                    self.assertEqual(restored["view_defaults"]["mode"], "channel")
                    view = post(
                        "/api/raw-qc/waveform",
                        {"run_id": "raw_view_test", "channels": ["O1"], "start": 1, "seconds": 2},
                    )
                    self.assertEqual(view["rows"][0]["stats"]["n"], 512)
                    exported = post(
                        "/api/raw-qc/view",
                        {
                            "run_id": "raw_view_test",
                            "channels": ["O1"],
                            "start": 1,
                            "seconds": 2,
                            "mode": "channel",
                        },
                    )
                    self.assertEqual(exported["kind"], "raw_view")
                    self.assertTrue(any(f["name"].endswith(".png") for f in exported["files"]))
                    upload = urllib.request.Request(
                        f"http://127.0.0.1:{httpd.server_port}/api/upload?kind=eeg&filename=readonly.bdf",
                        data=path.read_bytes(),
                        method="POST",
                    )
                    with urllib.request.urlopen(upload) as response:
                        token = json.load(response)["upload_id"]
                    opened = post("/api/raw-qc/open", {"upload_id": token})
                    self.assertTrue(opened["view_only_import"])
                    self.assertEqual(opened["checks"], [])
                    self.assertEqual(list((root / opened["run_id"]).glob("*.fif")), [])
                    self.assertEqual(digest(root / opened["run_id"] / "source.bdf"), before[0])
                    scan = post(
                        "/api/raw-qc/patrol",
                        {"run_id": "raw_view_test", "seconds": 2, "sensitivity": 6, "scope": "eeg"},
                    )
                    self.assertTrue(scan["candidates"])
                    self.assertEqual(scan["review"]["entries"], [])
                    reviewed = post(
                        "/api/raw-qc/review",
                        {
                            "run_id": "raw_view_test",
                            "action": "append",
                            "channels": ["O1"],
                            "start": 0,
                            "end": 0.5,
                            "status": "reviewed",
                        },
                    )
                    self.assertEqual(reviewed["entries"][0]["end"], 0.5)
                    self.assertTrue(
                        post("/api/raw-qc/review-export", {"run_id": "raw_view_test"})["files"]
                    )
                    self.assertEqual(
                        post("/api/raw-qc/review", {"run_id": "raw_view_test", "action": "undo"})[
                            "entries"
                        ],
                        [],
                    )
                    psd = post(
                        "/api/raw-qc/psd",
                        {
                            "run_id": "raw_view_test",
                            "channels": ["O1"],
                            "start": 1,
                            "seconds": 2,
                            "welch_seconds": 2,
                            "fmin": 0.5,
                            "fmax": 100,
                            "export": True,
                        },
                    )
                    self.assertEqual(psd["kind"], "view_psd")
                    self.assertTrue(psd["curves"])
                    self.assertTrue(psd["files"])
                self.assertEqual(before, (digest(path), digest(fif), fif.stat().st_mtime_ns))
                self.assertEqual(list((root / exported["run_id"]).glob("*.fif")), [])
            finally:
                httpd.shutdown()
                httpd.server_close()
                server.OUTPUT_ROOT = old


if __name__ == "__main__":
    unittest.main()
