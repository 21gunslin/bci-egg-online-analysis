"""HTTP format isolation, FIF sessions, and header clock integration."""

import json
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch
from http.server import ThreadingHTTPServer
import urllib.request
import urllib.error

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
from test_fif_workflow import make_fif
from test_events import event_file
import eeg_web_server as server


class FifApiTests(unittest.TestCase):
    def test_fif_session_and_strict_format_enforcement(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            previous = server.OUTPUT_ROOT
            server.OUTPUT_ROOT = root
            self.addCleanup(setattr, server, "OUTPUT_ROOT", previous)
            httpd = ThreadingHTTPServer(("127.0.0.1", 0), server.Handler)
            threading.Thread(target=httpd.serve_forever, daemon=True).start()
            self.addCleanup(httpd.server_close)
            self.addCleanup(httpd.shutdown)
            base = "http://127.0.0.1:" + str(httpd.server_port)

            def post(route, data, headers=None):
                req = urllib.request.Request(
                    base + route, data=data, headers=headers or {}, method="POST"
                )
                try:
                    with urllib.request.urlopen(req, timeout=120) as response:
                        return json.load(response)
                finally:
                    # The reply can arrive just before the server's finally releases its lock.
                    with server.ANALYSIS_LOCK:
                        pass

            def request(route, data):
                return post(route, json.dumps(data).encode(), {"Content-Type": "application/json"})

            def upload(path, kind="eeg"):
                return post(
                    "/api/upload?kind=" + kind + "&filename=" + path.name, path.read_bytes()
                )["upload_id"]

            a = make_fif(root / "a.fif")
            b = make_fif(root / "b.fif", 25)
            header = request("/api/fif-header", {"upload_id": upload(a)})
            self.assertEqual(header["first_time"], 0.5)
            self.assertEqual(header["duration"], 20)
            with self.assertRaises(urllib.error.HTTPError) as caught:
                post(
                    "/api/analyze?filename=a.fif",
                    a.read_bytes(),
                    {"X-EEG-Settings": json.dumps({"input_format": "bdf"})},
                )
            self.assertEqual(caught.exception.code, 400)
            with patch(
                "eeg_web_extra.run_manifest",
                side_effect=[
                    {"pipeline_version": server.PIPELINE_VERSION, "header": {"format": "BDF"}},
                    {"pipeline_version": server.PIPELINE_VERSION, "header": {"format": "FIF"}},
                ],
            ):
                with self.assertRaises(urllib.error.HTTPError) as mixed:
                    request("/api/compare", {"run_a": "example_a", "run_b": "example_b"})
            self.assertEqual(mixed.exception.code, 400)
            with self.assertRaises(urllib.error.HTTPError):
                request(
                    "/api/session",
                    {
                        "input_format": ["bdf", "fif"],
                        "items": [{"upload_id": upload(a)}, {"upload_id": upload(b)}],
                    },
                )
            evt = event_file(root / "evt.bdf", [b"+2\x14event\x14\0"])
            body = bytearray(evt.read_bytes())
            body[168:176] = b"24.04.24"
            evt.write_bytes(body)
            result = request(
                "/api/session",
                {
                    "input_format": "fif",
                    "items": [{"upload_id": upload(a)}, {"upload_id": upload(b)}],
                    "evt_upload": upload(evt, "evt"),
                    "settings": {"target_fs": 200, "notch": "none"},
                },
            )
            self.assertEqual(result["metrics"]["files"], 2)
            self.assertEqual(result["metrics"]["recorded_seconds"], 40)
            self.assertNotIn("Fp1", result["common_labels"])
            self.assertEqual(result["segments"][1]["距首段起点_s"], 25)
            with patch.object(
                server,
                "analyze_eeg_file",
                side_effect=server.EEGQualityError("synthetic QC failure"),
            ):
                review = post(
                    "/api/analyze?filename=a.fif",
                    a.read_bytes(),
                    {"X-EEG-Settings": json.dumps({"input_format": "fif"})},
                )
            self.assertTrue(review["review_only"])
            self.assertEqual(review["metrics"]["kept_epochs"], 0)
            self.assertEqual(sum(f["name"].endswith(".png") for f in review["files"]), 3)


if __name__ == "__main__":
    unittest.main()
