"""Synthetic-only checks. No patient recordings or follow-up labels are used."""
import json
import sys
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
from eeg_pair_spike import read_spike_counts, detect_candidates, compare_cached, analyze_spike
import eeg_web_server as server
from eeg_feature_engine import make_example_bdf, PIPELINE_VERSION


def marker_file(path, counts):
    fixed = bytearray(b" " * 256)
    fixed[:8] = b"\xffBIOSEMI"
    for start, width, value in [(168,8,"01.01.24"),(176,8,"08.30.00"),(184,8,"512"),
            (236,8,str(len(counts))),(244,8,"1"),(252,4,"1")]:
        fixed[start:start+width] = value.encode().ljust(width)
    fields = [("spike",16),("",80),("pcs",8),(0,8),(8388607,8),(0,8),(8388607,8),
              ("",80),(1,8),("",32)]
    header = b"".join(str(v).encode().ljust(w) for v,w in fields)
    path.write_bytes(fixed + header + b"".join(int(v).to_bytes(3,"little") for v in counts))


class WorkbenchTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.folder = Path(self.tmp.name)

    def test_marker_does_not_invent_waveform_findings(self):
        path = self.folder / "spike.bdf"
        marker_file(path, [0,2,0,1,0,0,0,0,0,0])
        counts, _ = read_spike_counts(path)
        self.assertEqual(int(counts.sum()), 3)
        result = analyze_spike(path, None, self.folder / "result")
        self.assertIsNone(result["metrics"]["candidate_events"])
        self.assertEqual(result["regions"], [])

    def test_synthetic_peak_and_following_slow_wave(self):
        fs = 500
        t = np.arange(5000) / fs
        data = np.random.default_rng(23).normal(0,1,(3,len(t)))
        data[0] += 100*np.exp(-.5*((t-3.1)/.012)**2)-60*np.exp(-.5*((t-3.32)/.085)**2)
        data[1] += 100*np.exp(-.5*((t-7.1)/.012)**2)
        counts = np.zeros(10,dtype=int)
        counts[[3,7]] = 1
        _, events, _ = detect_candidates(data, fs, ["F3","O2","Cz"], counts)
        self.assertTrue(any(abs(e["时间_s"]-3.1)<.03 and e["后续慢波"]=="有后续慢波候选" for e in events))
        self.assertTrue(any(abs(e["时间_s"]-7.1)<.03 for e in events))

    def test_identical_pair_has_zero_feature_difference(self):
        path = self.folder / "cache.npz"
        fs = 250
        data = np.random.default_rng(42).normal(0,5,(3,2500))
        np.savez_compressed(path,data=data,fs=fs,labels=np.array(["F3","O2","Cz"]),units=np.array(["µV"]*3))
        result = compare_cached(path,path,self.folder / "pair",max_seconds=8)
        self.assertGreater(len(result["rows"]),10)
        self.assertTrue(all(abs(r["变化量_B减A"])<1e-10 for r in result["rows"]))

    def test_single_analysis_provenance_and_versioned_pair_api(self):
        previous = server.OUTPUT_ROOT
        server.OUTPUT_ROOT = self.folder
        self.addCleanup(setattr, server, "OUTPUT_ROOT", previous)
        httpd = ThreadingHTTPServer(("127.0.0.1",0),server.Handler)
        threading.Thread(target=httpd.serve_forever,daemon=True).start()
        self.addCleanup(httpd.server_close)
        self.addCleanup(httpd.shutdown)
        base = "http://127.0.0.1:" + str(httpd.server_port)
        path = make_example_bdf(self.folder/"example.bdf",seconds=8,fs=200)
        request = urllib.request.Request(base+"/api/analyze?filename=example.bdf",data=path.read_bytes(),
            headers={"X-EEG-Settings":json.dumps({"cache_for_comparison":True})},method="POST")
        with urllib.request.urlopen(request,timeout=120) as response: result=json.load(response)
        manifest_path=self.folder/result["run_id"]/"analysis_manifest.json"
        manifest=json.loads(manifest_path.read_text(encoding="utf8"))
        self.assertEqual(manifest["pipeline_version"],PIPELINE_VERSION)
        self.assertEqual(manifest["provenance"]["analysis_voltage_unit"],"µV")
        self.assertEqual(manifest["preprocessed"]["fs"],200)
        body=json.dumps({"run_a":result["run_id"],"run_b":result["run_id"],"max_seconds":4}).encode()
        request=urllib.request.Request(base+"/api/compare",data=body,headers={"Content-Type":"application/json"},method="POST")
        with urllib.request.urlopen(request,timeout=120) as response: pair=json.load(response)
        self.assertTrue(pair["rows"])
        self.assertTrue(all(abs(row["变化量_B减A"])<1e-10 for row in pair["rows"]))
        manifest["pipeline_version"]="old"
        manifest_path.write_text(json.dumps(manifest),encoding="utf8")
        with self.assertRaises(urllib.error.HTTPError) as caught: urllib.request.urlopen(request,timeout=10)
        self.assertEqual(caught.exception.code,400)

    def test_static_routes_and_private_cache_block(self):
        previous = server.OUTPUT_ROOT
        server.OUTPUT_ROOT = self.folder
        self.addCleanup(setattr, server, "OUTPUT_ROOT", previous)
        private = self.folder / "test_run"
        private.mkdir()
        (private / "comparison_cache.npz").write_bytes(b"private")
        httpd = ThreadingHTTPServer(("127.0.0.1",0),server.Handler)
        thread = threading.Thread(target=httpd.serve_forever,daemon=True)
        thread.start()
        self.addCleanup(httpd.server_close)
        self.addCleanup(httpd.shutdown)
        base = "http://127.0.0.1:" + str(httpd.server_port)
        with urllib.request.urlopen(base+"/health") as response:
            self.assertTrue(json.load(response)["ok"])
        for route in ["/","/assets/eeg_app.js","/assets/eeg_app.css"]:
            with urllib.request.urlopen(base+route) as response:
                self.assertGreater(len(response.read()),100)
        with self.assertRaises(urllib.error.HTTPError) as caught:
            urllib.request.urlopen(base+"/download/test_run/comparison_cache.npz")
        self.assertEqual(caught.exception.code,404)


if __name__ == "__main__":
    unittest.main()
