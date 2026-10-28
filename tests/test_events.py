"""Synthetic BDF TAL and time-alignment regression tests."""
import json
import sys
import tempfile
import threading
import unittest
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"backend"))
from eeg_events import read_event_file, align_events
from eeg_feature_engine import make_example_bdf
import eeg_web_server as server


def event_file(path, entries, patient="Study-001", start="08.30.00"):
    fixed=bytearray(b" "*256);fixed[:8]=b"\xffBIOSEMI"
    for offset,width,value in [(8,80,patient),(168,8,"01.01.24"),(176,8,start),
            (184,8,"768"),(236,8,str(len(entries))),(244,8,"1"),(252,4,"2")]:
        fixed[offset:offset+width]=value.encode().ljust(width)
    fields=[(["dummy","BDF Annotations"],16),(["", ""],80),(["", ""],8),
            ([-1,-1],8),([1,1],8),([-8388608,-8388608],8),([8388607,8388607],8),
            (["", ""],80),([1,100],8),(["", ""],32)]
    header=b"".join(str(value).encode().ljust(width) for values,width in fields for value in values)
    records=[]
    for i,entry in enumerate(entries):
        payload=f"+{i}\x14\x14\0".encode()+entry
        records.append(b"\0"*3+payload.ljust(300,b"\0"))
    path.write_bytes(fixed+header+b"".join(records));return path


class EventTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.folder=Path(self.tmp.name)

    def test_tals_keep_duration_utf8_labels_and_ignore_timekeeping(self):
        p=event_file(self.folder/"evt.bdf",["+161.25\x150.5\x14KeyIn:253\x14中文\x14\0".encode()])
        data=read_event_file(p)
        self.assertEqual(len(data["events"]),2)
        self.assertEqual(data["events"][0],{"onset":161.25,"duration":.5,"label":"KeyIn:253"})
        self.assertEqual(data["events"][1]["label"],"中文")

    def test_truncated_records_are_rejected(self):
        p=event_file(self.folder/"evt.bdf",[b"+1\x14KeyIn:253\x14\0"])
        p.write_bytes(p.read_bytes()[:-1])
        with self.assertRaises(ValueError):read_event_file(p)

    def test_header_shift_manual_correction_and_qc_mapping(self):
        parsed={"start_date":"01.01.24","start_time":"08.30.00","events":[
            {"onset":1.,"duration":0.,"label":"A"},{"onset":4.,"duration":0.,"label":"B"}]}
        ref={"start_date":"01.01.24","start_time":"08.30.03","duration_seconds":10,"sampling_rate":1000}
        rows,shift=align_events(parsed,ref,{"kept_epoch_indices":[1,3],"original_epoch_samples":500},250,1)
        self.assertEqual(shift,-2)
        self.assertEqual(rows[0]["本段关系"],"本段外")
        self.assertEqual(rows[1]["原始样本索引_0基"],2000)
        self.assertEqual(rows[1]["有效拼接时间_s"],0)

    def test_duration_overlap_does_not_fabricate_onset_sample(self):
        parsed={"start_date":"01.01.24","start_time":"08.30.00","events":[{"onset":-1.,"duration":3.,"label":"A"}]}
        ref={"start_date":"01.01.24","start_time":"08.30.00","duration_seconds":10,"sampling_rate":1000}
        row=align_events(parsed,ref)[0][0]
        self.assertEqual(row["本段关系"],"区间与本段重叠");self.assertIsNone(row["原始样本索引_0基"])

    def test_mismatched_identity_and_missing_time_are_rejected(self):
        parsed={"start_date":"01.01.24","start_time":"08.30.00","patient":"A","events":[]}
        ref={"start_date":"01.01.24","start_time":"08.30.00","patient":"B","duration_seconds":10,"sampling_rate":1000}
        with self.assertRaises(ValueError):align_events(parsed,ref)
        ref["patient"]="A";ref["start_time"]=""
        with self.assertRaises(ValueError):align_events(parsed,ref)

    def test_all_three_api_modes_with_evt_and_unchecked_compatibility(self):
        previous=server.OUTPUT_ROOT;server.OUTPUT_ROOT=self.folder;self.addCleanup(setattr,server,"OUTPUT_ROOT",previous)
        httpd=ThreadingHTTPServer(("127.0.0.1",0),server.Handler)
        threading.Thread(target=httpd.serve_forever,daemon=True).start()
        self.addCleanup(httpd.server_close);self.addCleanup(httpd.shutdown)
        base="http://127.0.0.1:"+str(httpd.server_port)
        def post(route,body,headers=None):
            request=urllib.request.Request(base+route,data=body,headers=headers or {},method="POST")
            with urllib.request.urlopen(request,timeout=120) as response:return json.load(response)
        marker=event_file(self.folder/"evt.bdf",[b"+2\x14KeyIn:253\x14\0",b"+30\x14KeyIn:253\x14\0"])
        def upload_evt():return post('/api/upload?filename=evt.bdf&kind=evt',marker.read_bytes())["upload_id"]
        eeg=make_example_bdf(self.folder/"data.bdf",seconds=8,fs=200)
        a=post('/api/analyze?filename=data.bdf',eeg.read_bytes(),{'X-EEG-Settings':json.dumps({"evt_upload":upload_evt(),"cache_for_comparison":True})})
        self.assertEqual(a["event_info"]["total_events"],2);self.assertEqual(a["event_info"]["in_segment"],1)
        pair=post('/api/compare',json.dumps({"run_a":a['run_id'],"run_b":a['run_id'],"max_seconds":4}).encode(),{"Content-Type":"application/json"})
        self.assertEqual(pair["event_info_a"]["total_events"],2);self.assertEqual(pair["event_info_b"]["in_segment"],1)
        from test_workbench import marker_file
        spike=self.folder/"spike.bdf";marker_file(spike,[0,1,0,0,0,0,0,0])
        raw=bytearray(spike.read_bytes());raw[8:88]=b"Study-001".ljust(80);spike.write_bytes(raw)
        token=post('/api/upload?filename=spike.bdf&kind=spike',spike.read_bytes())["upload_id"]
        result=post('/api/spike',json.dumps({"spike_upload":token,"evt_upload":upload_evt()}).encode(),{"Content-Type":"application/json"})
        self.assertEqual(result["event_info"]["in_segment"],1)
        self.assertNotIn("recording_header",result)
        with urllib.request.urlopen(base+result['event_info']['files'][0]['url']) as response:self.assertEqual(response.status,200)

if __name__=='__main__':unittest.main()
