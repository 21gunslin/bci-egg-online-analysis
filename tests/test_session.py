"""Ordered multi-BDF session checks; synthetic recordings only."""
import json
from pathlib import Path
import shutil
import sys
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"backend"))
from eeg_session import check_order,retained_runs,weighted_features
from eeg_feature_engine import make_example_bdf
from test_events import event_file
import eeg_web_server as server

class SessionTests(unittest.TestCase):
    def test_order_and_real_gaps(self):
        a={"start_date":"01.01.24","start_time":"08.30.00","duration_seconds":8,"patient":"A"}
        b={**a,"start_time":"08.30.12"}
        self.assertEqual(check_order([a,b]),[0,12])
        with self.assertRaises(ValueError):check_order([b,a])
        with self.assertRaises(ValueError):check_order([a,{**a,"start_time":"08.30.06"}])
        with self.assertRaises(ValueError):check_order([a,{**b,"patient":"B"}])

    def test_split_retained_data_at_original_rejection_gaps(self):
        pre={"kept_epoch_indices":np.array([0,1,4]),"original_epoch_samples":400}
        self.assertEqual(retained_runs(pre),[(0,800),(800,1200)])

    def test_weighted_missing_features_use_valid_denominator(self):
        a=pd.Series({"x":2.,"y":np.nan,"z":np.nan})
        b=pd.Series({"x":6.,"y":5.,"z":np.nan})
        r=weighted_features([a,b],[1,3])
        self.assertEqual(r['x'],5);self.assertEqual(r['y'],5);self.assertTrue(np.isnan(r['z']))

    def test_session_api_maps_events_to_segments_gaps_and_outside(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);previous=server.OUTPUT_ROOT;server.OUTPUT_ROOT=root
            self.addCleanup(setattr,server,"OUTPUT_ROOT",previous)
            httpd=ThreadingHTTPServer(("127.0.0.1",0),server.Handler)
            threading.Thread(target=httpd.serve_forever,daemon=True).start()
            self.addCleanup(httpd.server_close);self.addCleanup(httpd.shutdown)
            base="http://127.0.0.1:"+str(httpd.server_port)
            def post(route,body,headers=None):
                req=urllib.request.Request(base+route,data=body,headers=headers or {},method="POST")
                with urllib.request.urlopen(req,timeout=180) as response:return json.load(response)
            a=make_example_bdf(root/'a.bdf',seconds=8,fs=200)
            b=root/'b.bdf';shutil.copy2(a,b);raw=bytearray(b.read_bytes());raw[176:184]=b'08.30.12';b.write_bytes(raw)
            evt=event_file(root/'evt.bdf',[b'+2\x14KeyIn:253\x14\0',b'+10\x14KeyIn:253\x14\0',b'+13\x14KeyIn:253\x14\0',b'+30\x14KeyIn:253\x14\0'])
            def upload(path,kind):return post('/api/upload?filename='+path.name+'&kind='+kind,path.read_bytes())['upload_id']
            def request(reverse=False):
                paths=[b,a] if reverse else [a,b]
                opts={'items':[{'upload_id':upload(p,'eeg')} for p in paths],'evt_upload':upload(evt,'evt'),'settings':{'target_fs':200}}
                return post('/api/session',json.dumps(opts).encode(),{'Content-Type':'application/json'})
            with self.assertRaises(urllib.error.HTTPError) as caught:request(True)
            self.assertEqual(caught.exception.code,400)
            result=request()
            self.assertEqual(result['metrics']['files'],2)
            self.assertEqual(result['metrics']['recorded_seconds'],16)
            self.assertEqual(result['metrics']['span_seconds'],20)
            self.assertEqual(result['metrics']['continuous_runs'],2)
            self.assertEqual(result['segments'][1]['与前段间隔_s'],4)
            self.assertEqual(result['segments'][0]['标签'],'第1段')
            info=result['event_info'];self.assertEqual(info['total_events'],4);self.assertEqual(info['in_segment'],2)
            rows=info['rows']
            self.assertEqual(rows[1]['状态'],'记录间隙（未采集/未选择）')
            self.assertIsNone(rows[1]['有效拼接时间_s'])
            self.assertEqual(rows[2]['所属分段'],2)
            self.assertEqual(rows[2]['段内原始样本索引_0基'],200)
            self.assertEqual(rows[2]['有效拼接时间_s'],9)
            self.assertEqual(rows[3]['状态'],'所选时间范围外')
            self.assertTrue(result['rows'])
            for item in result['files']:
                with urllib.request.urlopen(base+item['url']) as response:self.assertEqual(response.status,200)

if __name__=='__main__':unittest.main()
