"""Metadata schemas, privacy allow-list and end-to-end session integration."""
import json
from pathlib import Path
import shutil
import sys
import tempfile
import threading
import unittest
import urllib.request
from datetime import datetime,timezone,timedelta
from http.server import ThreadingHTTPServer
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"backend"))
from eeg_metadata import decode_metadata,parse_datainfo,parse_recordinfo,match_datainfo,select_record_channels
from eeg_feature_engine import make_example_bdf
from test_events import event_file
import eeg_web_server as server

SECRET='SENSITIVE_PASSWORD_X'
def datainfo_fixture():
    return {'count':3,'duration':20,'sampleRate':200,'channelCount':12,
        'startTime':datetime(2024,1,1,8,30,tzinfo=timezone(timedelta(hours=8))).timestamp(),
        'data1':{'duration':8,'durationFromBegin':8},'data2':{'duration':0,'durationFromBegin':12},
        'data3':{'duration':8,'durationFromBegin':20},'PatientID':'DO_NOT_EXPORT'}

def record_fixture():
    labels=['Fp1','Fp2','F3','F4','C3','C4','Cz','P3','P4','O1','O2']
    channels=[{'Name':name,'ChannelType':'EEG','IndexInAllChannels':99-i} for i,name in enumerate(labels)]
    return {'WorkChannel':{'EEGWorkChannel':{'Channels':channels},
        'DifferenceWorkChannel':{'Channels':[{'Name':'Status','ChannelType':'STIM','IndexInAllChannels':0}]}},
        'CameraSetting':{'Cameras':[{'User':'PRIVATE_USER','Pwd':SECRET,'Ip':'PRIVATE_IP'}]},'FullName':'DO_NOT_EXPORT'}

class MetadataTests(unittest.TestCase):
    def test_datainfo_end_positions_zero_entries_and_privacy(self):
        data=parse_datainfo(datainfo_fixture())
        self.assertEqual(data['waveform_count'],2);self.assertEqual(data['entry_count'],3)
        self.assertEqual(data['segments'][2]['起点_s'],12)
        self.assertEqual(data['recorded_seconds'],16);self.assertEqual(data['gap_seconds'],4)
        self.assertNotIn('PatientID',json.dumps(data))

    def test_malformed_count_timestamp_and_overlap_rejected(self):
        d=datainfo_fixture();d['count']=4
        with self.assertRaises(ValueError):parse_datainfo(d)
        d=datainfo_fixture();d['startTime']*=1000
        with self.assertRaises(ValueError):parse_datainfo(d)
        d=datainfo_fixture();d['data3']['durationFromBegin']=15
        with self.assertRaises(ValueError):parse_datainfo(d)

    def test_recordinfo_allowlist_and_conflicting_types(self):
        parsed=parse_recordinfo(record_fixture());text=json.dumps(parsed)
        for word in [SECRET,'CameraSetting','PRIVATE_USER','PRIVATE_IP','DO_NOT_EXPORT','IndexInAllChannels']:
            # Warnings mention the index field only to explain it is not used.
            if word!='IndexInAllChannels':self.assertNotIn(word,text)
        self.assertEqual(parsed['type_counts']['EEG'],11)
        d=record_fixture();d['WorkChannel']['DifferenceWorkChannel']['Channels'].append({'Name':'F3','ChannelType':'EMG'})
        with self.assertRaises(ValueError):parse_recordinfo(d)

    def test_type_selection_uses_names_not_hardware_indices(self):
        d=record_fixture();d['WorkChannel']['EEGWorkChannel']['Channels'] += [{'Name':'A1','ChannelType':'EEG'},{'Name':'PG1','ChannelType':'EEG'}]
        d['WorkChannel']['DifferenceWorkChannel']['Channels'].append({'Name':'X1','ChannelType':'EMG'})
        indices,rows=select_record_channels(parse_recordinfo(d),['F4','F3','Cz','X1','A1','PG1'],['uV']*6,1)
        self.assertEqual(indices,[0,1,2]);self.assertEqual(rows[3]['配置类型'],'EMG')
        self.assertEqual(rows[5]['分析选择'],'EEG配置但默认命名规则未纳入')

    def test_one_second_discrepancy_is_reported_not_overwritten(self):
        d=parse_datainfo(datainfo_fixture());header={'start_date':'01.01.24','start_time':'08.30.13','duration_seconds':8,'sampling_rate':200}
        result=match_datainfo(d,header,set())
        self.assertEqual(result['JSON编号'],3);self.assertEqual(result['起点差_BDF减JSON_s'],1)
        self.assertEqual(header['start_time'],'08.30.13')
        header['start_time']='08.30.15'
        with self.assertRaises(ValueError):match_datainfo(d,header,set())

    def test_bom_and_invalid_json(self):
        d=datainfo_fixture()
        self.assertEqual(decode_metadata(json.dumps(d).encode('utf-16'),'datainfo')['waveform_count'],2)
        with self.assertRaises(ValueError):decode_metadata(b'not json','datainfo')

    def test_preview_and_session_api_never_save_raw_credentials(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);previous=server.OUTPUT_ROOT;server.OUTPUT_ROOT=root
            self.addCleanup(setattr,server,'OUTPUT_ROOT',previous)
            httpd=ThreadingHTTPServer(('127.0.0.1',0),server.Handler)
            threading.Thread(target=httpd.serve_forever,daemon=True).start()
            self.addCleanup(httpd.server_close);self.addCleanup(httpd.shutdown)
            base='http://127.0.0.1:'+str(httpd.server_port)
            def post(route,body,headers=None):
                request=urllib.request.Request(base+route,data=body,headers=headers or {},method='POST')
                with urllib.request.urlopen(request,timeout=180) as response:return json.load(response)
            raw=json.dumps(record_fixture()).encode()
            preview=post('/api/upload?filename=RecordInfo.json&kind=recordinfo&preview=1',raw)
            self.assertNotIn(SECRET,json.dumps(preview));self.assertFalse((root/'_uploads').exists())
            record=post('/api/upload?filename=RecordInfo.json&kind=recordinfo',raw)
            token=root/'_uploads'/(record['upload_id']+'.json')
            self.assertNotIn(SECRET,token.read_text(encoding='utf8'))
            di=post('/api/upload?filename=datainfo.json&kind=datainfo',json.dumps(datainfo_fixture()).encode())
            a=make_example_bdf(root/'a.bdf',seconds=8,fs=200);b=root/'b.bdf';shutil.copy2(a,b)
            content=bytearray(b.read_bytes());content[176:184]=b'08.30.13';b.write_bytes(content)
            evt=event_file(root/'evt.bdf',[b'+2\x14KeyIn:253\x14\0',b'+14\x14KeyIn:253\x14\0'])
            def upload(path,kind):return post('/api/upload?filename='+path.name+'&kind='+kind,path.read_bytes())['upload_id']
            options={'items':[{'upload_id':upload(p,'eeg')} for p in [a,b]],'evt_upload':upload(evt,'evt'),
                'datainfo_upload':di['upload_id'],'recordinfo_upload':record['upload_id'],'settings':{'target_fs':200}}
            result=post('/api/session',json.dumps(options).encode(),{'Content-Type':'application/json'})
            self.assertEqual(result['metrics']['span_seconds'],21)
            self.assertEqual(result['metadata_info']['time_checks'][1]['起点差_BDF减JSON_s'],1)
            self.assertEqual(result['metadata_info']['recordinfo']['type_counts']['EEG'],11)
            self.assertTrue(result['metadata_info']['channel_checks'])
            self.assertNotIn(SECRET,json.dumps(result))
            self.assertFalse(list((root/'_uploads').glob('*.json')))
            self.assertTrue(any('通道类型核对' in f['name'] for f in result['files']))
            self.assertTrue(any('JSON分段计划' in f['name'] for f in result['files']))

if __name__=='__main__':unittest.main()
