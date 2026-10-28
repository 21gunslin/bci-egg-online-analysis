"""Regression checks for units, timing and mathematical feature definitions."""
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"backend"))
from eeg_feature_engine import (AnalysisSettings, _EDFReader, _choose_channels, _frequency_features,
    _windowed_features, clean_epochs, make_example_bdf, plv_matrix, preprocess_eeg, read_eeg_file,
    voltage_to_uv)

class EngineContracts(unittest.TestCase):
    def test_declared_voltage_conversion_and_unknown_rejection(self):
        values=np.ones((4,10))
        np.testing.assert_allclose(voltage_to_uv(values,["V","mV","uV","nV"])[:,0],[1e6,1e3,1,1e-3])
        with self.assertRaises(ValueError): voltage_to_uv(values,["pcs"]*4)

    def test_no_automatic_fallback_to_marker_or_reference_channels(self):
        self.assertEqual(_choose_channels(["spike","Status","A1","EHG"],["pcs","","uV","uV"],True)[0],[])
        self.assertEqual(_choose_channels(["EEG F3-REF","F4","Cz"],["uV"]*3,True)[0],[0,1,2])

    def test_actual_sampling_rate_controls_epoch_length(self):
        settings=AnalysisSettings(target_fs=250,epoch_seconds=2)
        epochs,mask=clean_epochs(np.zeros((3,1000)),settings,fs=200)
        self.assertEqual(epochs.shape,(3,2,400))
        short,mask=clean_epochs(np.zeros((3,20)),settings,fs=200)
        self.assertEqual(short.shape,(3,0,400))

    def test_scientific_notation_record_duration(self):
        self.assertEqual(_EDFReader._parse_float_field(b"1E-1    ",1),.1)

    def test_power_is_integrated_psd(self):
        fs=256;t=np.arange(2048)/fs;x=(10*np.sin(2*np.pi*10*t))[None,:]
        features=_frequency_features(x,fs,AnalysisSettings().bands)
        self.assertAlmostEqual(features["alpha_absolute_power"][0],50,places=6)
        self.assertAlmostEqual(features["total_power"][0],50,places=6)

    def test_hjorth_complexity_includes_mobility_denominator(self):
        x=np.random.default_rng(17).normal(0,3,(3,2000))
        dx=np.diff(x,axis=1);ddx=np.diff(dx,axis=1)
        expected=np.sqrt(np.var(ddx,axis=1)/np.var(dx,axis=1))/np.sqrt(np.var(dx,axis=1)/np.var(x,axis=1))
        np.testing.assert_allclose(_windowed_features(x)["hjorth_complexity"],expected,rtol=1e-10)

    def test_plv_phase_sampling_approximates_full_result(self):
        x=np.random.default_rng(9).normal(size=(3,20000))
        full=plv_matrix(x,250,30,45,max_samples=20000)
        capped=plv_matrix(x,250,30,45,max_samples=1000)
        np.testing.assert_allclose(capped,full,atol=.035)

    def test_preprocess_voltage_equivalence_and_qc_channel_labels(self):
        with tempfile.TemporaryDirectory() as folder:
            path=make_example_bdf(Path(folder)/"example.bdf",seconds=8,fs=200)
            _,_,_,header,_=read_eeg_file(path)
        x=np.vstack([np.zeros(1600),np.random.default_rng(19).normal(0,10,(3,1600))])
        labels=["Fp1","F3","F4","Cz"]
        hu=replace(header,channel_labels=labels,channel_units=["uV"]*4,sampling_rate=200)
        hv=replace(hu,channel_units=["V"]*4)
        settings=AnalysisSettings(target_fs=250,notches=())
        a=preprocess_eeg(x,hu,settings);b=preprocess_eeg(x*1e-6,hv,settings)
        np.testing.assert_allclose(a["data"],b["data"],atol=1e-10)
        self.assertEqual(a["channel_names"],["F3","F4","Cz"])
        self.assertEqual(a["channel_units"],["µV"]*3)
        kept=[r["通道"] for r in a["channel_records"] if r["状态"]=="保留"]
        self.assertEqual(kept,a["channel_names"])
        self.assertEqual(a["epoch_total"],4)

if __name__=="__main__": unittest.main()
