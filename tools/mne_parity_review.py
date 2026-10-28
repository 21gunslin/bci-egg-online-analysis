"""MNE public-API parity checks. Outputs aggregate measurements, never patient metadata."""
import argparse
import json
import platform
import sys
import tempfile
from pathlib import Path
import numpy as np
from scipy import signal
import mne

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'backend'))
from eeg_feature_engine import (_bandpass, _frequency_features, AnalysisSettings, clean_epochs,
    make_example_bdf, read_eeg_file, voltage_to_uv)


def compare_reader(path, name):
    custom, labels, units, header, _=read_eeg_file(path)
    raw=mne.io.read_raw_bdf(path,preload=False,verbose='ERROR')
    try:
        assert raw.info['sfreq']==header.sampling_rate
        assert raw.n_times==custom.shape[1]
        reference=raw.get_data(picks=labels)*1e6
        converted=voltage_to_uv(custom,units)
        error=float(np.max(np.abs(reference-converted)))
        np.testing.assert_allclose(reference,converted,rtol=1e-11,atol=1e-8)
        return {'case':name,'sampling_rate':header.sampling_rate,'samples':custom.shape[1],
            'matched_channels':len(labels),'max_absolute_error_uv':error,'status':'passed'}
    finally:
        raw.close()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--real-data-root',type=Path)
    parser.add_argument('--output',type=Path,default=ROOT/'results'/'mne_review'/'parity.json')
    args=parser.parse_args()
    result={'versions':{'python':platform.python_version(),'mne':mne.__version__,'numpy':np.__version__},'readers':[]}
    with tempfile.TemporaryDirectory() as folder:
        example=make_example_bdf(Path(folder)/'synthetic.bdf',seconds=12,fs=500)
        result['readers'].append(compare_reader(example,'synthetic'))
    if args.real_data_root:
        candidates=[p for p in args.real_data_root.rglob('data.bdf') if p.stat().st_size<50*1024**2]
        if not candidates: raise ValueError('No small data.bdf files found')
        chosen=sorted({0,len(candidates)//2,len(candidates)-1})
        for number,index in enumerate(chosen,1):
            result['readers'].append(compare_reader(candidates[index],f'real_recording_{number}'))
    fs=250
    x=np.random.default_rng(32).normal(0,10,(3,5000))
    custom=_bandpass(x,fs,.5,45)
    reference=mne.filter.filter_data(x,fs,.5,45,method='iir',
        iir_params={'order':4,'ftype':'butter','output':'sos','padlen':27},phase='zero',verbose='ERROR')
    np.testing.assert_allclose(custom,reference,rtol=1e-9,atol=1e-9)
    result['bandpass']={'max_absolute_error_uv':float(np.max(np.abs(custom-reference))),
        'padding':'matched 27-sample odd reflection','status':'passed'}
    freq,psd=signal.welch(x,fs=fs,nperseg=1024,axis=1)
    mp,mf=mne.time_frequency.psd_array_welch(x,sfreq=fs,n_fft=1024,n_per_seg=1024,
        n_overlap=512,window='hann',average='mean',remove_dc=True,verbose='ERROR')
    np.testing.assert_allclose(freq,mf)
    np.testing.assert_allclose(psd,mp,rtol=1e-11,atol=1e-11)
    features=_frequency_features(x,fs,AnalysisSettings().bands)
    mask=(mf>=8)&(mf<13)
    np.testing.assert_allclose(features['alpha_absolute_power'],mp[:,mask].sum(axis=1)*(mf[1]-mf[0]))
    result['welch']={'max_absolute_error':float(np.max(np.abs(psd-mp))),'integrated_alpha_power':'matched','status':'passed'}
    eeg=np.random.default_rng(33).normal(0,10,(3,1000))
    raw=mne.io.RawArray(eeg*1e-6,mne.create_info(['F3','F4','Cz'],200,'eeg'),verbose='ERROR')
    epochs=mne.make_fixed_length_epochs(raw,duration=2,preload=True,verbose='ERROR')
    a,mask=clean_epochs(eeg,AnalysisSettings(target_fs=250),fs=200)
    np.testing.assert_allclose(a,np.transpose(epochs.get_data(copy=True),(1,0,2))*1e6)
    result['epochs']={'actual_fs':200,'requested_fs':250,'epoch_samples':a.shape[-1],'epoch_count':a.shape[1],'status':'passed'}
    result['limitations']=['Parity follows declared file units, not independently verified device calibration.',
        'Matching IIR padding is required; default MNE padding can differ.',
        'This checks selected BDF recordings and synthetic cases, not every record, GDF file or clinical validity.']
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
