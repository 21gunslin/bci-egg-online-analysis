"""Ordered multi-file session: retain clock gaps, never filter across file joins."""
from collections import Counter
import hashlib
import json
from pathlib import Path
from importlib.metadata import version as package_version
from decimal import Decimal

import numpy as np
import pandas as pd
from scipy import signal

from eeg_events import align_events, read_event_file, recording_start
from eeg_feature_engine import (PIPELINE_VERSION, extract_features, plv_matrix,
                                preprocess_eeg, read_eeg_file)
from eeg_pair_spike import feature_domain, plotting


def retained_runs(pre):
    indices = np.asarray(pre['kept_epoch_indices'])
    samples = pre['original_epoch_samples']
    edges = np.r_[0, np.flatnonzero(np.diff(indices) != 1)+1, len(indices)]
    return [(int(a*samples), int(b*samples)) for a,b in zip(edges[:-1],edges[1:]) if b>a]


def weighted_features(series, weights):
    frame = pd.concat(series, axis=1)
    matrix = frame.to_numpy(dtype=float)
    valid = np.isfinite(matrix)
    weights = np.asarray(weights,dtype=float)[None,:]
    numerator = np.where(valid,matrix,0)*weights
    denominator = (valid*weights).sum(axis=1)
    return pd.Series(np.divide(numerator.sum(axis=1),denominator,
        out=np.full(len(frame),np.nan),where=denominator>0),index=frame.index)


def check_order(headers):
    starts = [recording_start(h) for h in headers]
    known = {h.get('patient','').strip() for h in headers} - {'','X','X X X X','unknown','Unknown'}
    if len(known)>1:raise ValueError('所选 data 的文件头受试者标识不同，不能拼为同一次记录。')
    for i in range(1,len(headers)):
        elapsed = (starts[i]-starts[i-1]).total_seconds()
        if elapsed < float(headers[i-1]['duration_seconds'])-1e-6:
            raise ValueError(f'第 {i+1} 份记录时间倒序或与前一份重叠。请调整顺序或核对文件头，系统不会自动剪裁。')
    return [(t-starts[0]).total_seconds() for t in starts]


def analyze_session(items, evt_path, output, settings, evt_offset=0):
    if not 2<=len(items)<=30:raise ValueError('多份分析一次支持 2–30 份 data.bdf。')
    if sum(Path(item['path']).stat().st_size for item in items)>2*1024**3:
        raise ValueError('本组 data 总大小超过 2 GB，请减少份数。')
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    prepared=[];headers=[];hashes=set();working_bytes=0
    for number,item in enumerate(items,1):
        fingerprint=hashlib.sha256()
        with Path(item['path']).open('rb') as handle:
            for chunk in iter(lambda:handle.read(1024**2),b''):fingerprint.update(chunk)
        digest=fingerprint.hexdigest()
        if digest in hashes:raise ValueError(f'第 {number} 份与已有文件完全重复，请移除重复记录。')
        hashes.add(digest)
        data,_,_,header,_=read_eeg_file(item['path'])
        try:pre=preprocess_eeg(data,header,settings)
        except ValueError as error:raise ValueError(f'第 {number} 份质控失败：{error}') from error
        del data
        working_bytes+=pre['data'].nbytes
        if working_bytes>512*1024**2:raise ValueError('本组有效数据超过 512 MB 内存限额，请减少文件或降低目标采样率。')
        prepared.append(pre);headers.append(header.to_dict())
    starts=check_order(headers)
    fs=prepared[0]['fs']
    if any(not np.isclose(p['fs'],fs) for p in prepared):
        raise ValueError('各份实际分析采样率不同，请把统一目标采样率设到不高于最小源采样率。')
    common=[name for name in prepared[0]['channel_names'] if all(name in p['channel_names'] for p in prepared)]
    if len(common)<3:raise ValueError(f'所有记录共同质控保留通道仅 {len(common)} 个；至少需要3个，请减少不兼容的记录。')
    segment_rows=[];run_specs=[];series=[];weights=[];segment_series=[];clean_cursor=0
    for number,(item,pre,header,start) in enumerate(zip(items,prepared,headers,starts),1):
        pre['data']=pre['data'][[pre['channel_names'].index(c) for c in common]]
        pre['channel_names']=common;pre['channel_units']=['µV']*len(common)
        clean_duration=pre['data'].shape[1]/fs
        label=f'第{number}段'
        row={'顺序':number,'标签':label,'文件头开始时间':header['start_date']+' '+header['start_time'],
             '距首段起点_s':start,'原始时长_s':header['duration_seconds'],'原始采样率_Hz':header['sampling_rate'],'共同通道数':len(common),
             '有效时长_s':clean_duration,'有效拼接起点_s':clean_cursor,
             '保留片段数':pre['epoch_count'],'总片段数':pre['epoch_total'],
             '与前段间隔_s':0 if number==1 else start-(starts[number-2]+headers[number-2]['duration_seconds'])}
        segment_rows.append(row)
        current=[];current_weights=[]
        for left,right in retained_runs(pre):
            run=pre['data'][:,left:right]
            if len(run_specs)>=200:raise ValueError('连续保留段超过200段，请减少份数；系统不静默丢弃片段。')
            info={**pre,'data':run,'duration_seconds':run.shape[1]/fs,'keep_ratio':1.0}
            _,summary,_=extract_features(info,settings)
            values=summary.set_index('特征')['数值']
            values=values[values.index.str.contains('__')]
            weight=run.shape[1]/fs
            series.append(values);weights.append(weight);current.append(values);current_weights.append(weight)
            run_specs.append((number-1,left,right,clean_cursor+left/fs))
        segment_series.append(weighted_features(current,current_weights))
        clean_cursor+=clean_duration
    overall=weighted_features(series,weights)
    feature_rows=[]
    for feature,value in overall.items():
        feature_rows.append({'汇总组':'全部记录','特征域':feature_domain(feature),'特征':feature,'数值':float(value)})
    files=[]
    def csv(frame,name):
        path=output/name;frame.to_csv(path,index=False,encoding='utf-8-sig');files.append(str(path))
    csv(pd.DataFrame(segment_rows),'多份_分段顺序与时间映射.csv')
    csv(pd.DataFrame(feature_rows),'多份_四域加权特征.csv')
    csv(pd.DataFrame([{'顺序':i+1,'标签':segment_rows[i]['标签'],'特征':key,'数值':float(value)}
        for i,s in enumerate(segment_series) for key,value in s.items()]),'多份_逐文件特征.csv')
    parsed=read_event_file(evt_path)
    per_file=[align_events(parsed,header,{'kept_epoch_indices':pre['kept_epoch_indices'].tolist(),
        'original_epoch_samples':pre['original_epoch_samples']},fs,evt_offset)[0]
        for header,pre in zip(headers,prepared)]
    shift=(recording_start(parsed)-recording_start(headers[0])).total_seconds()+float(evt_offset)
    span=starts[-1]+headers[-1]['duration_seconds']
    event_rows=[];related=set();gap_events=0
    for event_index,event in enumerate(parsed['events']):
        matches=[i for i,rows in enumerate(per_file) if rows[event_index]['本段关系']!='本段外']
        global_time=float(Decimal(str(event['onset']))+Decimal(str(shift)))
        gap_events+=int(0<=global_time<span and not any(rows[event_index]['本段关系']=='起点在本段内' for rows in per_file))
        if not matches:
            in_span=0<=global_time<span
            event_rows.append({'事件序号':event_index+1,'原始标签':event['label'],'evt原始起点_s':event['onset'],
                '持续时间_s':event['duration'],'距首段起点_s':global_time,'所属分段':None,'分段标签':'',
                '段内起点_s':None,'段内原始样本索引_0基':None,
                '状态':'记录间隙（未采集/未选择）' if in_span else '所选时间范围外',
                '有效拼接时间_s':None})
        for i in matches:
            related.add(event_index);r=per_file[i][event_index]
            event_rows.append({'事件序号':event_index+1,'原始标签':event['label'],'evt原始起点_s':event['onset'],
                '持续时间_s':event['duration'],'距首段起点_s':global_time,'所属分段':i+1,'分段标签':segment_rows[i]['标签'],
                '段内起点_s':r['相对本段起点_s'],'段内原始样本索引_0基':r['原始样本索引_0基'],
                '状态':r['本段关系']+'；'+r['质控状态_起点'],
                '有效拼接时间_s':segment_rows[i]['有效拼接起点_s']+r['有效拼接时间_s'] if r['有效拼接时间_s'] is not None else None})
    event_columns=['事件序号','原始标签','evt原始起点_s','持续时间_s','距首段起点_s','所属分段','分段标签','段内起点_s','段内原始样本索引_0基','状态','有效拼接时间_s']
    csv(pd.DataFrame(event_rows,columns=event_columns),'多份_同一evt事件映射.csv')
    warnings=['按你给定的顺序分析，并核对文件头时间；真实间隔保留在时间映射中，不补零、不自动剪裁重叠。',
        '滤波和特征在每份记录的各连续保留段内独立计算；总体为按有效时长加权的特征均值，不是跨边界直接计算的连续信号统计。',
        '使用全部记录共同质控保留通道；保留采集参考，文件头单位误标和时钟同步仍需核实。',
        '第N段仅表示你确认的时间顺序，没有状态分类；evt 原始编码含义未知，不推断宫缩/镇痛，也不自动配对事件。',
        '间隙中的事件不会移入任一 data；有效拼接时间是显示/索引坐标，不是原始连续记录，也不生成伪连续 BDF。',
        '文件头通常为秒级精度；短连续段的低频、相位和网络指标需谨慎解释。']
    event_info={'total_events':len(parsed['events']),'in_segment':len(related),
        'outside_segment':len(parsed['events'])-len(related),'alignment_shift_seconds':shift,
        'rows':event_rows[:1000],'table_truncated':len(event_rows)>1000,
        'labels':[{'原始标签':k,'次数':v} for k,v in Counter(e['label'] for e in parsed['events']).items()],
        'warnings':[f'有 {gap_events} 个事件起点落在所选记录的时间间隙中。',*warnings[-3:]],'files':[]}
    plt=plotting()
    def save(fig,name):
        path=output/name;fig.savefig(path,dpi=145,bbox_inches='tight');plt.close(fig);files.append(str(path))
    fig,ax=plt.subplots(figsize=(13,4.5),constrained_layout=True)
    for i,row in enumerate(segment_rows):
        ax.broken_barh([(row['距首段起点_s'],row['原始时长_s'])],(i-.3,.6),facecolors=f'C{i%10}',alpha=.65)
    for event in parsed['events']:
        ax.axvline(event['onset']+shift,color='#c94040',alpha=.45,lw=.8)
    ax.set_yticks(range(len(items)),[f"{r['顺序']} · {r['标签']}" for r in segment_rows])
    ax.set(xlabel='距首段文件头起点的原始时间 (s)，空白保留真实间隔',title='多份 data 原始时间覆盖与同一 evt 事件（红线）');ax.grid(axis='x',alpha=.2)
    save(fig,'多份_原始时间覆盖与evt.png')
    fig,ax=plt.subplots(figsize=(13,4.5),constrained_layout=True)
    for i,left,right,clean_start in run_specs:
        x=prepared[i]['data'][0,left:right];step=max(1,len(x)//12000)
        ax.plot(clean_start+np.arange(0,len(x),step)/fs,x[::step],lw=.6,color=f'C{i%10}')
    for row in segment_rows[1:]:ax.axvline(row['有效拼接起点_s'],color='red',ls='--',alpha=.5)
    ax.set(xlabel='有效片段拼接时间 (s)，文件边界不跨越计算',ylabel='µV',title=f'共同通道 {common[0]}：独立连续段波形缩略显示')
    save(fig,'多份_有效波形与文件边界.png')
    nperseg=min(1024,min(right-left for _,left,right,_ in run_specs))
    spectra=[];spectrum_weights=[];transforms=[]
    for i,left,right,start in run_specs:
        x=prepared[i]['data'][:,left:right]
        freq,power=signal.welch(x,fs=fs,nperseg=nperseg,axis=1)
        spectra.append(power.mean(axis=0));spectrum_weights.append((right-left)/fs)
        size=min(512,right-left)
        f,t,z=signal.stft(x[0],fs=fs,nperseg=size,noverlap=size*3//4,boundary=None,padded=False)
        transforms.append((f,t+start,10*np.log10(np.abs(z)**2+1e-20)))
    fig,ax=plt.subplots(figsize=(12,4),constrained_layout=True)
    for spectrum in spectra:ax.semilogy(freq,np.maximum(spectrum,1e-20),alpha=.2,lw=.7)
    ax.semilogy(freq,np.maximum(np.average(spectra,axis=0,weights=spectrum_weights),1e-20),color='black',lw=1.8,label='按有效时长加权PSD')
    ax.set(xlim=(.5,45),xlabel='Hz',ylabel='µV²/Hz',title='逐连续段功率谱与加权汇总');ax.legend();save(fig,'多份_独立功率谱与加权汇总.png')
    visible=np.concatenate([p[f<=45].ravel() for f,_,p in transforms]);lo,hi=np.percentile(visible,[2,98]);hi=max(hi,lo+1)
    fig,ax=plt.subplots(figsize=(13,4.5),constrained_layout=True)
    for f,t,power in transforms:mesh=ax.pcolormesh(t,f,power,shading='auto',cmap='magma',vmin=lo,vmax=hi)
    for row in segment_rows[1:]:ax.axvline(row['有效拼接起点_s'],color='cyan',ls='--',lw=.8)
    ax.set(ylim=(.5,45),xlabel='有效片段拼接时间 (s)，各连续段独立STFT',ylabel='Hz',title=f'{common[0]} 独立时频图，统一色标')
    fig.colorbar(mesh,ax=ax,label='功率 (dB)');save(fig,'多份_独立时频与文件边界.png')
    matrices=[plv_matrix(prepared[i]['data'][:,left:right],fs,8,13) for i,left,right,_ in run_specs]
    matrix=np.average(matrices,axis=0,weights=spectrum_weights)
    fig,ax=plt.subplots(figsize=(7,6),constrained_layout=True)
    image=ax.imshow(matrix,vmin=0,vmax=1,cmap='viridis');ax.set_xticks(range(len(common)),common,rotation=90);ax.set_yticks(range(len(common)),common)
    ax.set_title('α PLV：逐连续段矩阵按时长加权（非跨段相位）');fig.colorbar(image,ax=ax);save(fig,'多份_独立PLV加权矩阵.png')
    report=output/'多份_处理与时间映射说明.txt'
    report.write_text('\n'.join(warnings)+'\n\n'+json.dumps({'pipeline_version':PIPELINE_VERSION,
        'common_channels':common,'analysis_fs':fs,'continuous_runs':len(run_specs),
        'settings':settings.__dict__,'software':{name:package_version(name) for name in ('numpy','scipy','pandas','matplotlib')},'segments':segment_rows},ensure_ascii=False,indent=2),encoding='utf8');files.append(str(report))
    return {'kind':'session','metrics':{'files':len(items),'common_channels':len(common),'sampling_rate':fs,
        'recorded_seconds':sum(h['duration_seconds'] for h in headers),'span_seconds':span,
        'effective_seconds':clean_cursor,'continuous_runs':len(run_specs)},
        'segments':segment_rows,'common_labels':common,'rows':feature_rows[:2000],
        'table_truncated':len(feature_rows)>2000,'event_info':event_info,'warnings':warnings,'files':files}
