"""本机配对 EEG 对比与独立 spike 计数/候选波形复核。

棘波模块为透明的启发式研究工具，不进行癫痫诊断或脑源定位。
"""
from __future__ import annotations

import json
import re
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import signal
from scipy.integrate import trapezoid
from eeg_feature_engine import AnalysisSettings, extract_features, plv_matrix, read_eeg_file, _downsample


def plotting():
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt, font_manager
    for name in ('Microsoft YaHei', 'SimHei', 'Noto Sans CJK SC'):
        try:
            font_manager.findfont(name, fallback_to_default=False)
            plt.rcParams['font.sans-serif'] = [name]
            break
        except ValueError:
            pass
    plt.rcParams['axes.unicode_minus'] = False
    return plt


def feature_domain(name):
    return next((x for x in ('时域', '频域', '时频域', '网络') if name.startswith(x + '__')), '汇总')


def compare_cached(a, b, output, label_a='镇痛前', label_b='镇痛后', channel=None, max_seconds=180):
    """相同通道、采样率及有效时长下重算特征，避免比较不同图的尺度。"""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    za, zb = np.load(a, allow_pickle=False), np.load(b, allow_pickle=False)
    try:
        fa, fb = float(za['fs']), float(zb['fs'])
        if not np.isclose(fa, fb):
            raise ValueError('两份记录的分析采样率不同，请使用相同设置重新分析。')
        la, lb = za['labels'].tolist(), zb['labels'].tolist()
        common = [x for x in la if x in lb]
        if len(common) < 3:
            raise ValueError('两份记录共有的质控保留通道不足 3 个，无法进行一致的四域比较。')
        if channel and channel not in common:
            raise ValueError('指定的时频通道不在两份记录的共同保留通道中。')
        selected = channel or common[0]
        n = min(za['data'].shape[1], zb['data'].shape[1], int(max_seconds * fa))
        if n < fa * 2:
            raise ValueError('共同可用时长不足 2 秒。')
        arrays = [z['data'][[labels.index(x) for x in common], :n] for z, labels in ((za, la), (zb, lb))]
        factors = {'v':1e6,'mv':1e3,'nv':.001,'uv':1,'µv':1,'μv':1,'microvolt':1,'microvolts':1}
        for arr,z,labels in zip(arrays,(za,zb),(la,lb)):
            units=z['units'].tolist()
            for i,label in enumerate(common):
                unit=units[labels.index(label)].strip().lower()
                if unit not in factors: raise ValueError(f'通道 {label} 的电压单位无法解释，不能比较幅值。')
                arr[i] *= factors[unit]
    finally:
        za.close(); zb.close()
    fs = fa
    settings = AnalysisSettings(target_fs=fs)
    summaries = []
    for x in arrays:
        _, summary, _ = extract_features({'data': x, 'fs': fs, 'channel_names': common,
            'channel_units': ['µV'] * len(common), 'duration_seconds': n / fs,
            'keep_ratio': 1.0, 'epoch_count': int(n // (2 * fs)), 'epoch_total': int(n // (2 * fs))}, settings)
        summaries.append(summary.set_index('特征')['数值'])
    merged = pd.concat(summaries, axis=1, keys=['记录A', '记录B']).dropna()
    merged = merged[merged.index.str.contains('__')].copy()
    merged['变化量_B减A'] = merged['记录B'] - merged['记录A']
    merged['相对变化_pct'] = np.where(merged['记录A'].abs() > 1e-8, merged['变化量_B减A'] / merged['记录A'].abs() * 100, np.nan)
    merged.insert(0, '特征域', [feature_domain(x) for x in merged.index])
    merged.index.name = '特征'
    rows = merged.reset_index()
    csv = output / '共同通道_配对特征变化.csv'
    rows.to_csv(csv, index=False, encoding='utf-8-sig')
    plt = plotting(); files = [csv]
    def save(fig, name):
        p = output / name; fig.savefig(p, dpi=150, bbox_inches='tight'); plt.close(fig); files.append(p)
    idx = common.index(selected)
    fig, axes = plt.subplots(2, 1, figsize=(12, 5), sharex=True, sharey=True, constrained_layout=True)
    count = min(n, int(fs * 5))
    for ax, x, title in zip(axes, arrays, (label_a, label_b)):
        ax.plot(np.arange(count) / fs, x[idx, :count], lw=.8)
        ax.set_title(f'{title} · {selected}'); ax.set_ylabel('µV'); ax.grid(alpha=.2)
    axes[-1].set_xlabel('有效片段拼接时间 (s)')
    save(fig, '时域_同通道同尺度.png')
    bands = settings.bands
    psds = [signal.welch(x, fs=fs, nperseg=min(1024, n), axis=1) for x in arrays]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    rels = []
    for (f, psd), title in zip(psds, (label_a, label_b)):
        mean = psd.mean(axis=0)
        axes[0].semilogy(f, np.maximum(mean, 1e-20), label=title)
        total = trapezoid(psd[:, (f >= .5) & (f <= 45)], f[(f >= .5) & (f <= 45)], axis=1)
        rels.append([float(np.mean(trapezoid(psd[:, (f >= lo) & (f < hi)], f[(f >= lo) & (f < hi)], axis=1) / np.maximum(total, 1e-20)) * 100) for lo, hi in bands.values()])
    axes[0].set(xlim=(0, 45), xlabel='频率 (Hz)', ylabel='平均 PSD (µV²/Hz)', title='共同通道平均功率谱'); axes[0].legend()
    xx = np.arange(len(bands))
    axes[1].bar(xx - .18, rels[0], .36, label=label_a); axes[1].bar(xx + .18, rels[1], .36, label=label_b)
    axes[1].set_xticks(xx, list(bands)); axes[1].set(ylabel='相对功率 (%)', title='共同通道平均相对频段功率'); axes[1].legend()
    save(fig, '频域_共同通道.png')
    stfts = [signal.stft(x[idx], fs=fs, nperseg=min(512, n), noverlap=min(384, n // 2), boundary=None, padded=False) for x in arrays]
    powers = [10 * np.log10(np.abs(z[2]) ** 2 + 1e-20) for z in stfts]
    visible = np.concatenate([pow_[z[0] <= 45].ravel() for z, pow_ in zip(stfts, powers)])
    lo, hi = np.percentile(visible, [2, 98]); hi = max(hi, lo + 1)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharex=True, sharey=True, constrained_layout=True)
    for ax, (f, t, _), pow_, title in zip(axes, stfts, powers, (label_a, label_b)):
        mesh = ax.pcolormesh(t, f, pow_, shading='auto', cmap='magma', vmin=lo, vmax=hi)
        ax.set(ylim=(0, 45), xlabel='有效片段拼接时间 (s)', ylabel='Hz', title=f'{title} · {selected}')
    fig.colorbar(mesh, ax=axes, label='功率 (dB)，统一色标')
    save(fig, '时频_固定通道统一色标.png')
    for band, (low, high) in bands.items():
        matrices = [plv_matrix(x, fs, low, high) for x in arrays]
        diff = matrices[1] - matrices[0]
        fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
        for ax, mat, title, limits, cmap in zip(axes, [*matrices, diff], [label_a, label_b, 'B − A'], [(0, 1), (0, 1), (-1, 1)], ['viridis', 'viridis', 'RdBu_r']):
            im = ax.imshow(mat, vmin=limits[0], vmax=limits[1], cmap=cmap)
            ax.set_xticks(range(len(common)), common, rotation=90, fontsize=7); ax.set_yticks(range(len(common)), common, fontsize=7)
            ax.set_title(f'{band} PLV · {title}'); fig.colorbar(im, ax=ax, shrink=.8)
        save(fig, f'网络_{band}_同通道及差值.png')
    note = output / '配对对比说明.txt'
    note.write_text(f'A：{label_a}\nB：{label_b}\n共同通道：{", ".join(common)}\n统一采样率：{fs:g} Hz\n每份使用有效数据：{n/fs:g} 秒\n时频通道：{selected}\n网络密度：0.25\n'
        '仅使用两次均质控保留的通道，并截取相同长度的有效片段拼接信号重新计算四域特征。\n'
        '时间轴为各记录的有效片段拼接时间，不能用于定位原始记录时刻或宫缩起止。\n'
        'B−A 是个体内描述性变化，不提供 P 值、因果结论或 PPD 风险判定。\n'
        '相对变化分母为 |A|，A 接近零时留空。γ 30–45 Hz 接近滤波上限，需审慎解释。\n', encoding='utf-8')
    files.append(note)
    return {'kind': 'comparison', 'metrics': {'common_channels': len(common), 'duration_seconds': n / fs, 'sampling_rate': fs, 'channel': selected},
        'labels': common, 'label_a': label_a, 'label_b': label_b, 'rows': rows.to_dict('records'), 'files': [str(p) for p in files]}


def read_spike_counts(path):
    raw = Path(path).read_bytes()
    if len(raw) < 512:
        raise ValueError('spike.bdf 文件过短。')
    n = int(raw[252:256]); hb = int(raw[184:192]); duration = float(raw[244:252]); nr = int(raw[236:244])
    sh = raw[256:hb]
    label = sh[:16].decode('latin1').strip()
    ns = int(sh[216*n:216*n+8])
    unit = sh[96*n:96*n+8].decode('latin1').strip()
    if n != 1 or label.lower() != 'spike' or ns != 1 or not np.isclose(duration, 1):
        raise ValueError('本模块支持文件夹中的单通道 spike、1 Hz 计数型 BDF；此文件结构不同，不能猜测其事件编码。')
    body = np.frombuffer(raw[hb:], dtype=np.uint8)
    count = len(body) // 3 if nr < 0 else min(nr, len(body) // 3)
    if not count:
        raise ValueError('spike 文件中没有完整数据记录。')
    bb = body[:count*3].reshape(-1, 3)
    vals = bb[:, 0].astype(np.int32) | (bb[:, 1].astype(np.int32) << 8) | (bb[:, 2].astype(np.int32) << 16)
    vals = np.where(vals & 0x800000, vals - 0x1000000, vals).astype(float)
    def field(offset): return float(sh[offset*n:offset*n+8])
    pmin, pmax, dmin, dmax = (field(o) for o in (104, 112, 120, 128))
    if dmax != dmin:
        vals = (vals - dmin) * (pmax - pmin) / (dmax - dmin) + pmin
    if np.any(vals < -1e-4) or not np.allclose(vals, np.round(vals), atol=1e-3):
        raise ValueError('spike 字段不是非负整数计数；需要设备导出说明后才能解释。')
    vals = np.round(vals).astype(int)
    return vals, {'duration_seconds': count, 'sampling_rate': 1, 'unit': unit,
        'patient': raw[8:88].decode('latin1').strip(),
        'start_date': raw[168:176].decode('latin1').strip(), 'start_time': raw[176:184].decode('latin1').strip()}


def electrode_site(label):
    clean = re.sub(r'^EEG\s*', '', label.upper()).strip()
    m = re.match(r'^(FP[12]|AF[Z12345678]|F[Z12345678]|C[Z123456]|P[Z12345678]|O[Z12]|T[12345678])(?:$|[^0-9])', clean)
    if not m: return ('未映射头皮区', '未知')
    key = m.group(1)
    region = ('额区' if key.startswith(('FP', 'AF', 'F')) else '中央区' if key.startswith('C') else '顶区' if key.startswith('P') else '枕区' if key.startswith('O') else '颞区')
    side = '中线' if key.endswith('Z') else '左侧' if int(re.search(r'\d+', key).group()) % 2 else '右侧'
    return (region, side)


def detect_candidates(data, fs, labels, counts, scope='marked', z_threshold=6):
    """候选半高宽和后续慢波均为启发式测量，不等同于临床 IED 判读。"""
    high = min(70, fs * .4)
    wide = signal.sosfiltfilt(signal.butter(4, [.5, high], btype='bandpass', fs=fs, output='sos'), data, axis=1)
    sharp = signal.sosfiltfilt(signal.butter(4, [5, high], btype='bandpass', fs=fs, output='sos'), data, axis=1)
    rows = []
    for c, label in enumerate(labels):
        yy = sharp[c]; sigma = max(float(np.median(np.abs(yy - np.median(yy))) * 1.4826), 1e-12)
        if np.std(yy) < 1e-10: continue
        peaks, props = signal.find_peaks(np.abs(yy), height=z_threshold*sigma, prominence=3*sigma,
            distance=max(1, int(.08*fs)), width=(max(1, .008*fs), .1*fs))
        smooth = np.convolve(wide[c], np.ones(max(1, int(.02*fs))) / max(1, int(.02*fs)), mode='same')
        for j, peak in enumerate(peaks):
            second = int(peak // fs)
            if scope == 'marked' and (second >= len(counts) or counts[second] <= 0): continue
            if peak < .8*fs or peak + .6*fs >= data.shape[1]:
                slow_status, slow_amp, slow_ms = '无法评估（记录边界）', np.nan, np.nan
            else:
                baseline = smooth[int(peak-.75*fs):int(peak-.15*fs)]
                center = np.median(baseline)
                noise = max(float(np.median(np.abs(baseline-center))*1.4826), 1e-12)
                polarity = np.sign(wide[c, peak] - center) or 1
                follow_start = max(0, peak-int(.1*fs))
                follow_end = min(len(smooth), peak+int(.85*fs))
                follow = -polarity * (smooth[follow_start:follow_end] - center)
                sp, pr = signal.find_peaks(follow, height=max(1.5*noise, .25*abs(wide[c,peak]-center)),
                    prominence=noise, width=(.08*fs, .5*fs))
                eligible = np.where((sp+follow_start >= peak+.08*fs)&(sp+follow_start <= peak+.5*fs))[0]
                if eligible.size:
                    k = eligible[np.argmax(pr['peak_heights'][eligible])]
                    slow_status, slow_amp, slow_ms = '有后续慢波候选', float(pr['peak_heights'][k]), float(pr['widths'][k]/fs*1000)
                else: slow_status, slow_amp, slow_ms = '未满足慢波启发式阈值', np.nan, np.nan
            region, side = electrode_site(label)
            rows.append({'时间_s': peak/fs, '通道': label, '对应头皮区': region, '侧别': side,
                '峰幅_µV': float(wide[c,peak]), '背景倍数_z': float(props['peak_heights'][j]/sigma),
                '峰半高宽_ms': float(props['widths'][j]/fs*1000), '后续慢波': slow_status,
                '慢波幅_µV': slow_amp, '慢波半高宽_ms': slow_ms,
                '设备该秒计数': int(counts[second]) if second < len(counts) else None})
            if len(rows) > 20000: raise ValueError('候选超过 20000 个，可能存在大量伪迹；请提高背景倍数阈值或限制在设备标记秒内。')
    rows.sort(key=lambda x:x['时间_s'])
    groups = []
    for row in rows:
        if not groups or row['时间_s'] - groups[-1][0]['时间_s'] > .05: groups.append([])
        groups[-1].append(row)
        row['事件组'] = len(groups)
    events = []
    for i,g in enumerate(groups,1):
        representative = max(g,key=lambda r:r['背景倍数_z'])
        events.append({'事件组':i, '时间_s':representative['时间_s'], '主候选通道':representative['通道'],
            '对应头皮区':representative['对应头皮区'], '侧别':representative['侧别'],
            '涉及通道':', '.join(sorted({r['通道'] for r in g})), '通道数':len({r['通道'] for r in g}),
            '后续慢波': '有后续慢波候选' if any(r['后续慢波']=='有后续慢波候选' for r in g) else representative['后续慢波'],
            '复核提示':'多通道同步，需排除共模/体动' if len({r['通道'] for r in g})>=max(3,len(labels)*.6) else '需人工复核形态与生理电场'})
    return rows, events, wide


def analyze_spike(marker_path, eeg_path, output, scope='marked', z_threshold=6, amplitude_scale=1):
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    counts, header = read_spike_counts(marker_path)
    recording_header = dict(header)
    files = []; warnings = ['spike 是 1 Hz 设备计数字段，不包含精确波形、通道或空间位置。计数含义须结合设备说明确认。']
    timeline = pd.DataFrame({'记录秒':np.arange(len(counts)), '设备计数':counts})
    count_path = output/'设备spike逐秒计数.csv'; timeline.to_csv(count_path,index=False,encoding='utf-8-sig'); files.append(count_path)
    plt=plotting(); fig,ax=plt.subplots(figsize=(12,3.5),constrained_layout=True)
    ax.step(np.arange(len(counts)),counts,where='post');ax.set(xlabel='原始记录时间 (s)',ylabel='设备计数/秒',title='spike.bdf 设备计数字段（非已确诊棘波数）');ax.grid(alpha=.2)
    path=output/'设备spike时间分布.png';fig.savefig(path,dpi=150);plt.close(fig);files.append(path)
    metrics={'duration_seconds':len(counts),'device_count':int(counts.sum()),'marked_seconds':int(np.count_nonzero(counts)),
        'device_count_per_min':float(counts.sum()/len(counts)*60),'candidate_events':None,'candidate_rate_per_min':None,'slow_wave_candidates':None}
    rows=[];events=[];regions=[]
    if eeg_path:
        data,labels,units,eeg_header,_=read_eeg_file(eeg_path,auto_eeg_only=True)
        recording_header = eeg_header.to_dict()
        for key in ('start_date','start_time'):
            if header[key] and getattr(eeg_header,key) and header[key]!=getattr(eeg_header,key):
                raise ValueError('spike 与 data 文件头开始时间不同，请从同一记录段文件夹选择配套文件。')
        if abs(eeg_header.duration_seconds-len(counts))>max(1,.01*len(counts)):
            raise ValueError('spike 与 data 记录时长不匹配，请选择同一记录段中的配套文件。')
        if eeg_header.sampling_rate < 200:
            raise ValueError('候选棘波形态分析需要原始 EEG 采样率至少 200 Hz。')
        keep=[i for i,l in enumerate(labels) if l.upper().strip() not in ('A1','A2','M1','M2')]
        data=data[keep];labels=[labels[i] for i in keep];units=[units[i] for i in keep]
        for i,u in enumerate(units):
            factors={'v':1e6,'mv':1e3,'nv':.001,'uv':1,'µv':1,'μv':1}
            if u.lower().strip() not in factors: raise ValueError(f'通道 {labels[i]} 的电压单位无法解释：{u}')
            data[i]*=factors[u.lower().strip()]*amplitude_scale
        data,fs,_=_downsample(data,eeg_header.sampling_rate,min(500,eeg_header.sampling_rate))
        rows,events,wide=detect_candidates(data,fs,labels,counts,scope,z_threshold)
        columns=['事件组','时间_s','通道','对应头皮区','侧别','峰幅_µV','背景倍数_z','峰半高宽_ms','后续慢波','慢波幅_µV','慢波半高宽_ms','设备该秒计数']
        detections=pd.DataFrame(rows,columns=columns)
        ev=pd.DataFrame(events,columns=['事件组','时间_s','主候选通道','对应头皮区','侧别','涉及通道','通道数','后续慢波','复核提示'])
        for df,name in [(detections,'候选_通道明细.csv'),(ev,'候选_去重事件.csv')]:
            pp=output/name;df.to_csv(pp,index=False,encoding='utf-8-sig');files.append(pp)
        if rows:
            regions=detections.groupby(['对应头皮区','侧别']).size().reset_index(name='通道候选次数').to_dict('records')
            fig,axes=plt.subplots(1,2,figsize=(12,4.5),constrained_layout=True)
            by=detections.groupby('通道').size().sort_values();axes[0].barh(by.index,by.values);axes[0].set(title='头皮通道候选次数（可跨通道重复）',xlabel='次数')
            by2=detections.groupby(['对应头皮区','侧别']).size();axes[1].barh([' · '.join(k) for k in by2.index],by2.values);axes[1].set(title='对应头皮区分布（非脑源定位）',xlabel='通道候选次数')
            pp=output/'候选_通道及头皮区分布.png';fig.savefig(pp,dpi=150,bbox_inches='tight');plt.close(fig);files.append(pp)
            selected=events[:min(6,len(events))]
            fig,axes=plt.subplots(len(selected),1,figsize=(12,2.5*len(selected)),squeeze=False,constrained_layout=True)
            for ax,event in zip(axes[:,0],selected):
                c=labels.index(event['主候选通道']);center=int(event['时间_s']*fs);left=max(0,center-int(fs));right=min(data.shape[1],center+int(fs))
                tt=np.arange(left,right)/fs-event['时间_s'];ax.plot(tt,data[c,left:right],lw=.7,alpha=.45,label='原始信号（已换算单位）');ax.plot(tt,wide[c,left:right],lw=.9,label='0.5–70 Hz 复核信号');ax.axvline(0,color='red',ls='--');ax.axvspan(.08,.5,color='orange',alpha=.1)
                ax.set(title=f"事件 {event['事件组']} · {event['主候选通道']} · {event['时间_s']:.3f} s · {event['后续慢波']}",xlabel='相对候选峰时间 (s)',ylabel='µV');ax.legend(fontsize=8)
            pp=output/'候选_波形与后续慢波复核.png';fig.savefig(pp,dpi=150,bbox_inches='tight');plt.close(fig);files.append(pp)
        metrics.update(candidate_events=len(events),candidate_rate_per_min=len(events)/len(counts)*60,
            slow_wave_candidates=sum(e['后续慢波']=='有后续慢波候选' for e in events),analysis_channels=len(labels),sampling_rate=fs)
        warnings += ['独立候选分析使用 0.5–70 Hz 双向滤波与 5–70 Hz 形态检测，不复用常规 EEG 的片段剔除，以保留短暂事件。',
            '候选峰：背景稳健尺度阈值、8–100 ms 半高宽；后续慢波：峰后 80–500 ms 的反向宽波及相对幅值规则。该半高宽不等同于临床基线持续时间。',
            '50 ms 内跨通道峰合并为事件组；空间分布按电极对应头皮区统计，无法判定脑内病灶或深部起源。',
            '有/无慢波仅表示是否满足此启发式阈值，不构成临床结论；需人工排除肌电、眼动、电极伪迹及滤波振铃。',
            f'搜索范围：{"仅设备计数非零的原始秒区间" if scope=="marked" else "全记录"}；背景倍数阈值 {z_threshold:g}；电压额外倍率 {amplitude_scale:g}。']
    else:
        warnings += ['未上传配套原始 data.bdf，出现通道、对应头皮区和后续慢波均为“无法评估”。零计数也不证明不存在棘波。']
    report=output/'spike分析说明.txt'
    report.write_text(json.dumps(metrics,ensure_ascii=False,indent=2)+'\n\n'+'\n'.join(warnings)+'\n\n方法参考：ACNS 2021 EEG terminology (PMC8135051)；IFCN clinical validation, DOI 10.1212/WNL.0000000000009439。\n本启发式尚未在有专家标签的本队列中验证。',encoding='utf-8');files.append(report)
    return {'kind':'spike','metrics':metrics,'warnings':warnings,'rows':rows[:500],'events':events[:200],
        'recording_header':recording_header,
        'regions':regions,'table_truncated':len(rows)>500 or len(events)>200,'files':[str(p) for p in files]}
