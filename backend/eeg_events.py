"""BDF annotation-only event support, preserving TAL time and original labels."""
from collections import Counter
from datetime import datetime
import math
from pathlib import Path
import re

import numpy as np
import pandas as pd

NUMBER = rb"[+-]?(?:\d+(?:\.\d*)?|\.\d+)"
TAL = re.compile(rb"(" + NUMBER + rb")(?:\x15(" + NUMBER + rb"))?\x14(.*)\x00", re.S)


def read_event_file(path):
    """Read byte-packed BDF TALs; do not interpret empty timekeeping labels."""
    raw = Path(path).read_bytes()
    if not 256 <= len(raw) <= 20 * 1024**2:
        raise ValueError("evt.bdf 为空、过短或超过 20 MB。")
    try:
        channels = int(raw[252:256])
        header_bytes = int(raw[184:192])
        records = int(raw[236:244])
        if not 1 <= channels <= 128 or header_bytes < 256 + channels*256 or header_bytes > len(raw):
            raise ValueError()
        sh = raw[256:header_bytes]
        labels = [sh[i*16:(i+1)*16].decode('ascii').strip() for i in range(channels)]
        samples = [int(sh[216*channels+i*8:216*channels+(i+1)*8]) for i in range(channels)]
        if any(n <= 0 for n in samples): raise ValueError()
    except (ValueError, UnicodeDecodeError) as error:
        raise ValueError("evt.bdf 的 BDF 头或通道采样数无效。") from error
    if not (raw[:1] == b'\xff' or raw[:8].startswith(b'BDF')):
        raise ValueError("事件入口需要 BDF 文件，不能按 EDF 或其他编码猜测。")
    indices = [i for i,label in enumerate(labels) if label.upper() in ('BDF ANNOTATIONS','EDF ANNOTATIONS')]
    if not indices: raise ValueError("未找到 BDF Annotations 通道；此入口不猜测普通数字通道的编码。")
    record_bytes = sum(samples)*3
    available, remainder = divmod(len(raw)-header_bytes, record_bytes)
    if remainder or (records >= 0 and records != available):
        raise ValueError("evt.bdf 数据记录被截断或与文件头不一致。")
    offsets = np.cumsum([0]+samples)*3
    events = []
    for record in range(available):
        for channel in indices:
            start = header_bytes + record*record_bytes + int(offsets[channel])
            block = raw[start:start+samples[channel]*3]
            for entry in block.split(b'\0'):
                if not entry: continue
                match = TAL.fullmatch(entry+b'\0')
                if not match: raise ValueError("事件 TAL 编码无效；无法可靠解析时间。")
                onset = float(match[1]); duration = float(match[2] or b'0')
                if not math.isfinite(onset) or not math.isfinite(duration) or duration < 0:
                    raise ValueError("事件起点或持续时间无效。")
                for value in match[3].split(b'\x14'):
                    if not value: continue
                    try: label = value.decode('utf8')
                    except UnicodeDecodeError as error:
                        raise ValueError("事件描述不是有效 UTF-8；不进行有损猜测。") from error
                    events.append({'onset':onset,'duration':duration,'label':label})
                    if len(events)>100000: raise ValueError("事件超过 100000 条，无法在本机页面处理。")
    events.sort(key=lambda e:e['onset'])
    return {'events':events,'start_date':raw[168:176].decode('ascii').strip(),
            'start_time':raw[176:184].decode('ascii').strip(),
            'patient':raw[8:88].decode('latin1').replace('\x00',' ').strip()}


def recording_start(header):
    try:
        day,month,year=map(int,header['start_date'].split('.'))
        hour,minute,second=map(int,header['start_time'].split('.'))
        if year < 100: year += 1900 if year>=85 else 2000
        return datetime(year,month,day,hour,minute,second)
    except (KeyError,ValueError,TypeError) as error:
        raise ValueError("文件头缺少可靠的开始日期/时间，不能自动对齐 evt。") from error


def align_events(parsed, reference, provenance=None, analysis_fs=None, offset_seconds=0):
    offset = float(offset_seconds)
    if not math.isfinite(offset) or abs(offset)>86400:
        raise ValueError("事件额外校正偏移须为 ±86400 秒以内的有限数。")
    # A marker-only file may cover an entire session, so its record count is NOT its EEG duration.
    shift = (recording_start(parsed)-recording_start(reference)).total_seconds()+offset
    duration = float(reference['duration_seconds'])
    fs = float(reference['sampling_rate'])
    if not math.isfinite(duration) or duration<=0 or not math.isfinite(fs) or fs<=0:
        raise ValueError("参考记录时长或采样率无效。")
    left=parsed.get('patient','').strip();right=reference.get('patient','').strip()
    unknown = {'','X','X X X X','Unknown','unknown'}
    if left not in unknown and right not in unknown and left != right:
        raise ValueError("evt 与参考 EEG 的文件头受试者标识不同，请核对配套文件。")
    provenance = provenance or {}
    kept = provenance.get('kept_epoch_indices')
    epoch_samples = provenance.get('original_epoch_samples')
    epoch_seconds = epoch_samples/float(analysis_fs) if epoch_samples and analysis_fs else None
    kept_positions = {value:i for i,value in enumerate(kept)} if kept is not None else {}
    rows=[]
    for i,event in enumerate(parsed['events'],1):
        time = event['onset']+shift
        point = 0 <= time < duration
        overlap = point if event['duration']==0 else time < duration and time+event['duration']>0
        clean_time = None
        quality = '未应用片段质控（原始时间轴）'
        if not point: quality = '起点在本段外'
        elif kept is not None and epoch_seconds:
            epoch = int(math.floor(time/epoch_seconds))
            if epoch in kept_positions:
                quality='保留片段内';clean_time=kept_positions[epoch]*epoch_seconds+time-epoch*epoch_seconds
            elif epoch >= int(math.floor(duration/epoch_seconds)): quality='未纳入完整片段的尾部'
            else: quality='已剔除片段内'
        rows.append({'事件序号':i,'原始标签':event['label'],'evt原始起点_s':event['onset'],
            '持续时间_s':event['duration'],'相对本段起点_s':time,
            '本段关系':'起点在本段内' if point else '区间与本段重叠' if overlap else '本段外',
            '原始样本索引_0基':min(int(time*fs),max(0,round(duration*fs)-1)) if point else None,
            '质控状态_起点':quality,'有效拼接时间_s':clean_time})
    return rows, shift


def analyze_events(path, reference, output, provenance=None, analysis_fs=None, offset_seconds=0, prefix='evt'):
    from eeg_pair_spike import plotting
    parsed = read_event_file(path)
    rows, shift = align_events(parsed,reference,provenance,analysis_fs,offset_seconds)
    output=Path(output)
    columns=['事件序号','原始标签','evt原始起点_s','持续时间_s','相对本段起点_s','本段关系','原始样本索引_0基','质控状态_起点','有效拼接时间_s']
    csv=output/f'{prefix}_事件明细.csv'
    pd.DataFrame(rows,columns=columns).to_csv(csv,index=False,encoding='utf-8-sig')
    counts=Counter(e['原始标签'] for e in rows)
    inside=[e for e in rows if e['本段关系']!='本段外']
    warnings=['保留 evt 原始标签，不推断 KeyIn 编码的宫缩、镇痛、棘波或刺激含义，也不自动配对成区间。',
              '相对时间 = evt 起点 + 两文件头开始时间之差 + 手动偏移；文件头通常只有秒级精度，未验证共同硬件时钟。',
              '事件文件的数据记录数不是整次 EEG 时长。只有落在当前 data/spike 段内的事件可与本段对应。',
              '事件用于独立时间轴和表格定位，不改变现有特征、自动质控或进行事件锁定 ERP。样本索引为向下取整的0基索引。']
    if reference['sampling_rate']==1:
        warnings.append('当前参考为1 Hz设备计数；样本索引是计数序列索引，不是原始EEG采样点。')
    if not inside: warnings.append('当前片段内没有事件；片段外事件已保留列出，不会强行平移进当前片段。')
    plt=plotting();fig,axes=plt.subplots(2,1,figsize=(12,6),constrained_layout=True)
    labels=list(counts)[:20];positions={name:i for i,name in enumerate(labels)}
    for ax,subset,title in [(axes[0],rows,'整次 evt：已换算到当前记录的原始时间轴'),(axes[1],inside,'当前记录内事件（原始时间，不是有效拼接时间）')]:
        ax.axvspan(0,reference['duration_seconds'],color='green',alpha=.08,label='当前记录范围')
        for e in subset:
            y=positions.get(e['原始标签'],20)
            ax.plot(e['相对本段起点_s'],y,'|',ms=14,color='#2464ae')
            if e['持续时间_s']>0:
                ax.plot([e['相对本段起点_s'],e['相对本段起点_s']+e['持续时间_s']],[y,y],lw=3,color='#2464ae')
        ax.set_yticks(range(len(labels)),labels);ax.set(title=title,xlabel='相对本段原始时间 (s)');ax.grid(alpha=.2)
        if not subset: ax.text(.5,.5,'无事件',transform=ax.transAxes,ha='center')
    axes[1].set_xlim(0,reference['duration_seconds'])
    image=output/f'{prefix}_事件时间轴.png';fig.savefig(image,dpi=150,bbox_inches='tight');plt.close(fig)
    notes=output/f'{prefix}_事件对齐说明.txt'
    notes.write_text('\n'.join(warnings)+f'\n总偏移：{shift:g} s\n总事件：{len(rows)}；与本段相关：{len(inside)}\n',encoding='utf8')
    return {'total_events':len(rows),'in_segment':len(inside),'outside_segment':len(rows)-len(inside),
            'alignment_shift_seconds':shift,'rows':rows[:1000],'table_truncated':len(rows)>1000,
            'labels':[{'原始标签':k,'次数':v} for k,v in counts.items()],
            'warnings':warnings,'files':[str(csv),str(image),str(notes)]}
