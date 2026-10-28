"""Allow-listed session metadata. Identity/camera fields never leave this parser."""
from datetime import datetime, timedelta, timezone
import json
import math
import re

from eeg_events import recording_start
from eeg_feature_engine import _is_eeg_like, _normalize_channel_name

LOCAL_ZONE = timezone(timedelta(hours=8))
MAX_JSON_BYTES = 2*1024**2


def number(value, name, minimum=0):
    if isinstance(value,bool):raise ValueError(f'{name} 必须是数值。')
    try:result=float(value)
    except (TypeError,ValueError) as error:raise ValueError(f'{name} 必须是数值。') from error
    if not math.isfinite(result) or result<minimum:raise ValueError(f'{name} 的范围无效。')
    return result


def integer(value, name, minimum=0):
    result=number(value,name,minimum)
    if not result.is_integer():raise ValueError(f'{name} 必须是整数。')
    return int(result)


def decode_metadata(raw, kind):
    if not 0<len(raw)<=MAX_JSON_BYTES:raise ValueError('JSON 元数据为空或超过2 MB。')
    try:
        encoding='utf-16' if raw[:2] in (b'\xff\xfe',b'\xfe\xff') else 'utf-8-sig'
        data=json.loads(raw.decode(encoding))
    except (ValueError,UnicodeError) as error:raise ValueError('元数据不是有效的 UTF-8/UTF-16 JSON。') from error
    if not isinstance(data,dict):raise ValueError('元数据顶层必须是对象。')
    if kind=='datainfo':return parse_datainfo(data)
    if kind=='recordinfo':return parse_recordinfo(data)
    raise ValueError('不支持的 JSON 元数据类型。')


def parse_datainfo(data):
    timestamp=number(data.get('startTime'),'startTime',0)
    if not 0<=timestamp<=7258118400:raise ValueError('startTime 应为 Unix 秒，不支持猜测毫秒时间戳。')
    try:origin=datetime.fromtimestamp(timestamp,LOCAL_ZONE)
    except (ValueError,OSError,OverflowError) as error:raise ValueError('startTime 无法转换为时间。') from error
    fs=number(data.get('sampleRate'),'sampleRate',1)
    channels=integer(data.get('channelCount'),'channelCount',1)
    count=integer(data.get('count'),'count',1)
    if count>1000 or channels>1024:raise ValueError('元数据条目或通道数超过限额。')
    total=number(data.get('duration'),'duration')
    entries=[]
    for key,value in data.items():
        match=re.fullmatch(r'data([1-9]\d*)',key)
        if not match:continue
        if not isinstance(value,dict):raise ValueError(f'{key} 必须是对象。')
        duration=number(value.get('duration'),f'{key}.duration')
        end=number(value.get('durationFromBegin'),f'{key}.durationFromBegin')
        if end<duration:raise ValueError(f'{key} 结束位置早于其时长。')
        entries.append({'JSON编号':int(match[1]),'起点_s':end-duration,'终点_s':end,'持续_s':duration})
    entries.sort(key=lambda row:row['JSON编号'])
    if len(entries)!=count or [r['JSON编号'] for r in entries]!=list(range(1,count+1)):
        raise ValueError('count 与 data1…dataN 条目不一致或编号不连续。')
    previous=0
    for row in entries:
        if row['起点_s']<previous-1e-6:raise ValueError('datainfo 的条目倒序或重叠。')
        row['与前节点间隔_s']=row['起点_s']-previous
        row['预计开始时间_UTC8']=(origin+timedelta(seconds=row['起点_s'])).isoformat()
        row['条目类型']='有波形时长' if row['持续_s']>0 else '零时长时间节点（不要求data文件）'
        previous=row['终点_s']
    if entries[-1]['终点_s']>total+1e-6:raise ValueError('分段结束位置超过整次记录时长。')
    recorded=sum(r['持续_s'] for r in entries)
    return {'kind':'datainfo','start_timestamp':timestamp,'start_time_utc8':origin.isoformat(),
        'duration_seconds':total,'sampling_rate':fs,'channel_count':channels,
        'entry_count':count,'waveform_count':sum(r['持续_s']>0 for r in entries),
        'recorded_seconds':recorded,'gap_seconds':total-recorded,'segments':entries,
        'warnings':['此格式按 durationFromBegin 为累计结束位置解释，起点=结束位置−时长；零时长条目不当作波形文件。',
                    'JSON 时间以 Unix 秒/UTC+8显示；与BDF存在差异时保留BDF时间，不静默校正或重设采样率。']}


def parse_recordinfo(data):
    work=data.get('WorkChannel')
    if not isinstance(work,dict):raise ValueError('RecordInfo 缺少 WorkChannel 配置。')
    result=[];seen={}
    known={'EEG','ECG','EMG','EOG','STIM','MISC','RESP','ECG','EKG'}
    for group in ('EEGWorkChannel','DifferenceWorkChannel','PeripheralWorkChannel'):
        section=work.get(group,{})
        if not isinstance(section,dict):raise ValueError('RecordInfo 通道分组无效。')
        channels=section.get('Channels',[])
        if not isinstance(channels,list):raise ValueError('RecordInfo Channels 必须是列表。')
        for channel in channels:
            if not isinstance(channel,dict):raise ValueError('RecordInfo 通道条目无效。')
            name=channel.get('Name');kind=channel.get('ChannelType')
            if not isinstance(name,str) or not name.strip() or len(name)>64 or not isinstance(kind,str):
                raise ValueError('通道名称或 ChannelType 无效。')
            name=name.strip();kind=kind.upper().strip();kind='ECG' if kind=='EKG' else kind
            if kind not in known:kind='未知'
            key=_normalize_channel_name(name)
            if key in seen:
                if seen[key]!=kind:raise ValueError('同名通道存在冲突类型，不能自动选择。')
                continue
            seen[key]=kind;result.append({'通道名称':name,'配置类型':kind,'配置分组':group})
            if len(result)>1024:raise ValueError('RecordInfo 通道数超过限额。')
    if not result:raise ValueError('RecordInfo 没有通道配置。')
    counts={kind:sum(r['配置类型']==kind for r in result) for kind in sorted({r['配置类型'] for r in result})}
    return {'kind':'recordinfo','channel_count':len(result),'type_counts':counts,'channels':result,
        'warnings':['仅提取通道名称、类型和分组；身份、摄像头地址、账号/密码等字段不保存也不展示。',
                    '按通道名称匹配，不把 IndexInAllChannels 当作BDF列号；不自动推断参考或电极坐标。',
                    'RecordInfo 只用于类型核对和排除非EEG；原有默认头皮通道规则仍保留，PG/A1/A2等不因此自动纳入。']}


def match_datainfo(metadata, header, used):
    start=recording_start(header).replace(tzinfo=LOCAL_ZONE).timestamp()-metadata['start_timestamp']
    duration=float(header['duration_seconds'])
    candidates=[]
    for row in metadata['segments']:
        if row['持续_s']<=0 or row['JSON编号'] in used:continue
        delta=start-row['起点_s'];duration_delta=duration-row['持续_s']
        if abs(delta)<=1.05 and abs(duration_delta)<=max(.001,1/float(header['sampling_rate'])):
            candidates.append((abs(delta),row,delta,duration_delta))
    candidates.sort(key=lambda item:item[0])
    if not candidates:raise ValueError('所选data无法与datainfo时间和时长匹配；请核对同一次记录，或取消该元数据。')
    if len(candidates)>1 and abs(candidates[0][0]-candidates[1][0])<1e-9:
        raise ValueError('data对应的JSON分段不唯一，不能猜测编号。')
    _,row,delta,duration_delta=candidates[0];used.add(row['JSON编号'])
    return {'JSON编号':row['JSON编号'],'JSON起点_s':row['起点_s'],'BDF相对JSON起点_s':start,
        '起点差_BDF减JSON_s':delta,'时长差_BDF减JSON_s':duration_delta,
        '核对状态':'匹配' if abs(delta)<1e-6 else '时间差提示（保留BDF时间）'}


def select_record_channels(metadata, labels, units, number):
    lookup={_normalize_channel_name(r['通道名称']):r for r in metadata['channels']}
    selected=[];rows=[];seen=set()
    for i,(label,unit) in enumerate(zip(labels,units)):
        key=_normalize_channel_name(label);entry=lookup.get(key);seen.add(key)
        kind=entry['配置类型'] if entry else '未匹配'
        choose=kind=='EEG' and _is_eeg_like(label,unit)
        if choose:selected.append(i)
        decision='纳入质控候选' if choose else 'EEG配置但默认命名规则未纳入' if kind=='EEG' else '非EEG/未知，排除'
        rows.append({'顺序':number,'BDF通道':label,'配置通道':entry['通道名称'] if entry else '',
            '配置类型':kind,'映射状态':'按名称匹配' if entry else '配置未找到','分析选择':decision})
    for key,entry in lookup.items():
        if key not in seen:rows.append({'顺序':number,'BDF通道':'','配置通道':entry['通道名称'],
            '配置类型':entry['配置类型'],'映射状态':'配置通道不在BDF中','分析选择':'未纳入'})
    if len(selected)<3:raise ValueError('按RecordInfo核对后可用头皮EEG候选不足3个；请核对配置文件。')
    return selected,rows
