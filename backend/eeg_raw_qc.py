"""原始BDF质控与FIF导出：不滤波、不删样本，MNE负责Raw、坐标、注释和FIF读回核对。

Independent, unfiltered BDF QC and audited double-precision FIF export.

This module deliberately never calls the filtered feature analysis pipeline.
Numerical PSD estimators and display envelopes do not modify exported samples.
"""

from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from eeg_mne_ops import welch_psd as welch
from scipy.signal import find_peaks

from eeg_raw_bdf import NativeBDF
from eeg_montage import montage_mapping, template, export_montage

REFERENCE64 = (
    "Fp1 AF7 AF3 F1 F3 F5 F7 FT7 FC5 FC3 FC1 C1 C3 C5 T7 TP7 CP5 CP3 CP1 P1 P3 P5 P7 P9 PO7 PO3 O1 Iz Oz POz Pz CPz "
    "Fpz Fp2 AF8 AF4 AFz Fz F2 F4 F6 F8 FT8 FC6 FC4 FC2 FCz Cz C2 C4 C6 T8 TP8 CP6 CP4 CP2 P2 P4 P6 P8 P10 PO8 PO4 O2"
).split()
METHOD = """原始BDF质控与FIF导出 v1
不滤波、不陷波、不重采样、不重参考、不插值、不自动剔除通道或样本。
BDF有符号24位数据按文件头线性标定解码；已声明电压单位换算成SI伏特；不施加额外倍率。
Status保留完整标定序列，不按触发位做掩码；未知单位保留头单位数值并标为misc，不能当伏特解释。
无电压单位且物理上下限相等的通道保留原始有符号数字计数，不猜测物理倍率；通道表明确标注。
混合采样率按原采样率分组分别导出FIF，各组通道顺序沿用源文件；原顺序见通道表。
FIF用fmt=double保存，坏道仅info['bads']，坏段仅BAD_注释，均不删除。
中文等非Latin1标签在FIF description中按MNE兼容方式转义；完整原文与精确秒数同时写入Annotations.extras。
BAD_ACQ_SKIP会触发MNE的跳过样本保存，故此保留其原文于extras、description改为BAD_source_acquisition_skip。
标准10–20（含扩展点位）模板坐标为head米制，非个体实测；未确认/重复位置留空。
FIF不等同原BDF逐字节副本。原采集预滤波不可逆，见各通道prefilter记录。本入口不新增滤波。
文件头日期/时刻按原文保留在汇总，不猜测时区，因此FIF meas_date为空，相对时间从首样本起。
Welch：Hann窗，50%重叠，detrend=False，无去均值，nfft=nperseg，不补零，0至Nyquist。
全记录PSD包含所有原始样本范围（Welch窗末端不足一窗部分不参与该估计），包含人工坏道/坏段。
分段汇总含最后不足整段的尾部；PSD窗不足时缩短窗长并报告实际分辨率。
工频候选：50/60Hz±max(0.5Hz,Δf)峰值与两侧2–5Hz中位PSD的dB比；>10dB仅提示复核。
频谱形状离群：同采样率同单位EEG中位中心化log10 PSD，1Hz至Nyquist逐频稳健z，取95百分位；
比较不足4个通道时不评分，>6仅提示复核，非自动坏道判据。坏道不会被自动设定。
α峰：每个手动睁/闭眼区间独立计算，只用实际存在的O1/Oz/O2，不跨区间拼窗；
排除人工/源注释BAD时段后按连续余段Welch，按Welch窗数加权；人工坏道不检峰。
在全谱log PSD上找prominence≥3dB的局部峰，并取8–13Hz内功率最大的一个；未满足则记缺失。
α功率=sum(8–13Hz的PSD)*Δf；峰高为dB相对1 µV²/Hz；prominence为局部峰突出度而非诊断阈值。
重叠睁闭眼区间拒绝；过短的清洁余段不补齐；α未检出不等于无α或临床异常。
枕区图为O1/Oz/O2中实际可用、未标坏道通道的平均功率谱：先逐通道算PSD，再平均线性功率，不平均波形或dB。
状态区间在每通道内按Welch窗数加权，再按通道等权平均；睁闭眼使用相同共同通道；不同原采样率分组绘制。
首窗与交互波形从额区到枕区排列，辅助/参考信号分组；默认按当前片段完整范围自适应坐标，另附100µV统一标尺图。
自适应只改变坐标与显示中心，每行标注实际单位、范围、中心值；不修改样本、PSD、坐标装配或FIF。平线保持平线。
波形浏览的min/max包络仅用于显示；FIF导出仍包含全部原始采样点。
"""


def validate_intervals(items, duration, eyes=False):
    if not isinstance(items, list) or len(items) > 1000:
        raise ValueError("区间列表无效或超过1000条。")
    rows = []
    for item in items:
        a = float(item["start"])
        b = float(item["end"])
        label = str(item.get("label", "artifact"))[:80].strip()
        if (
            not all(math.isfinite(x) for x in (a, b))
            or not 0 <= a < b <= duration + 1e-9
            or not label
        ):
            raise ValueError("区间须满足0≤起点<终点≤记录时长，并填写标签。")
        if eyes and label not in ("open", "closed"):
            raise ValueError("眼状态只能填open或closed。")
        rows.append({"start": a, "end": min(b, duration), "label": label})
    rows.sort(key=lambda r: r["start"])
    if eyes and any(b["start"] < a["end"] for a, b in zip(rows, rows[1:])):
        raise ValueError("睁/闭眼区间不能相互重叠，不能重复计算相同数据。")
    return rows


def assemble_mapping(source, recordinfo=None, overrides=None):
    """把通道名、可选RecordInfo与人工覆盖配置合并成装配表，冲突或未知项明确指出。"""
    mapping = montage_mapping([r["name"] for r in source.signals], recordinfo=recordinfo)
    lookup = {r["name"]: r for r in source.signals}
    overrides = overrides or []
    if not isinstance(overrides, list) or len(overrides) > len(lookup):
        raise ValueError("手动电极映射条数无效。")
    seen = set()
    by_name = {r["原始通道"]: r for r in mapping["rows"]}
    for item in overrides:
        name = item["name"]
        kind = item["type"]
        standard = item.get("standard", "")
        if (
            name not in lookup
            or name in seen
            or kind not in ("eeg", "eog", "ecg", "emg", "misc", "stim", "resp")
        ):
            raise ValueError("手动映射包含未知/重复通道或不支持的类型。")
        seen.add(name)
        if standard and (standard not in template()["positions"] or kind != "eeg"):
            raise ValueError("手动模板名称不在标准模板内，或不是EEG类型。")
        row = by_name[name]
        xyz = template()["positions"].get(standard)
        row.update(
            {
                "MNE类型": kind,
                "标准通道": standard,
                "类型依据": "用户手动确认",
                "可装配": bool(xyz),
                "映射状态": "手动映射" if xyz else "手动留空",
                "匹配说明": "人工指定；仅更改元数据",
                "x_m": xyz[0] if xyz else None,
                "y_m": xyz[1] if xyz else None,
                "z_m": xyz[2] if xyz else None,
            }
        )
    coordinates = defaultdict(list)
    for row in mapping["rows"]:
        native = lookup[row["原始通道"]]
        if native["scale"] is None and row["MNE类型"] in ("eeg", "eog", "ecg", "emg"):
            row.update(
                {
                    "MNE类型": "misc",
                    "可装配": False,
                    "映射状态": "单位不支持电压解释；留空",
                    "标准通道": "",
                    "x_m": None,
                    "y_m": None,
                    "z_m": None,
                }
            )
        if row["可装配"]:
            coordinates[tuple(round(row[k], 12) for k in ("x_m", "y_m", "z_m"))].append(row)
    for rows in coordinates.values():
        if len(rows) > 1:
            raise ValueError(
                "多个通道指向同一模板位置，请用手动映射消除重复："
                + ", ".join(r["原始通道"] for r in rows)
            )
    from eeg_montage import mapping_summary

    return mapping_summary(mapping["rows"], "原始BDF全部信号；坏道只标记，不删除")


def spectrum(data, fs, window):
    nperseg = min(len(data), max(2, int(round(window * fs))))
    if nperseg < 2:
        return np.array([]), np.array([]), 0, 0
    overlap = nperseg // 2
    f, psd = welch(
        data,
        fs=fs,
        window="hann",
        nperseg=nperseg,
        noverlap=overlap,
        nfft=nperseg,
        detrend=False,
        scaling="density",
    )
    windows = 1 + (len(data) - nperseg) // (nperseg - overlap)
    return f, psd, nperseg, windows


def line_ratio(f, p, hz):
    if len(f) < 2 or f[-1] < hz + 5:
        return None
    df = f[1] - f[0]
    distance = np.abs(f - hz)
    center = p[distance <= max(0.5, df)]
    baseline = p[(distance >= 2) & (distance <= 5)]
    if not len(center) or not len(baseline) or np.median(baseline) <= 0:
        return None
    return float(
        10
        * (
            np.log10(max(float(np.max(center)), np.finfo(float).tiny))
            - np.log10(np.median(baseline))
        )
    )


def alpha_peak(f, p):
    result = {
        "alpha_peak_hz": None,
        "peak_db_uv2_hz": None,
        "prominence_db": None,
        "alpha_power_uv2": None,
        "detected": 0,
    }
    if len(f) < 3 or f[-1] < 13:
        return result
    log = 10 * np.log10(np.maximum(p, np.finfo(float).tiny))
    peaks, props = find_peaks(log, prominence=3)
    valid = [
        (int(i), float(prom)) for i, prom in zip(peaks, props["prominences"]) if 8 <= f[i] <= 13
    ]
    result["alpha_power_uv2"] = float(np.sum(p[(f >= 8) & (f <= 13)]) * (f[1] - f[0]))
    if valid:
        i, prominence = max(valid, key=lambda pair: p[pair[0]])
        result.update(
            alpha_peak_hz=float(f[i]),
            peak_db_uv2_hz=float(log[i]),
            prominence_db=prominence,
            detected=1,
        )
    return result


def clean_spans(start, end, bads):
    spans = [(start, end)]
    for bad in bads:
        new = []
        for a, b in spans:
            if bad["end"] <= a or bad["start"] >= b:
                new.append((a, b))
                continue
            if a < bad["start"]:
                new.append((a, bad["start"]))
            if bad["end"] < b:
                new.append((bad["end"], b))
        spans = new
    return spans


def condition_spectrum(data, fs, window, interval, bads):
    """No bridging bad intervals; all retained windows have the same FFT length."""
    n = max(2, round(window * fs))
    accumulator = None
    weight = 0
    f = None
    for a, b in clean_spans(interval["start"], interval["end"], bads):
        values = data[math.ceil(a * fs) : math.ceil(b * fs)]
        if len(values) < n:
            continue
        freq, p, _, count = spectrum(values, fs, window)
        accumulator = p * count if accumulator is None else accumulator + p * count
        weight += count
        f = freq
    return f, accumulator / weight if weight else None, weight


def roundtrip(raw, path, mapping):
    """用MNE保存FIF后重新读取，逐块比较样本、通道、标记和坐标，验证导出没有悄悄改变数据。"""
    import mne

    raw.save(
        path,
        fmt="double",
        picks=None,
        proj=False,
        drop_small_buffer=False,
        overwrite=False,
        verbose="ERROR",
    )
    restored = mne.io.read_raw_fif(path, preload=False, verbose="ERROR")
    if (
        restored.ch_names != raw.ch_names
        or restored.n_times != raw.n_times
        or restored.info["sfreq"] != raw.info["sfreq"]
    ):
        raise RuntimeError("FIF通道、采样点或采样率回读不一致。")
    error = 0.0
    for start in range(0, raw.n_times, 100000):
        a = raw.get_data(start=start, stop=start + 100000)
        b = restored.get_data(start=start, stop=start + 100000)
        error = max(error, float(np.max(np.abs(a - b))))
        if not np.allclose(a, b, rtol=2e-15, atol=0):
            raise RuntimeError("FIF双精度回读与原始标定样本不一致。")
    if (
        restored.info["bads"] != raw.info["bads"]
        or restored.get_channel_types() != raw.get_channel_types()
    ):
        raise RuntimeError("FIF坏道标记或通道类型回读不一致。")
    if list(restored.annotations.description) != list(raw.annotations.description) or len(
        restored.annotations
    ) != len(raw.annotations):
        raise RuntimeError("FIF注释标签回读不一致。")
    if [dict(e) for e in restored.annotations.extras] != [dict(e) for e in raw.annotations.extras]:
        raise RuntimeError("FIF原始注释标签或精确时间extras回读不一致。")
    if not np.allclose(
        restored.annotations.onset, raw.annotations.onset, atol=1e-5
    ) or not np.allclose(restored.annotations.duration, raw.annotations.duration, atol=1e-5):
        raise RuntimeError("FIF注释时间回读不一致。")
    for original, reloaded in zip(raw.info["chs"], restored.info["chs"]):
        if original["unit"] != reloaded["unit"] or original["unit_mul"] != reloaded["unit_mul"]:
            raise RuntimeError("FIF通道单位回读不一致。")
        if not np.allclose(original["loc"][:3], reloaded["loc"][:3], atol=1e-8, equal_nan=True):
            raise RuntimeError("FIF模板坐标回读不一致。")
    paths = [str(p) for p in restored.filenames]
    restored.close()
    return paths, error


def reference_layout(output):
    from eeg_pair_spike import plotting

    assert len(REFERENCE64) == len(set(REFERENCE64)) == 64
    model = template()
    positions = model["positions"]
    plt = plotting()
    fig, ax = plt.subplots(figsize=(10, 10), constrained_layout=True)
    for i, name in enumerate(REFERENCE64, 1):
        x, y, _ = positions[name]
        ax.scatter(x, y, s=100, color="#3278a0")
        ax.annotate(f"{i}:{name}", (x, y), xytext=(3, 4), textcoords="offset points", fontsize=8)
    ax.set(
        xlabel="右向 x (m)",
        ylabel="前向 y (m)",
        aspect="equal",
        title="64导参考点位示意（扩展10–20模板，非本次实际采集/非设备针脚顺序）",
    )
    ax.grid(alpha=0.2)
    path = output / "64导参考示意_非实际采集.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return str(path)


def analyze_raw(source_path, output, options, recordinfo=None):
    """原始数据流程总入口：保留全部信号，生成质控图表和带标记的FIF；这里不调用滤波或删段。"""
    import mne
    from mne.io.constants import FIFF
    from eeg_pair_spike import plotting

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    source = NativeBDF(source_path)
    events = source.annotations()
    bads = options.get("bads", [])
    if (
        not isinstance(bads, list)
        or len(set(bads)) != len(bads)
        or any(n not in [r["name"] for r in source.signals] for n in bads)
    ):
        raise ValueError("坏道必须是源文件中存在的原始通道名，且不能重复。")
    intervals = validate_intervals(options.get("bad_segments", []), source.duration)
    eyes = validate_intervals(options.get("eyes", []), source.duration, eyes=True)
    window = float(options.get("welch_seconds", 4))
    segment = float(options.get("segment_seconds", 30))
    if (
        not math.isfinite(window)
        or not 1 <= window <= 30
        or not math.isfinite(segment)
        or not 1 <= segment <= 3600
    ):
        raise ValueError("Welch窗需1–30秒，汇总段长需1–3600秒。")
    if math.ceil(source.duration / segment) * len(source.signals) > 200000:
        raise ValueError("汇总条目超过20万，请增大汇总段长。")
    for event in events:
        if event["onset"] < 0 or event["onset"] + event["duration"] > source.duration + 1e-6:
            raise ValueError("源BDF存在范围外注释，不能自动裁剪后宣称注释完整；请核对事件时间。")
    mapping = assemble_mapping(source, recordinfo, options.get("overrides"))
    for row in mapping["rows"]:
        row["人工坏道"] = row["原始通道"] in bads
        row["分析状态"] = "人工坏道，保留全部样本" if row["人工坏道"] else "本次保留"
    by_name = {r["原始通道"]: r for r in mapping["rows"]}
    files = export_montage(mapping, output, "实际通道_Montage")
    files.append(reference_layout(output))
    groups = defaultdict(list)
    for row in source.signals:
        groups[row["spr"]].append(row)
    source_bad = [
        {"start": e["onset"], "end": e["onset"] + e["duration"], "label": e["description"]}
        for e in events
        if e["description"].lower().startswith("bad")
    ]
    alpha_bads = intervals + source_bad
    annotations = [dict(e, source="BDF") for e in events]
    annotations += [
        {
            "onset": r["start"],
            "duration": r["end"] - r["start"],
            "description": "BAD_" + r["label"],
            "source": "manual",
        }
        for r in intervals
    ]
    annotations += [
        {
            "onset": r["start"],
            "duration": r["end"] - r["start"],
            "description": "EYES_" + r["label"],
            "source": "manual",
        }
        for r in eyes
    ]
    inventory = []
    quality = []
    segments = []
    alpha = []
    checks = []
    conditions = defaultdict(list)
    occ_full = {}
    plt = plotting()
    for group_number, rows in enumerate(groups.values(), 1):
        names = [r["name"] for r in rows]
        fs = rows[0]["fs"]
        data = np.stack([source.read(r) for r in rows])
        info = mne.create_info(names, fs, ch_types=[by_name[n]["MNE类型"] for n in names])
        for channel, row in zip(info["chs"], rows):
            channel["unit"] = FIFF.FIFF_UNIT_V if row["scale"] is not None else FIFF.FIFF_UNIT_NONE
            channel["unit_mul"] = FIFF.FIFF_UNITM_NONE
        info["description"] = (
            "Unfiltered calibrated BDF; no rereference/resample/notch; template head coordinates, not individual measurements; "
            "original local clock (timezone unverified): "
            + source.local_date
            + " "
            + source.local_time
        )
        info["bads"] = [n for n in names if n in bads]
        raw = mne.io.RawArray(data, info, copy="auto", verbose="ERROR")
        positions = {
            n: [by_name[n][k] for k in ("x_m", "y_m", "z_m")] for n in names if by_name[n]["可装配"]
        }
        if positions:
            fid = template()["fiducials"]
            montage = mne.channels.make_dig_montage(
                ch_pos=positions,
                nasion=fid["nasion"],
                lpa=fid["lpa"],
                rpa=fid["rpa"],
                coord_frame="head",
            )
            raw.set_montage(montage, on_missing="ignore", verbose="ERROR")
        descriptions = [
            (
                "BAD_source_acquisition_skip"
                if e["description"].lower().startswith("bad_acq_skip")
                else e["description"].encode("latin1", "xmlcharrefreplace").decode("latin1")
            )
            for e in annotations
        ]
        raw.set_annotations(
            mne.Annotations(
                [e["onset"] for e in annotations],
                [e["duration"] for e in annotations],
                descriptions,
                extras=[
                    {
                        "original_description": e["description"],
                        "exact_onset_s": e["onset"],
                        "exact_duration_s": e["duration"],
                        "source": e["source"],
                    }
                    for e in annotations
                ],
            )
        )
        fif_paths, error = roundtrip(raw, output / f"group{group_number}_{fs:g}Hz_raw.fif", mapping)
        files.extend(fif_paths)
        checks.append(
            {
                "组": group_number,
                "采样率_Hz": fs,
                "通道数": len(names),
                "每通道样本数": raw.n_times,
                "最大回读绝对误差": error,
                "坐标坏道注释回读": "通过",
            }
        )
        spectra = []
        freq = None
        for index, row in enumerate(rows):
            values = data[index]
            display = values * 1e6 if row["scale"] is not None else values
            f, psd, nfft, count = spectrum(display, fs, window)
            freq = f
            spectra.append(psd)
            if by_name[row["name"]]["标准通道"] in ("O1", "Oz", "O2") and row["scale"] is not None:
                occ_full[row["name"]] = (f, psd, fs)
            digital = source.read(row, digital=True)
            clip = float(np.mean((digital <= row["dmin"]) | (digital >= row["dmax"])))
            unit = (
                "µV"
                if row["scale"] is not None
                else "digital counts" if row["digital_only"] else row["unit"] or "unspecified"
            )
            metrics = {
                "通道": row["name"],
                "组": group_number,
                "类型": by_name[row["name"]]["MNE类型"],
                "单位": unit,
                "采样率_Hz": fs,
                "样本数": len(values),
                "标准差": float(np.std(display)),
                "峰峰值": float(np.ptp(display)),
                "RMS": float(np.sqrt(np.mean(display**2))),
                "数字边界占比": clip,
                "50Hz突出_dB": line_ratio(f, psd, 50),
                "60Hz突出_dB": line_ratio(f, psd, 60),
                "频谱离群评分": None,
                "人工坏道": row["name"] in bads,
                "Welch窗样本": nfft,
                "Welch窗数": count,
                "频率分辨率_Hz": fs / nfft if nfft else None,
                "候选提示": "",
            }
            flags = []
            if np.ptp(values) == 0:
                flags.append("完全平坦")
            if clip > 0.01:
                flags.append("数字边界≥1%")
            if any(
                metrics[k] is not None and metrics[k] > 10 for k in ("50Hz突出_dB", "60Hz突出_dB")
            ):
                flags.append("工频尖峰候选")
            metrics["候选提示"] = "；".join(flags)
            quality.append(metrics)
            inventory.append(
                {
                    "原顺序": source.signals.index(row) + 1,
                    "通道": row["name"],
                    "组": group_number,
                    "采样率_Hz": fs,
                    "源单位": row["unit"],
                    "FIF数值单位": (
                        "V"
                        if row["scale"] is not None
                        else (
                            "digital counts"
                            if row["digital_only"]
                            else row["unit"] or "unspecified"
                        )
                    ),
                    "标定退化保留数字计数": row["digital_only"],
                    "源数字最小": row["dmin"],
                    "源数字最大": row["dmax"],
                    "源物理最小": row["pmin"],
                    "源物理最大": row["pmax"],
                    "采集预滤波原文": row["prefilter"],
                    "额外信号处理": "无",
                    "FIF类型": by_name[row["name"]]["MNE类型"],
                }
            )
            for start in np.arange(0, source.duration, segment):
                end = min(source.duration, start + segment)
                block = display[round(start * fs) : round(end * fs)]
                if len(block) == 0:
                    continue
                sf, sp, sn, sc = spectrum(block, fs, window)
                marked = [b["label"] for b in alpha_bads if b["start"] < end and b["end"] > start]
                segment_alpha = {
                    "枕区α峰检出": None,
                    "枕区α峰_Hz": None,
                    "枕区α峰突出_dB": None,
                    "枕区α功率_µV2": None,
                }
                if (
                    by_name[row["name"]]["标准通道"] in ("O1", "Oz", "O2")
                    and row["scale"] is not None
                    and row["name"] not in bads
                ):
                    af, ap, aw = condition_spectrum(
                        display, fs, window, {"start": float(start), "end": float(end)}, alpha_bads
                    )
                    segment_alpha["枕区α峰检出"] = 0
                    if aw:
                        peak = alpha_peak(af, ap)
                        segment_alpha.update(
                            {
                                "枕区α峰检出": peak["detected"],
                                "枕区α峰_Hz": peak["alpha_peak_hz"],
                                "枕区α峰突出_dB": peak["prominence_db"],
                                "枕区α功率_µV2": peak["alpha_power_uv2"],
                            }
                        )
                segments.append(
                    {
                        "通道": row["name"],
                        "组": group_number,
                        "起点_s": float(start),
                        "终点_s": float(end),
                        "样本数": len(block),
                        "采样率_Hz": fs,
                        "单位": unit,
                        "标准差": float(np.std(block)),
                        "峰峰值": float(np.ptp(block)),
                        "人工坏道": row["name"] in bads,
                        "坏段重叠标签": "；".join(marked),
                        "Welch窗样本": sn,
                        "Welch窗数": sc,
                        "频率分辨率_Hz": fs / sn if sn else None,
                        "50Hz突出_dB": line_ratio(sf, sp, 50),
                        "60Hz突出_dB": line_ratio(sf, sp, 60),
                        "眼状态重叠": "；".join(
                            e["label"] for e in eyes if e["start"] < end and e["end"] > start
                        ),
                        **segment_alpha,
                    }
                )
            if by_name[row["name"]]["标准通道"] in ("O1", "Oz", "O2") and row["scale"] is not None:
                for number, eye in enumerate(eyes, 1):
                    af, ap, aw = condition_spectrum(display, fs, window, eye, alpha_bads)
                    record = {
                        "通道": row["name"],
                        "区间": number,
                        "状态": eye["label"],
                        "起点_s": eye["start"],
                        "终点_s": eye["end"],
                        "可用Welch窗数": aw,
                        "alpha_peak_hz": None,
                        "peak_db_uv2_hz": None,
                        "prominence_db": None,
                        "alpha_power_uv2": None,
                        "detected": 0,
                        "说明": (
                            "人工坏道，不检峰"
                            if row["name"] in bads
                            else "清洁余段不足一窗" if not aw else "≥3dB局部峰规则"
                        ),
                    }
                    if aw and row["name"] not in bads:
                        record.update(alpha_peak(af, ap))
                        conditions[row["name"]].append((eye["label"], af, ap, aw, number))
                    alpha.append(record)
        # Compare only same-rate, same-unit EEG. Other signals remain visible and exported.
        units = defaultdict(list)
        for i, row in enumerate(rows):
            units[
                (
                    "µV"
                    if row["scale"] is not None
                    else "digital counts" if row["digital_only"] else row["unit"] or "unspecified"
                )
            ].append(i)
        for unit_number, (unit, indices) in enumerate(units.items(), 1):
            eeg = [i for i in indices if by_name[names[i]]["MNE类型"] == "eeg"]
            if len(eeg) >= 4 and len(freq) > 3:
                matrix = np.log10(
                    np.maximum(np.array(spectra)[eeg][:, freq >= 1], np.finfo(float).tiny)
                )
                if matrix.shape[1]:
                    matrix -= np.median(matrix, axis=1, keepdims=True)
                    median = np.median(matrix, axis=0)
                    mad = np.median(np.abs(matrix - median), axis=0) * 1.4826
                    scores = np.percentile(
                        np.abs(matrix - median) / np.maximum(mad, 0.1), 95, axis=1
                    )
                    for i, score in zip(eeg, scores):
                        metric = next(r for r in quality if r["通道"] == names[i])
                        metric["频谱离群评分"] = float(score)
                        if score > 6:
                            metric["候选提示"] += "；频谱形状离群候选"
            fig, axes = plt.subplots(2, 1, figsize=(15, 10), constrained_layout=True)
            for i in indices:
                color = "#cf342e" if names[i] in bads else None
                axes[0].plot(
                    freq,
                    10 * np.log10(np.maximum(spectra[i], np.finfo(float).tiny)),
                    lw=0.8,
                    alpha=0.75,
                    color=color,
                    label=names[i] + (" [BAD]" if names[i] in bads else ""),
                )
            axes[0].set(
                title=f"全部通道Welch PSD｜组{group_number} {fs:g}Hz｜单位{unit}²/Hz｜未滤波，含坏道/坏段",
                xlabel="Frequency (Hz)",
                ylabel=f"PSD (dB re 1 {unit}²/Hz)",
                xlim=(0, fs / 2),
            )
            for hz in range(50, int(fs / 2) + 1, 50):
                axes[0].axvline(hz, color="#df9b34", ls=":", lw=0.7)
            for hz in range(60, int(fs / 2) + 1, 60):
                axes[0].axvline(hz, color="#8e70ac", ls=":", lw=0.7)
            axes[0].legend(fontsize=6, ncol=8)
            axes[0].grid(alpha=0.2)
            display_psd = np.log10(np.maximum(np.array(spectra)[indices], np.finfo(float).tiny))
            display_psd -= np.median(display_psd, axis=1, keepdims=True)
            if len(freq):
                image = axes[1].imshow(
                    display_psd,
                    aspect="auto",
                    origin="lower",
                    extent=[0, fs / 2, -0.5, len(indices) - 0.5],
                    cmap="coolwarm",
                    vmin=-3,
                    vmax=3,
                )
                fig.colorbar(
                    image, ax=axes[1], label="log10 PSD minus per-channel median (display only)"
                )
            axes[1].set_yticks(range(len(indices)), [names[i] for i in indices], fontsize=6)
            axes[1].set(
                title="全频谱形状复核：每通道中位中心化，仅用于显示，色标截断±3",
                xlabel="Frequency (Hz)",
            )
            unit_label = "电压" if unit == "µV" else f"辅助单位{unit_number}"
            path = output / f"全通道Welch_PSD_组{group_number}_{unit_label}.png"
            fig.savefig(path, dpi=145)
            plt.close(fig)
            files.append(str(path))
        del data
        raw.close()
    from eeg_occipital_psd import export_roi_spectra

    occipital_mean, roi_files = export_roi_spectra(
        occ_full, conditions, eyes, bads, output, alpha_peak
    )
    files.extend(roi_files)
    from eeg_raw_display import first_window_plots, ordered_channels

    files.extend(first_window_plots(source_path, output, mapping, bads, alpha_bads))
    display_order = ordered_channels([r["name"] for r in source.signals], mapping)
    native_channels = {r["name"]: r for r in source.signals}
    summary = [
        {
            "记录时长_s": source.duration,
            "全部信号通道": len(source.signals),
            "采样率组数": len(groups),
            "人工坏道数": len(bads),
            "源BAD注释数": len(source_bad),
            "人工坏段数": len(intervals),
            "源事件注释数": len(events),
            "眼状态区间数": len(eyes),
            "alpha检出区间通道数": sum(r["detected"] for r in alpha),
            "alpha评估区间通道数": len(alpha),
            "源本地日期": source.local_date,
            "源本地时刻": source.local_time,
            "源TAL首样本偏移_s": getattr(source, "annotation_origin", 0.0),
            "时区": "未确认，FIF meas_date为空",
            "新增滤波": "无",
            "重采样": "无",
            "重参考": "无",
            "删道删段": "无",
            "Welch设定窗_s": window,
            "汇总段长_s": segment,
        }
    ]
    annotation_rows = [
        {
            "起点_s": e["onset"],
            "持续_s": e["duration"],
            "标签": e["description"],
            "来源": e["source"],
        }
        for e in annotations
    ]
    tables = [
        ("数据质控汇总", summary),
        ("逐通道参数及PSD质控", quality),
        ("逐段参数与坏段标记", segments),
        ("源通道标定与顺序", inventory),
        ("全部事件及坏段注释", annotation_rows),
        ("Alpha峰数值", alpha),
        ("FIF回读核验", checks),
        ("人工坏道", [{"通道": n, "标记": "BAD（保留样本）"} for n in bads]),
    ]
    for title, rows in tables:
        path = output / (title + ".csv")
        pd.DataFrame(rows or [{"说明": "无记录"}]).to_csv(path, index=False, encoding="utf-8-sig")
        files.append(str(path))
    digest = hashlib.sha256()
    with Path(source_path).open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    notes = output / "原始质控_方法与使用说明.txt"
    notes.write_text(
        METHOD
        + "\nMNE: "
        + mne.__version__
        + "\n源BDF SHA256: "
        + digest.hexdigest()
        + "\n坐标模板SHA256: "
        + template()["sha256"]
        + '\n后续调用：\nimport mne\nraw = mne.io.read_raw_fif("group1_原采样率Hz_raw.fif", preload=False)\n'
        '# raw.info["bads"] 保留坏道；raw.annotations含BAD_时段\n'
        "# 注释原文与精确秒数：raw.annotations.extras\n"
        "# 切片时显式决定是否使用reject_by_annotation；当前数据未删段。\n",
        encoding="utf8",
    )
    files.append(str(notes))
    warnings = [
        "全部原始信号及样本保留；候选提示不自动写成坏道。确认后填入坏道/坏段，再生成新版本FIF。",
        "FIF是标定后的双精度数据；原BDF应另行保留。下载同一组全部FIF分卷后放在同一文件夹。",
        "标准模板坐标不是个体实测；未知单位/未确认点位留空，参见通道表。64导图是参考示意。",
    ]
    if not eyes:
        warnings.append("未填写已确认的睁闭眼时段，α对比不作状态判断。")
    if len(groups) > 1:
        warnings.append("存在混合采样率，已按原速率分组保存，未上采样或丢弃低速通道。")
    from eeg_viewing import viewing_defaults

    return {
        "kind": "raw_qc",
        "summary": summary,
        "quality": quality,
        "segments": segments[:500],
        "segment_total": len(segments),
        "alpha": alpha,
        "occipital_mean": occipital_mean,
        "checks": checks,
        "annotations": annotation_rows[:1000],
        "montage_info": mapping,
        "channels": [
            {
                **r,
                "fs": native_channels[r["name"]]["fs"],
                "unit": native_channels[r["name"]]["unit"],
            }
            for r in display_order
        ],
        "duration": source.duration,
        "options": {**options, "welch_seconds": window, "segment_seconds": segment},
        "files": files,
        "warnings": warnings,
        "source_sha256": digest.hexdigest(),
        "display_version": "adaptive_axes_v3",
        "view_defaults": viewing_defaults(source_path, mapping),
    }


def main():
    """命令行入口：解析启动参数后调用主流程，不在这里编写分析公式。"""
    import argparse
    import os
    import tempfile

    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("result")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf8"))
    with tempfile.TemporaryDirectory(prefix="eeg_mne_qc_") as config_home:
        os.environ["_MNE_FAKE_HOME_DIR"] = config_home
        try:
            result = analyze_raw(
                config["source"], config["output"], config["options"], config.get("recordinfo")
            )
            Path(args.result).write_text(
                json.dumps(
                    result,
                    ensure_ascii=False,
                    allow_nan=False,
                    default=lambda value: (
                        value.item() if isinstance(value, np.generic) else str(value)
                    ),
                ),
                encoding="utf8",
            )
        except Exception as exc:
            Path(args.result).write_text(
                json.dumps(
                    {"error": f"{type(exc).__name__}: {str(exc)[:350]}"}, ensure_ascii=False
                ),
                encoding="utf8",
            )
            raise


if __name__ == "__main__":
    main()
