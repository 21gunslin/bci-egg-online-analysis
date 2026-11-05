"""查看用的原始PSD：直接用原始样本估计Welch谱，选择频率范围只改变显示，不是滤波。

Read-only Welch PSD for the original-data inspection UI."""

import math

import numpy as np
import pandas as pd
from eeg_mne_ops import welch_psd as welch

from eeg_raw_bdf import NativeBDF


def inspect_spectrum(path, options):
    """按所选原始时间窗计算MNE Welch PSD；包括已有坏道和坏段，不预先滤波。"""
    source = NativeBDF(path)
    names = options.get("channels")
    if not isinstance(names, list) or not 1 <= len(names) <= 64 or len(set(names)) != len(names):
        raise ValueError("PSD需选择1–64个不同通道。")
    lookup = {r["name"]: r for r in source.signals}
    if any(n not in lookup for n in names):
        raise ValueError("PSD通道不存在。")
    full = options.get("scope", "window") == "full"
    if options.get("scope", "window") not in ("window", "full"):
        raise ValueError("PSD范围应为当前时间窗或全记录。")
    start = 0.0 if full else float(options.get("start", 0))
    seconds = source.duration if full else float(options.get("seconds", 4))
    window = float(options.get("welch_seconds", 4))
    low = float(options.get("fmin", 0.5))
    high = float(options.get("fmax", 100))
    if (
        not all(math.isfinite(v) for v in (start, seconds, window, low, high))
        or not 0 <= start < source.duration
        or seconds <= 0
        or not 0.25 <= window <= 10
        or not 0 <= low < high
    ):
        raise ValueError("PSD时间、频率或Welch窗参数无效；Welch窗需0.25–10秒。")
    stop = min(source.duration, start + seconds)
    curves = []
    summary = []
    notes = []
    if stop - start < window:
        notes.append(
            "实际时间窗短于设定Welch窗，已按可用样本缩短，实际分辨率见表。未去均值的大直流偏置可能影响最低频点，不能直接视为脑节律。"
        )
    from eeg_raw_qc import line_ratio

    for name in names:
        row = lookup[name]
        fs = row["fs"]
        a = math.floor(start * fs)
        b = min(source.records * row["spr"], math.ceil(stop * fs))
        data = source.read(row, a, b)
        unit = (
            "µV"
            if row["scale"] is not None
            else "digital counts" if row["digital_only"] else row["unit"] or "unspecified"
        )
        if row["scale"] is not None:
            data = data * 1e6
        n = min(len(data), max(2, round(window * fs)))
        if n < 2:
            notes.append(f"{name}样本不足，无法估计PSD。")
            continue
        overlap = n // 2
        f, p = welch(
            data,
            fs=fs,
            window="hann",
            nperseg=n,
            noverlap=overlap,
            nfft=n,
            detrend=False,
            scaling="density",
        )
        mask = (f >= low) & (f <= high)
        if not mask.any():
            notes.append(f"{name}的Nyquist为{fs/2:g}Hz，所选频率范围内没有谱点。")
        curves.append(
            {
                "name": name,
                "fs": fs,
                "unit": unit,
                "frequency": f[mask].tolist(),
                "power": p[mask].tolist(),
                "flat": bool(np.ptp(data) == 0),
                "nyquist": fs / 2,
                "df": fs / n,
                "nperseg": n,
                "noverlap": overlap,
                "windows": 1 + (len(data) - n) // (n - overlap),
            }
        )
        summary.append(
            {
                "通道": name,
                "采样率_Hz": fs,
                "单位": unit + "²/Hz",
                "原始样本数": len(data),
                "Welch窗样本": n,
                "原始均值": float(np.mean(data)),
                "原始标准差": float(np.std(data)),
                "频率分辨率_Hz": fs / n,
                "50Hz突出_dB": line_ratio(f, p, 50),
                "60Hz突出_dB": line_ratio(f, p, 60),
            }
        )
    return {
        "kind": "view_psd",
        "start": start,
        "stop": stop,
        "scope": "full" if full else "window",
        "fmin": low,
        "fmax": high,
        "curves": curves,
        "summary": summary,
        "warnings": notes,
        "method": "Welch：Hann窗、50%重叠、detrend=False、不补零；包含当前原始坏道/坏段。不同单位分图；只估计PSD，不滤波、不改写BDF/FIF。",
    }


def export_spectrum(result, output):
    """把当前PSD的线性数值和参数写成CSV，不把仅用于画图的dB截底写进数值表。"""
    path = output / "当前频域视图_Welch数值.csv"
    rows = [
        {
            "通道": c["name"],
            "采样率_Hz": c["fs"],
            "频率_Hz": f,
            "PSD": p,
            "单位": c["unit"] + "²/Hz",
            "起点_s": result["start"],
            "终点_s": result["stop"],
            "Welch窗样本": c["nperseg"],
            "重叠样本": c["noverlap"],
            "去均值": False,
        }
        for c in result["curves"]
        for f, p in zip(c["frequency"], c["power"])
    ]
    pd.DataFrame(rows or [{"说明": "当前频率范围无可估计的谱点"}]).to_csv(
        path, index=False, encoding="utf-8-sig"
    )
    return str(path)
