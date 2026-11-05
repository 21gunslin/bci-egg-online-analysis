"""辅助巡检：在整份原始记录里找可疑通道和时间窗，供人工优先复核。扫描不是诊断，也不自动删除数据。

Read-only triage for manual artifact review, not automatic rejection.

Rules flag flat/rail, unusually large excursions/steps, baseline shifts and
line-noise candidates. No filtering, interpolation, calibration correction,
Raw/FIF writes or diagnostic labels occur here.
"""

from collections import defaultdict
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from eeg_mne_ops import welch_psd as welch

from eeg_raw_bdf import NativeBDF

VERSION = "patrol_v1"
NOTICE = "候选仅用于优先定位，可能漏检或误报；未触发不代表无伪迹。不能自动确认眨眼、肌电或疾病。辅助通道本可包含尖峰或固定状态，不能直接当脑电伪迹。扫描不滤波、不删数据，不设置坏道/BAD注释。"


def scan_record(path, mapping, *, seconds=2.0, sensitivity=6.0, scope="eeg"):
    """用平线、饱和、跳变、基线和工频规则寻找复核候选；规则触发不代表已确认坏道。"""
    source = NativeBDF(path)
    if (
        not math.isfinite(seconds)
        or not 1 <= seconds <= 120
        or not math.isfinite(sensitivity)
        or not 3 <= sensitivity <= 12
        or scope not in ("eeg", "all")
    ):
        raise ValueError("巡检分窗须1–120秒，阈值3–12，范围eeg/all。")
    from eeg_raw_display import ordered_channels

    order = ordered_channels([r["name"] for r in source.signals], mapping)
    selected = [r for r in order if scope == "all" or r["eeg"]]
    bins = math.ceil(source.duration / seconds)
    if not selected:
        raise ValueError("没有可识别的EEG候选通道，可切换“全部信号”扫描。")
    if bins > 2000 or bins * len(selected) > 100000:
        raise ValueError("巡检单元过多，请增大分窗；单份最多2000个时间窗、10万个通道×时间单元。")
    lookup = {r["name"]: r for r in source.signals}
    cells = []
    candidates = []
    channel_summary = []
    for display in selected:
        row = lookup[display["name"]]
        fs = row["fs"]
        digital = source.read(row, digital=True)
        x = source.read(row) * (1e6 if row["scale"] is not None else 1.0)
        quantum = (
            1.0
            if row["digital_only"]
            else abs((row["pmax"] - row["pmin"]) / (row["dmax"] - row["dmin"]))
            * (row["scale"] * 1e6 if row["scale"] is not None else 1.0)
        )
        quantum = max(quantum, 1e-12)
        delta = np.abs(np.diff(x))
        dmed = float(np.median(delta)) if len(delta) else 0.0
        dmad = float(np.median(np.abs(delta - dmed))) * 1.4826 if len(delta) else 0.0
        jump_threshold = dmed + 2 * sensitivity * max(dmad, quantum, dmed * 0.2)
        current = []
        for i in range(bins):
            start, end = i * seconds, min(source.duration, (i + 1) * seconds)
            a, b = round(start * fs), min(len(x), round(end * fs))
            values = x[a:b]
            if not len(values):
                current.append(
                    {
                        "channel": row["name"],
                        "bin": i,
                        "start": start,
                        "end": end,
                        "reasons": [],
                        "score": 0.0,
                        "evaluable": False,
                    }
                )
                continue
            center = float(np.median(values))
            ptp = float(np.ptp(values))
            sd = float(np.std(values))
            rail = float(np.mean((digital[a:b] <= row["dmin"]) | (digital[a:b] >= row["dmax"])))
            dstart = max(0, a - 1)
            differences = delta[dstart : max(dstart, b - 1)]
            max_step = float(differences.max()) if len(differences) else 0.0
            step_time = (
                (dstart + int(np.argmax(differences)) + 1) / fs if len(differences) else start
            )
            peak_time = (a + int(np.argmax(np.abs(values - center)))) / fs
            reasons = []
            score = 0.0
            line_db = None
            line_fraction = None
            if ptp <= quantum * 0.1:
                reasons.append("平线")
                score = max(score, 8.0)
            if rail >= 0.01:
                reasons.append("数字边界/饱和候选")
                score = max(score, 9.0)
            if max_step > jump_threshold:
                reasons.append("瞬时跳变候选")
                score = max(score, min(10.0, 5 + math.log10(max_step / jump_threshold)))
                peak_time = step_time
            if len(values) >= max(16, round(fs)) and fs >= 112 and ptp > 0:
                n = min(len(values), max(16, round(2 * fs)))
                f, p = welch(
                    values, fs=fs, nperseg=n, noverlap=n // 2, window="hann", detrend=False
                )
                positive = f >= max(1.0, 2 * fs / n)
                total = float(p[positive].sum())
                tests = []
                for hz in (50, 60):
                    distance = np.abs(f - hz)
                    band = distance <= max(1.0, fs / n)
                    background = (distance >= 3) & (distance <= 6)
                    if (
                        f[-1] < hz + 6
                        or not band.any()
                        or not background.any()
                        or np.median(p[background]) <= 0
                        or total <= 0
                    ):
                        continue
                    db = float(
                        10
                        * (
                            np.log10(max(float(p[band].max()), 1e-300))
                            - np.log10(np.median(p[background]))
                        )
                    )
                    fraction = float(p[band].sum() / total)
                    tests.append((db, fraction))
                    if db >= 15 and fraction >= 0.2:
                        reasons.append(f"{hz}Hz强工频候选")
                        score = max(score, 6.0)
                if tests:
                    line_db, line_fraction = max(tests)
            current.append(
                {
                    "channel": row["name"],
                    "bin": i,
                    "start": float(start),
                    "end": float(end),
                    "reasons": reasons,
                    "score": score,
                    "evaluable": True,
                    "ptp": ptp,
                    "std": sd,
                    "median": center,
                    "rail_fraction": rail,
                    "max_step": max_step,
                    "jump_threshold": jump_threshold,
                    "line_db": line_db,
                    "line_fraction": line_fraction,
                    "focus_time": float(peak_time),
                }
            )
        valid = [c for c in current if c["evaluable"]]
        log_amp = np.array([math.log10(max(c["ptp"], quantum)) for c in valid])
        log_med = float(np.median(log_amp))
        log_mad = max(0.15, float(np.median(np.abs(log_amp - log_med))) * 1.4826)
        centers = np.array([c["median"] for c in valid])
        shifts = np.abs(np.diff(centers))
        shift_med = float(np.median(shifts)) if len(shifts) else 0.0
        shift_mad = float(np.median(np.abs(shifts - shift_med))) * 1.4826 if len(shifts) else 0.0
        shift_floor = max(quantum, float(np.median([c["ptp"] for c in valid])) * 0.05)
        for index, c in enumerate(valid):
            z = (log_amp[index] - log_med) / log_mad
            c["amplitude_robust_z"] = float(z)
            if len(valid) >= 5 and z >= sensitivity:
                c["reasons"].append("片段幅度离群候选")
                c["score"] = max(c["score"], min(10.0, float(z)))
            if index and abs(
                c["median"] - valid[index - 1]["median"]
            ) > shift_med + sensitivity * max(shift_mad, shift_floor):
                c["reasons"].append("基线突变候选")
                c["score"] = max(c["score"], 7.0)
        flags = defaultdict(list)
        for c in valid:
            for reason in c["reasons"]:
                flags[reason].append(c)
        # Persistent problems become one channel-level row instead of hundreds of alarms.
        persistent = []
        for reason, matches in flags.items():
            coverage = len(matches) / len(valid)
            groups = [matches] if coverage >= 0.8 else []
            if coverage >= 0.8:
                persistent.append(reason)
            else:
                for cell in matches:
                    if groups and cell["bin"] == groups[-1][-1]["bin"] + 1:
                        groups[-1].append(cell)
                    else:
                        groups.append([cell])
            for group in groups:
                worst = max(
                    group, key=lambda c: c["max_step"] if reason == "瞬时跳变候选" else c["score"]
                )
                candidates.append(
                    {
                        "channel": row["name"],
                        "reason": reason,
                        "start": group[0]["start"],
                        "end": group[-1]["end"],
                        "focus_time": worst["focus_time"],
                        "score": max(c["score"] for c in group),
                        "persistent": coverage >= 0.8,
                        "fraction": coverage,
                        "windows": len(group),
                        "eeg": display["eeg"],
                        "region": display["region"],
                    }
                )
        channel_summary.append(
            {
                **display,
                "candidate_windows": sum(bool(c["reasons"]) for c in valid),
                "windows": len(valid),
                "persistent_reasons": persistent,
            }
        )
        cells.append(current)
    candidates.sort(key=lambda r: (r["persistent"], -r["score"], r["start"], r["channel"]))
    for index, candidate in enumerate(candidates):
        candidate["id"] = f"{VERSION}_{index}"
    return {
        "version": VERSION,
        "notice": NOTICE,
        "duration": source.duration,
        "seconds": seconds,
        "scope": scope,
        "sensitivity": sensitivity,
        "channels": channel_summary,
        "bins": bins,
        "cells": cells,
        "candidates": candidates,
        "rules": [
            "平线：峰峰值≤0.1个文件头标定量化步长。",
            "饱和候选：数字上下边界占比≥1%。",
            "跳变：相邻样本差超过通道自身稳健背景，阈值=中位差+2×灵敏度×背景尺度。",
            "幅度离群：通道内log10峰峰值稳健z≥灵敏度，至少5窗；MAD设0.15下限。",
            "基线突变：相邻窗中位数差超过通道自身稳健背景。",
            "工频：50/60Hz峰相对邻域≥15dB且占max(1Hz,2Δf)以上谱功率≥20%；不去均值，仅在统计分母跳过DC邻格。",
            "同一原因触发≥80%时间窗时合并为持续性通道提示；未触发窗口仍需人工复核。",
        ],
    }


def export_scan(scan, directory):
    path = Path(directory) / "人工巡检_候选定位.csv"
    pd.DataFrame(
        [
            {
                "通道": c["channel"],
                "原因": c["reason"],
                "起点_s": c["start"],
                "终点_s": c["end"],
                "建议复核点_s": c["focus_time"],
                "持续性提示": c["persistent"],
                "触发窗口比例": c["fraction"],
                "说明": "候选，未自动设置坏道或删段",
            }
            for c in scan["candidates"]
        ]
        or [{"说明": "未触发候选规则，不等于无伪迹"}]
    ).to_csv(path, index=False, encoding="utf-8-sig")
    return str(path)


def review_state(directory, source, options):
    """单独登记人工实际浏览的通道和时间；程序扫描或自动翻页不能冒充人工复核。"""
    path = Path(directory) / "manual_review_state.json"
    state = json.loads(path.read_text(encoding="utf8")) if path.is_file() else {"entries": []}
    action = options.get("action", "get")
    if action == "append":
        names = options.get("channels")
        allowed = {r["name"] for r in source.signals}
        a, b = float(options["start"]), float(options["end"])
        status = options.get("status", "reviewed")
        if (
            not isinstance(names, list)
            or not names
            or len(set(names)) != len(names)
            or any(n not in allowed for n in names)
            or status not in ("reviewed", "uncertain")
            or not all(math.isfinite(t) for t in (a, b))
            or not 0 <= a < b <= source.duration
        ):
            raise ValueError("人工复核范围、通道或状态无效。")
        if len(state["entries"]) >= 10000:
            raise ValueError("复核记录超过10000条，请导出后分开管理。")
        state["entries"].append({"channels": names, "start": a, "end": b, "status": status})
    elif action == "undo":
        if state["entries"]:
            state["entries"].pop()
    elif action != "get":
        raise ValueError("未知复核操作。")
    if action != "get":
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(state, ensure_ascii=False), encoding="utf8")
        temporary.replace(path)
    return state
