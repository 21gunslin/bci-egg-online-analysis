"""查看坐标与图片导出：自动调横纵轴帮助看清波形，不改变原始数值。

Axis-only viewing parameters and snapshots. No Raw/FIF processing calls."""

import math
from pathlib import Path

import numpy as np
import pandas as pd
from eeg_mne_ops import welch_psd as welch


def nice_ceil(value):
    if not math.isfinite(value) or value <= 0:
        return 1.0
    order = 10.0 ** math.floor(math.log10(value))
    return next(n * order for n in (1, 2, 5, 10) if n * order >= value * (1 - 1e-12))


def axis_settings(rows, mode="channel", gain=1.0, fixed_step=100.0):
    if (
        mode not in ("channel", "shared", "fixed")
        or not math.isfinite(gain)
        or not 0.001 <= gain <= 100
        or not math.isfinite(fixed_step)
        or fixed_step <= 0
    ):
        raise ValueError("纵轴模式、增益或手动标尺无效。")
    prepared = []
    shared = {}
    for row in rows:
        values = np.array([p[1] for p in row["points"]])
        stats = row.get("stats") or {
            "min": float(values.min()),
            "max": float(values.max()),
            "median": float(np.median(values)),
            "ptp": float(np.ptp(values)),
        }
        center = stats["median"]
        extent = max(abs(stats["max"] - center), abs(stats["min"] - center))
        prepared.append((row, stats, center, extent))
        shared[row["unit"]] = max(shared.get(row["unit"], 0), extent)
    result = []
    for row, stats, center, extent in prepared:
        if mode == "shared":
            extent = shared[row["unit"]]
        floor = 1.0 if row["unit"] == "µV" else max(abs(center) * 1e-12, 1e-6)
        step = (
            fixed_step
            if mode == "fixed" and row["unit"] == "µV"
            else nice_ceil(max(extent / 1.25, floor))
        ) / gain
        half_range = step * 1.4
        result.append(
            {
                "通道": row["name"],
                "单位": row["unit"],
                "纵轴模式": mode,
                "中心原值": center,
                "每格幅度": step,
                "纵轴最小": center - half_range,
                "纵轴最大": center + half_range,
                "原始最小": stats["min"],
                "原始最大": stats["max"],
                "峰峰值": stats["ptp"],
                "平线": stats["ptp"] == 0,
                "超出显示范围": stats["min"] < center - half_range
                or stats["max"] > center + half_range,
            }
        )
    return result


def viewing_defaults(path, mapping):
    """Choose a readable initial time span from dominant frequencies, not labels."""
    from eeg_raw_bdf import NativeBDF

    source = NativeBDF(path)
    by_name = {r["原始通道"]: r for r in mapping["rows"]}
    peaks = []
    for row in source.signals:
        if row["scale"] is None or by_name.get(row["name"], {}).get("MNE类型") != "eeg":
            continue
        x = source.read(row, 0, min(source.records * row["spr"], math.ceil(10 * row["fs"])))
        if len(x) < 4 or np.ptp(x) == 0:
            continue
        n = min(len(x), max(4, round(2 * row["fs"])))
        f, p = welch(x, fs=row["fs"], nperseg=n, noverlap=n // 2, window="hann", detrend=False)
        mask = (f >= 1) & (f <= min(100, row["fs"] / 2))
        if mask.any() and np.max(p[mask]) > 0:
            peaks.append(float(f[mask][np.argmax(p[mask])]))
        if len(peaks) >= 6:
            break
    dominant = float(np.median(peaks)) if peaks else None
    target = max(0.1, min(10, 40 / dominant)) if dominant else 10
    seconds = min(source.duration, next(t for t in (0.1, 0.2, 0.5, 1, 2, 5, 10) if t >= target))
    return {
        "mode": "channel",
        "seconds": seconds,
        "dominant_frequency_hz": dominant,
        "notice": "建议窗长按前10秒中若干EEG通道的主导频率选择，仅用于看清波形，主导成分可能是工频。纵轴默认适配当前窗全部幅度。",
    }


def export_view(
    wave,
    output,
    *,
    mode="channel",
    gain=1.0,
    fixed_step=100.0,
    bads=(),
    bad_segments=(),
    prefix="当前原始波形视图",
):
    """Export exactly the chosen display window and axes; never save a FIF."""
    from eeg_pair_spike import plotting

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    plt = plotting()
    axes = axis_settings(wave["rows"], mode, float(gain), float(fixed_step))
    files = []
    for page, begin in enumerate(range(0, len(wave["rows"]), 32), 1):
        rows = wave["rows"][begin : begin + 32]
        settings = axes[begin : begin + 32]
        fig, plot = plt.subplots(figsize=(15, max(5, len(rows) * 0.62 + 1.7)))
        fig.subplots_adjust(left=0.18, right=0.8, bottom=0.09, top=0.91)
        centers = np.arange(len(rows))[::-1] * 3.0
        for i, (row, axis, center) in enumerate(zip(rows, settings, centers)):
            points = np.asarray(row["points"])
            normalized = (points[:, 1] - axis["中心原值"]) / axis["每格幅度"]
            plot.axhspan(center - 1.5, center + 1.5, color="#f5f8fb" if i % 2 == 0 else "#fff")
            for tick in (-1, 0, 1):
                plot.axhline(center + tick, color="#d7e1eb", lw=0.5, ls="-" if tick == 0 else ":")
            plot.plot(
                points[:, 0],
                center + np.clip(normalized, -1.4, 1.4),
                lw=0.65,
                color="#c34234" if row["name"] in bads else "#214b68",
            )
            unit = row["unit"]
            label = f"{axis['每格幅度']:.5g} {unit}/格\n范围 {axis['纵轴最小']:.6g}～{axis['纵轴最大']:.6g}\n中心 {axis['中心原值']:.6g}"
            plot.text(
                1.018, center, label, transform=plot.get_yaxis_transform(), fontsize=7, va="center"
            )
            if axis["平线"]:
                plot.text(
                    0.99,
                    center + 0.8,
                    "原始平线",
                    transform=plot.get_yaxis_transform(),
                    ha="right",
                    fontsize=7,
                    color="#b34235",
                )
            elif axis["超出显示范围"]:
                plot.text(
                    0.99,
                    center + 1.1,
                    "超出显示范围（仅绘图）",
                    transform=plot.get_yaxis_transform(),
                    ha="right",
                    fontsize=7,
                    color="#b34235",
                )
        for interval in bad_segments:
            a, b = max(wave["start"], interval["start"]), min(wave["stop"], interval["end"])
            if a < b:
                plot.axvspan(a, b, color="#e9947c", alpha=0.15)
        labels = [r["name"] + (" [BAD]" if r["name"] in bads else "") for r in rows]
        plot.set_yticks(centers, labels, fontsize=9)
        plot.set_ylim(-1.6, centers[0] + 1.6)
        plot.set_xlim(wave["start"], wave["stop"])
        plot.set_xlabel("原始相对时间 (s)")
        plot.grid(axis="x", alpha=0.2)
        for side in ("left", "right", "top"):
            plot.spines[side].set_visible(False)
        label = {
            "channel": "逐通道完整范围适配（各行刻度不同）",
            "shared": "同单位统一尺度适配",
            "fixed": f"电压手动标尺 {fixed_step:g} µV/格",
        }[mode]
        fig.suptitle(
            f'原始波形｜{label}｜{wave["start"]:g}–{wave["stop"]:g} s\n仅调横纵坐标与显示中心；不滤波、不改原始采样或FIF',
            fontsize=12,
        )
        fig.text(
            0.18,
            0.025,
            "额区→枕区；振幅须结合数值标尺比较。完整范围适配保留极值；平线仍为平线。",
            fontsize=8,
            color="#556a7e",
        )
        path = output / (f"{prefix}.png" if page == 1 else f"{prefix}_{page}.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        files.append(str(path))
    table = output / f"{prefix}_坐标参数.csv"
    pd.DataFrame(
        [
            {
                **a,
                "时间起点_s": wave["start"],
                "时间终点_s": wave["stop"],
                "额外数据处理": "无，仅显示设置",
            }
            for a in axes
        ]
    ).to_csv(table, index=False, encoding="utf-8-sig")
    files.append(str(table))
    return files
