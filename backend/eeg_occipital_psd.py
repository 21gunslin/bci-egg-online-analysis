"""枕区平均功率谱：分别计算实际O1/Oz/O2的PSD，再对线性功率平均，不先平均波形。

Occipital ROI power average, not waveform or dB average."""

from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


def mean_linear_power(curves):
    """Equal channel weights; callers average Welch windows within each channel."""
    if not curves:
        raise ValueError("平均功率谱需要至少一条可用通道谱。")
    freq = np.asarray(curves[0][0])
    if any(len(f) != len(freq) or not np.allclose(f, freq, rtol=0, atol=1e-9) for f, _ in curves):
        raise ValueError("平均PSD的频率网格不同；应按原采样率分组，不能静默插值。")
    return freq.copy(), np.mean([p for _, p in curves], axis=0)


def build_roi_spectra(full_spectra, conditions, eyes, bads):
    """One average per native rate and condition, with common paired channels."""
    groups = defaultdict(list)
    for name, (f, p, fs) in full_spectra.items():
        if name not in bads:
            groups[fs].append(name)
    records = []
    notes = []
    for fs, names in groups.items():
        f, psd = mean_linear_power([(full_spectra[n][0], full_spectra[n][1]) for n in names])
        records.append(
            {
                "state": "full_record",
                "fs": fs,
                "channels": names,
                "freq": f,
                "psd": psd,
                "windows": None,
                "scope": "全记录，保留BAD时段；排除人工坏道",
            }
        )
        per_state = {}
        for state in ("open", "closed"):
            channel_curves = {}
            for name in names:
                subset = [c for c in conditions.get(name, []) if c[0] == state]
                if subset:
                    frequency = subset[0][1]
                    total = sum(c[3] for c in subset)
                    power = sum(c[2] * c[3] for c in subset) / total
                    channel_curves[name] = (frequency, power, total)
            per_state[state] = channel_curves
        available = [s for s in ("open", "closed") if per_state[s]]
        paired = len(available) == 2
        if paired:
            common = [n for n in names if all(n in per_state[s] for s in available)]
            if not common:
                notes.append(f"{fs:g}Hz组没有睁闭眼均可用的共同枕区通道，未生成状态比较。")
                continue
        else:
            common = None
        for state in available:
            selected = common if paired else [n for n in names if n in per_state[state]]
            f, psd = mean_linear_power(
                [(per_state[state][n][0], per_state[state][n][1]) for n in selected]
            )
            records.append(
                {
                    "state": state,
                    "fs": fs,
                    "channels": selected,
                    "freq": f,
                    "psd": psd,
                    "windows": {n: per_state[state][n][2] for n in selected},
                    "scope": "每通道内按Welch窗数加权，再对共同通道等权平均；不跨BAD拼窗",
                }
            )
        if eyes and not paired:
            notes.append(
                f"{fs:g}Hz组只有{len(available)}种眼状态有足够清洁Welch窗，不构成完整睁闭眼对比。"
            )
    if len(groups) > 1:
        notes.append("枕区通道存在不同原采样率，分别展示均值，不对原信号或PSD插值。")
    return records, notes


def export_roi_spectra(full_spectra, conditions, eyes, bads, output, peak_function):
    from eeg_pair_spike import plotting

    output = Path(output)
    plt = plotting()
    records, notes = build_roi_spectra(full_spectra, conditions, eyes, bads)
    rates = sorted({r["fs"] for r in records})
    summary = []
    values = []
    states = {"full_record": "全记录平均", "open": "睁眼平均", "closed": "闭眼平均"}
    fig, axes = plt.subplots(
        max(1, len(rates)),
        2,
        figsize=(15, max(5, len(rates) * 4.5)),
        squeeze=False,
        constrained_layout=True,
    )
    for row_index, fs in enumerate(rates):
        full = next(r for r in records if r["fs"] == fs and r["state"] == "full_record")
        left, right = axes[row_index]
        left.plot(
            full["freq"], 10 * np.log10(np.maximum(full["psd"], 1e-300)), color="#334f68", lw=1.4
        )
        left.set(
            title=f'全记录枕区平均谱（{fs:g} Hz；{len(full["channels"])}通道）',
            xlabel="Frequency (Hz)",
            ylabel="10 log10(mean PSD / 1 µV²/Hz)",
            xlim=(0, fs / 2),
        )
        left.text(
            0.98,
            0.97,
            " + ".join(full["channels"]) + f' / {len(full["channels"])}\n含BAD时段；排除人工坏道',
            transform=left.transAxes,
            va="top",
            ha="right",
            fontsize=8,
        )
        for hz in (50, 60):
            if hz < fs / 2:
                left.axvline(hz, ls=":", lw=0.8, color="#b17a4a")
        chosen = [r for r in records if r["fs"] == fs and r["state"] in ("open", "closed")]
        if not eyes:
            chosen = [full]
        if not chosen:
            right.text(
                0.5,
                0.5,
                "已填写状态，但无足够清洁窗口\n不以全记录曲线冒充睁/闭眼",
                ha="center",
                va="center",
                transform=right.transAxes,
            )
        for record in chosen:
            f, p = record["freq"], record["psd"]
            state = record["state"]
            label = states[state] + f'（{len(record["channels"])}通道）'
            color = {"full_record": "#334f68", "open": "#c67b43", "closed": "#2479aa"}[state]
            right.plot(f, p, lw=1.8, color=color, label=label)
            peak = peak_function(f, p)
            if peak["detected"]:
                hz = peak["alpha_peak_hz"]
                index = int(np.argmin(np.abs(f - hz)))
                right.plot(hz, p[index], "o", color=color)
                right.annotate(
                    f"{hz:g} Hz",
                    (hz, p[index]),
                    xytext=(7, 8),
                    textcoords="offset points",
                    color=color,
                    fontsize=9,
                )
        right.axvspan(8, 13, alpha=0.08, color="#4a9b73")
        right.set(
            title="枕区平均功率谱：睁眼 / 闭眼" if eyes else "枕区平均功率谱：未区分眼状态",
            xlabel="Frequency (Hz)",
            ylabel="平均 PSD (µV²/Hz)",
            xlim=(0.5, min(30, fs / 2)),
            ylim=(0, None),
        )
        if chosen:
            right.legend(fontsize=8)
        for ax in (left, right):
            ax.grid(alpha=0.2)
    if not rates:
        for ax in axes[0]:
            ax.text(
                0.5,
                0.5,
                "无可确认、单位明确且未标坏道的O1/Oz/O2\n不生成虚构的枕区平均谱",
                transform=ax.transAxes,
                ha="center",
                va="center",
            )
    fig.suptitle(
        "枕区ROI平均Welch功率谱｜先逐通道算PSD，再平均线性功率；不平均原始波形或dB", fontsize=12
    )
    image = output / "枕区平均功率谱_睁闭眼对比.png"
    fig.savefig(image, dpi=160)
    plt.close(fig)
    for record in records:
        peak = peak_function(record["freq"], record["psd"])
        summary.append(
            {
                "状态": states[record["state"]],
                "原采样率_Hz": record["fs"],
                "平均通道": ", ".join(record["channels"]),
                "通道数": len(record["channels"]),
                "频率分辨率_Hz": (
                    float(np.diff(record["freq"][:2])[0]) if len(record["freq"]) > 1 else None
                ),
                "α峰_Hz": peak["alpha_peak_hz"],
                "α功率_µV2": peak["alpha_power_uv2"],
                "α峰突出_dB": peak["prominence_db"],
                "α峰检出": peak["detected"],
                "平均方法": record["scope"],
            }
        )
        values.extend(
            {
                "状态": states[record["state"]],
                "原采样率_Hz": record["fs"],
                "平均通道": ", ".join(record["channels"]),
                "频率_Hz": float(f),
                "平均PSD_µV2_Hz": float(p),
            }
            for f, p in zip(record["freq"], record["psd"])
        )
    csv = output / "枕区平均功率谱_数值.csv"
    peaks = output / "枕区平均功率谱_α峰.csv"
    pd.DataFrame(values or [{"说明": "无可用枕区平均谱"}]).to_csv(
        csv, index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(summary or [{"说明": "无可用枕区平均谱"}]).to_csv(
        peaks, index=False, encoding="utf-8-sig"
    )
    return {
        "rows": summary,
        "warnings": notes,
        "method": "逐通道Welch后线性功率等权平均；每通道内状态区间按Welch窗数加权，双状态使用相同共同通道。",
    }, [str(image), str(csv), str(peaks)]
