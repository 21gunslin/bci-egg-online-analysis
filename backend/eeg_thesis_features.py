"""论文特征方案：按已写明的公式计算固定287列。独立连续段分别计算，缺失值不补成零。

Versioned paper feature profile; all input voltages are physical microvolts.

Windows never cross rejected epochs or file boundaries. Missing values remain NaN.
This module does not diagnose depression or assign clinical meanings to events.
"""

from __future__ import annotations

import json
from pathlib import Path
import warnings
from eeg_mne_ops import power_spectrum_windows

import numpy as np
import pandas as pd
from scipy import signal
from scipy.sparse.csgraph import shortest_path

PROFILE = "thesis_v1"
BANDS = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 45.0),
}
ROIS = {
    "额区": ("FP1", "FP2", "F3", "F4", "F7", "F8", "FZ", "AF3", "AF4"),
    "中央区": ("C3", "C4", "CZ", "C1", "C2"),
    "顶区": ("P3", "P4", "PZ", "P1", "P2"),
    "颞区": ("T3", "T4", "T5", "T6", "FT9", "FT10"),
    "枕区": ("O1", "O2", "OZ"),
}
PAIRS = (("Fp1", "Fp2"), ("F3", "F4"), ("F7", "F8"), ("C3", "C4"), ("T3", "T4"))
TIME_UNITS = dict(
    mean="µV",
    std="µV",
    variance="µV²",
    skewness="1",
    kurtosis="1",
    zero_crossing_rate="1",
    line_length="µV",
    rms="µV",
    peak_to_peak="µV",
    hjorth_activity="µV²",
    hjorth_mobility="s⁻¹",
    hjorth_complexity="1",
)
LABELS = dict(
    mean="均值",
    std="样本标准差",
    variance="样本方差",
    skewness="偏度",
    kurtosis="超额峰度",
    zero_crossing_rate="过零率",
    line_length="线长（差分绝对值总和）",
    rms="均方根",
    peak_to_peak="峰峰值",
    hjorth_activity="Hjorth活动性（等于方差）",
    hjorth_mobility="Hjorth移动性",
    hjorth_complexity="Hjorth复杂性",
    total_power="0.5–45Hz总功率",
    spectral_entropy="谱熵（自然对数、概率质量）",
    peak_alpha_frequency="峰值α频率",
    theta_beta_ratio="θ/β功率比",
    absolute_power="绝对功率",
    relative_power="相对功率",
    TF_std="频段功率的时间样本标准差",
    TF_cv="频段功率的时间变异系数",
    clustering="二值聚类系数",
    degree_mean="节点度均值",
    degree_std="节点度标准差",
    global_efficiency="加权全局效率",
    characteristic_path_length="可达对加权特征路径长度",
    connectivity_strength="阈值前平均PLV连接强度",
)


def describe(code):
    if code in LABELS:
        return LABELS[code]
    for band in BANDS:
        if code.startswith(band + "_"):
            return band + "频段 / " + LABELS.get(code[len(band) + 1 :], code)
    return code


METHOD = [
    "论文方案 thesis_v1：每时相固定287列（时域84、频域103、时频70、网络30）；缺失列保留为空。",
    "7种汇总：保留通道均值、保留通道标准差(ddof=0)、额区、中央区、顶区、颞区、枕区均值；不是7种独立特征。",
    "时域：逐2秒质控合格片段计算12项，再对片段等权平均。标准差/方差用N−1；偏度/超额峰度用中心矩均值除以样本标准差的幂，不作偏差校正。",
    "过零率采用sign公式，零点贡献半次；线长为差分绝对值之和；Hjorth导数为diff(x)×fs，移动性单位s⁻¹，不能直接等同Hz；Activity与variance数值重复。",
    "Welch：2秒周期Hann窗、50%重叠、常量去趋势、density、nfft=窗长，Δf=0.5Hz；按独立连续段的窗口数合并PSD。",
    "频段为[0.5,4)、[4,8)、[8,13)、[13,30)、[30,45]Hz；功率为ΣPSD×Δf，五段之和等于总功率。",
    "谱熵：0.5–45Hz内p=PSD×Δf/总功率，使用自然对数、未归一化；θ/β分母为零时缺失。",
    "FAA=ln(右α功率)−ln(左α功率)，固定五对；缺电极或功率≤0时缺失。C3/C4与T3/T4分别为中央/颞区α不对称，并非额叶。T7/T8映射T3/T4；P7/P8映射T5/T6。",
    "STFT：2秒周期Hann窗、50%重叠、常量去趋势、单边PSD校正、不补零；先Σ频段PSD×Δf得到功率时间序列，再计算时间标准差(ddof=1)和CV=标准差/均值；少于2个窗口时缺失。",
    "数值零：频段功率不高于同窗总功率×100×float64机器精度时置为0，避免纯信号频段外舍入噪声产生虚假的CV/比值；零均值CV和零分母比值记为缺失。",
    "多份/断续记录：时域汇集有效2秒片段；Welch按窗口数合并；时频汇集各独立连续段的频段功率窗口后计算标准差/CV；不把各段CV再取均值。",
    "PLV：每段内独立四阶Butterworth双向带通与Hilbert；汇集复相位差均值后取模，不跨断点滤波；每段短于max(3秒,3个频段下界周期)时不参与该频段网络，减轻低频边缘误差。",
    "网络：每频段固定密度25%，按PLV由高到低保留round(密度×可能边数)条正权边；并列按通道索引稳定排序。聚类系数为二值局部聚类的节点均值；节点度均值/标准差(ddof=0)来自阈值后二值图。",
    "全局效率：Σ(1/dij)/[n(n−1)]，断连对贡献0；加权距离=1/PLV。特征路径长度仅对可达节点对取均值；连接强度为阈值前PLV上三角均值。固定密度下度均值受节点数与密度约束，不能视作独立生理证据。",
    "电压按文件头单位换算，保留采集参考；未验证额外÷1000校准。相对功率、CV与PLV更少受统一幅值缩放影响，但仍受伪迹、参考和片段选择影响。输出用于探索性研究，不是PPD诊断或已验证风险概率。",
]


def canonical(label):
    from eeg_feature_engine import _normalize_channel_name

    label = _normalize_channel_name(label)
    return {"T7": "T3", "T8": "T4", "P7": "T5", "P8": "T6"}.get(label, label)


def continuous_blocks(pre):
    """Return views of true retained runs; validate the provenance before using it."""
    data = np.asarray(pre["data"], dtype=float)
    if "kept_epoch_indices" not in pre:
        return [data]
    indices = np.asarray(pre["kept_epoch_indices"], dtype=int)
    samples = int(pre["original_epoch_samples"])
    if samples <= 0 or len(indices) * samples != data.shape[1] or np.any(np.diff(indices) <= 0):
        raise ValueError("论文方案的保留片段索引与数据长度不一致，请重新分析。")
    edges = np.r_[0, np.flatnonzero(np.diff(indices) != 1) + 1, len(indices)]
    return [
        data[:, int(a * samples) : int(b * samples)] for a, b in zip(edges[:-1], edges[1:]) if b > a
    ]


def divide(a, b):
    a, b = np.broadcast_arrays(np.asarray(a, float), np.asarray(b, float))
    return np.divide(a, b, out=np.full(a.shape, np.nan), where=np.isfinite(b) & (b > 0))


def mean(values, axis=None):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(values, axis=axis)


def time_features(x, fs):
    """x is channels × epochs × samples; exact finite-sample conventions."""
    mu = x.mean(axis=-1)
    variance = x.var(axis=-1, ddof=1)
    std = np.sqrt(variance)
    dx = np.diff(x, axis=-1) * fs
    mobility = np.sqrt(divide(dx.var(axis=-1, ddof=1), variance))
    derivative_mobility = np.sqrt(
        divide((np.diff(dx, axis=-1) * fs).var(axis=-1, ddof=1), dx.var(axis=-1, ddof=1))
    )
    values = dict(
        mean=mu,
        std=std,
        variance=variance,
        skewness=divide(((x - mu[..., None]) ** 3).mean(axis=-1), std**3),
        kurtosis=divide(((x - mu[..., None]) ** 4).mean(axis=-1), std**4) - 3,
        zero_crossing_rate=(np.abs(np.diff(np.sign(x), axis=-1)) / 2).mean(axis=-1),
        line_length=np.abs(np.diff(x, axis=-1)).sum(axis=-1),
        rms=np.sqrt((x * x).mean(axis=-1)),
        peak_to_peak=np.ptp(x, axis=-1),
        hjorth_activity=variance,
        hjorth_mobility=mobility,
        hjorth_complexity=divide(derivative_mobility, mobility),
    )
    return {name: mean(value, axis=1) for name, value in values.items()}


def spectral_windows(blocks, fs):
    """按2秒Hann窗做短时傅里叶变换并换成PSD；保留每个窗口，供后续频段统计。"""
    size = int(round(2 * fs))
    if not np.isclose(size / fs, 2):
        raise ValueError("论文方案要求采样率可构成完整2秒窗。")
    transforms = []
    for block in blocks:
        if block.shape[1] < size:
            continue
        # 不手写单边谱的“乘2”校正；MNE公开接口统一处理。
        f, t, power = power_spectrum_windows(block, fs, size, size // 2)
        transforms.append((f, t, power))
    if not transforms:
        raise ValueError("论文方案没有完整的2秒功率谱窗口。")
    return transforms


def band_mask(f, band):
    lo, hi = BANDS[band]
    return (f >= lo) & (f <= hi if band == "gamma" else f < hi)


def network_metrics(matrix, density=0.25):
    n = len(matrix)
    upper = np.triu_indices(n, 1)
    values = np.asarray(matrix)[upper]
    wanted = int(np.floor(density * len(values) + 0.5))
    chosen = np.argsort(-values, kind="stable")[:wanted]
    chosen = chosen[np.isfinite(values[chosen]) & (values[chosen] > 0)]
    adjacency = np.zeros((n, n), dtype=bool)
    adjacency[upper[0][chosen], upper[1][chosen]] = True
    adjacency |= adjacency.T
    binary = adjacency.astype(float)
    degrees = binary.sum(axis=1)
    cluster = np.divide(
        np.diag(binary @ binary @ binary),
        degrees * (degrees - 1),
        out=np.zeros(n),
        where=degrees > 1,
    )
    lengths = np.full((n, n), np.inf)
    lengths[adjacency] = 1 / np.asarray(matrix)[adjacency]
    np.fill_diagonal(lengths, 0)
    distances = shortest_path(lengths, directed=False)
    off_diagonal = ~np.eye(n, dtype=bool)
    finite = off_diagonal & np.isfinite(distances) & (distances > 0)
    efficiency = np.sum(1 / distances[finite]) / (n * (n - 1))
    return dict(
        clustering=float(cluster.mean()),
        degree_mean=float(degrees.mean()),
        degree_std=float(degrees.std()),
        global_efficiency=float(efficiency),
        characteristic_path_length=float(distances[finite].mean()) if finite.any() else np.nan,
        connectivity_strength=float(values.mean()),
    ), {
        "nodes": n,
        "edges": len(chosen),
        "density": 2 * len(chosen) / (n * (n - 1)),
        "reachable_fraction": float(finite.sum() / (n * (n - 1))),
    }


def network(blocks, fs, settings, band):
    """按当前频段计算独立连续段的PLV，再统计网络，不跨文件或BAD边界滤波。"""
    lo, hi = BANDS[band]
    usable = [b for b in blocks if b.shape[1] / fs >= max(3.0, 3 / lo)]
    n = blocks[0].shape[0]
    if not usable or band not in settings.network_bands:
        return None, {
            "reason": "无足够长连续段（至少3秒且覆盖下界3个周期），或该域未启用",
            "samples": 0,
        }
    total = sum(b.shape[1] for b in usable)
    stride = max(1, int(np.ceil(total / settings.max_network_samples)))
    sos = signal.butter(4, [lo, hi], btype="bandpass", fs=fs, output="sos")
    accumulated = np.zeros((n, n), dtype=complex)
    count = 0
    for block in usable:
        filtered = signal.sosfiltfilt(sos, block, axis=-1)
        phases = np.exp(1j * np.angle(signal.hilbert(filtered, axis=-1)))[:, ::stride]
        accumulated += phases @ phases.conj().T
        count += phases.shape[1]
    matrix = np.clip(np.abs(accumulated / count), 0, 1)
    np.fill_diagonal(matrix, 0)
    return matrix, {"samples": count, "stride": stride, "continuous_runs": len(usable)}


def extract_collection(prepared, settings, progress=lambda *_: None):
    """论文方案：汇集独立连续段的合格窗口，按固定定义计算287列，缺失保持NaN。"""
    fs = float(prepared[0]["fs"])
    names = list(prepared[0]["channel_names"])
    if settings.epoch_seconds != 2:
        raise ValueError("论文方案使用固定2秒质控片段，请把片段长度设为2秒。")
    if any(p["channel_names"] != names or not np.isclose(p["fs"], fs) for p in prepared):
        raise ValueError("论文方案汇集时要求通道顺序和采样率一致。")
    blocks = [b for p in prepared for b in continuous_blocks(p)]
    size = int(round(2 * fs))
    epochs = np.concatenate(
        [
            b[:, : b.shape[1] // size * size].reshape(len(names), -1, size)
            for b in blocks
            if b.shape[1] >= size
        ],
        axis=1,
    )
    channel = {f"时域__{key}": value for key, value in time_features(epochs, fs).items()}
    transforms = spectral_windows(blocks, fs)
    f = transforms[0][0]
    df = f[1] - f[0]
    spectrum = np.concatenate([p for _, _, p in transforms], axis=-1)
    psd = spectrum.mean(axis=-1)
    total_mask = (f >= 0.5) & (f <= 45)
    total = psd[:, total_mask].sum(axis=1) * df
    band_series = {band: spectrum[:, band_mask(f, band), :].sum(axis=1) * df for band in BANDS}
    window_total = spectrum[:, total_mask, :].sum(axis=1) * df
    for band in BANDS:
        band_series[band] = np.where(
            band_series[band] > window_total * np.finfo(float).eps * 100, band_series[band], 0.0
        )
    for band in BANDS:
        absolute = band_series[band].mean(axis=-1)
        channel[f"频域__{band}_absolute_power"] = absolute
        channel[f"频域__{band}_relative_power"] = divide(absolute, total)
        variation = (
            band_series[band].std(axis=-1, ddof=1)
            if spectrum.shape[-1] >= 2
            else np.full(len(names), np.nan)
        )
        channel[f"时频域__{band}_TF_std"] = variation
        channel[f"时频域__{band}_TF_cv"] = divide(variation, band_series[band].mean(axis=-1))
    mass = divide(psd[:, total_mask] * df, total[:, None])
    with np.errstate(divide="ignore", invalid="ignore"):
        entropy = -np.where(mass > 0, mass * np.log(mass), 0).sum(axis=1)
    entropy[~np.isfinite(total) | (total <= 0)] = np.nan
    alpha_mask = band_mask(f, "alpha")
    alpha_peak = f[alpha_mask][psd[:, alpha_mask].argmax(axis=1)]
    alpha_peak[channel["频域__alpha_absolute_power"] <= 0] = np.nan
    channel.update(
        {
            "频域__total_power": total,
            "频域__spectral_entropy": entropy,
            "频域__peak_alpha_frequency": alpha_peak,
            "频域__theta_beta_ratio": divide(
                channel["频域__theta_absolute_power"], channel["频域__beta_absolute_power"]
            ),
        }
    )
    canonical_names = [canonical(c) for c in names]
    roi_indices = {
        roi: [i for i, c in enumerate(canonical_names) if c in electrodes]
        for roi, electrodes in ROIS.items()
    }
    summary, dictionary = {}, []

    def add(key, value, unit, definition, reason=""):
        summary[key] = float(value)
        dictionary.append(
            {
                "特征": key,
                "单位": unit,
                "定义": definition,
                "缺失原因": reason if not np.isfinite(value) else "",
            }
        )

    for key, values in channel.items():
        code = key.split("__")[1]
        unit = (
            TIME_UNITS[code]
            if key.startswith("时域__")
            else (
                "µV²"
                if "absolute_power" in key or code == "total_power" or code.endswith("TF_std")
                else "Hz" if code == "peak_alpha_frequency" else "1"
            )
        )
        scopes = {
            "全脑均值": list(range(len(names))),
            "全脑标准差": list(range(len(names))),
            **roi_indices,
        }
        for scope, indices in scopes.items():
            selected = values[indices]
            valid = selected[np.isfinite(selected)]
            value = (
                (valid.std() if scope == "全脑标准差" else valid.mean()) if len(valid) else np.nan
            )
            add(
                f"{key}__{scope}",
                value,
                unit,
                f"{describe(code)}；{scope}；有限值通道数{len(valid)}/{len(indices)}",
                (
                    "该脑区无保留电极"
                    if not indices
                    else "该指标在全部所选通道均不可计算（零方差/零分母/时频窗口不足）"
                ),
            )
    alpha = channel["频域__alpha_absolute_power"]
    for left, right in PAIRS:
        pair = [np.flatnonzero(np.asarray(canonical_names) == canonical(c)) for c in (left, right)]
        value = np.nan
        reason = "指定电极缺失或别名重复，不能唯一匹配"
        if all(len(indices) == 1 for indices in pair):
            l, r = [alpha[indices[0]] for indices in pair]
            reason = "α功率非正或不可计算"
            if l > 0 and r > 0:
                value = np.log(r) - np.log(l)
        add(f"频域__FAA__{left}_{right}", value, "1", "ln(右α绝对功率)−ln(左α绝对功率)", reason)
    matrices, network_details, network_features = {}, {}, {}
    for band in BANDS:
        progress(0.72, f"论文方案：计算{band}网络")
        matrix, details = network(blocks, fs, settings, band)
        if matrix is None:
            metrics = dict.fromkeys(
                (
                    "clustering",
                    "degree_mean",
                    "degree_std",
                    "global_efficiency",
                    "characteristic_path_length",
                    "connectivity_strength",
                ),
                np.nan,
            )
        else:
            matrices[band] = matrix
            metrics, topology = network_metrics(matrix, settings.density)
            details.update(topology)
        network_details[band] = details
        for key, value in metrics.items():
            code = f"{band}_{key}"
            network_features[code] = value
            add(
                f"网络__{code}",
                value,
                "1",
                f"{describe(code)}；见方案中的图定义",
                details.get("reason", "无可达节点对"),
            )
    assert len(summary) == 287
    table = pd.DataFrame(
        {"通道": names, **channel, "单位": ["µV"] * len(names), "分析采样率_Hz": fs}
    )
    metadata = {
        "channels": names,
        "bands": BANDS,
        "network_features": network_features,
        "global_features": {},
        "feature_categories": {
            domain: [k for k in summary if k.startswith(prefix)]
            for domain, prefix in [
                ("时域", "时域__"),
                ("频域", "频域__"),
                ("时频域", "时频域__"),
                ("脑网络", "网络__"),
            ]
        },
        "definition": METHOD,
        "dictionary": dictionary,
        "coverage": {
            "epochs": epochs.shape[1],
            "psd_tf_windows": spectrum.shape[-1],
            "continuous_runs": len(blocks),
            "roi_channels": {
                roi: [names[i] for i in indices] for roi, indices in roi_indices.items()
            },
            "network": network_details,
        },
        "_spectra": (f, psd),
        "_transforms": transforms,
        "_matrices": matrices,
    }
    return table, pd.DataFrame({"特征": list(summary), "数值": list(summary.values())}), metadata


def export_profile(summary, metadata, output, prefix="论文方案"):
    from eeg_research_appendix import make_appendix, write_appendix

    output = Path(output)
    wide = output / f"{prefix}_特征宽表.csv"
    pd.DataFrame([summary.set_index("特征")["数值"].to_dict()]).to_csv(
        wide, index=False, encoding="utf-8-sig"
    )
    dictionary = output / f"{prefix}_特征字典与缺失原因.csv"
    pd.DataFrame(metadata["dictionary"]).to_csv(dictionary, index=False, encoding="utf-8-sig")
    method = output / f"{prefix}_计算口径.txt"
    method.write_text(
        "\n\n".join(METHOD)
        + "\n\n实际覆盖：\n"
        + json.dumps(metadata["coverage"], ensure_ascii=False, indent=2),
        encoding="utf8",
    )
    metadata["research_appendix"] = make_appendix(summary, metadata, prefix)
    return [
        str(wide),
        str(dictionary),
        str(method),
        *write_appendix(metadata["research_appendix"], output, prefix),
    ]


def plot_profile(prepared, channel_table, metadata, output, prefix="论文方案"):
    """Plots use the exact same PSD, STFT windows and PLV as the exported values."""
    from eeg_pair_spike import plotting

    plt = plotting()
    output = Path(output)
    files = []

    def save(fig, name):
        path = output / f"{prefix}_{name}.png"
        fig.savefig(path, dpi=140, bbox_inches="tight")
        plt.close(fig)
        files.append(path)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    f, psd = metadata["_spectra"]
    axes[0].semilogy(f, np.maximum(psd.mean(axis=0), 1e-20))
    axes[0].set(
        xlim=(0.5, 45), xlabel="Hz", ylabel="PSD (µV²/Hz)", title="2秒Hann窗 / 50%重叠 / Welch平均"
    )
    axes[1].bar(list(BANDS), [channel_table[f"频域__{b}_relative_power"].mean() for b in BANDS])
    axes[1].set(ylabel="相对功率 (0–1)", title="共同保留通道平均频段功率")
    save(fig, "频域特征")
    transforms = metadata["_transforms"]
    visible = np.concatenate([p[0, f <= 45, :].ravel() for f, _, p in transforms])
    db = 10 * np.log10(np.maximum(visible, 1e-20))
    lo, hi = np.percentile(db, [2, 98])
    hi = max(hi, lo + 1)
    fig, ax = plt.subplots(figsize=(12, 4), constrained_layout=True)
    cursor = 0
    for f, t, power in transforms:
        # Explicit cell edges also support the single-window (2-second) case.
        edges = np.r_[t - 0.5, t[-1] + 0.5] + cursor
        frequency_edges = np.r_[f - 0.25, f[-1] + 0.25]
        image = ax.pcolormesh(
            edges,
            frequency_edges,
            10 * np.log10(np.maximum(power[0], 1e-20)),
            shading="flat",
            cmap="magma",
            vmin=lo,
            vmax=hi,
        )
        cursor += t[-1] + 1
        ax.axvline(cursor, color="cyan", linestyle="--", linewidth=0.6)
    ax.set(
        ylim=(0.5, 45),
        xlabel="独立连续段展示坐标 (s)，虚线为断点",
        ylabel="Hz",
        title=metadata["channels"][0] + " · STFT单边PSD；不跨断点",
    )
    fig.colorbar(image, ax=ax, label="PSD (dB re 1 µV²/Hz)")
    save(fig, "时频图")
    fs = prepared[0]["fs"]
    blocks = [b for p in prepared for b in continuous_blocks(p)]
    fig, ax = plt.subplots(figsize=(12, 4), constrained_layout=True)
    cursor = 0
    for block in blocks:
        step = max(1, block.shape[1] // 5000)
        ax.plot(cursor + np.arange(0, block.shape[1], step) / fs, block[0, ::step], lw=0.6)
        cursor += block.shape[1] / fs
    ax.set(
        xlabel="有效时间展示坐标 (s)，各段独立绘制",
        ylabel="µV",
        title=metadata["channels"][0] + " · 质控保留波形",
    )
    save(fig, "时域波形")
    for band, matrix in metadata["_matrices"].items():
        fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
        image = ax.imshow(matrix, vmin=0, vmax=1, cmap="viridis")
        ax.set_xticks(range(len(matrix)), metadata["channels"], rotation=90, fontsize=7)
        ax.set_yticks(range(len(matrix)), metadata["channels"], fontsize=7)
        ax.set_title(f"{band} PLV · 独立段复相位汇集")
        fig.colorbar(image, ax=ax)
        save(fig, f"网络_{band}")
    return files


def compare_profile(a, b, output, label_a, label_b, channel, max_seconds):
    from eeg_feature_engine import AnalysisSettings
    from eeg_pair_spike import plotting

    output = Path(output)
    with np.load(a, allow_pickle=False) as za, np.load(b, allow_pickle=False) as zb:
        if str(zb.get("feature_profile", "standard")) != PROFILE:
            raise ValueError("两次记录的特征方案不一致，请重新分析。")
        fs = float(za["fs"])
        if not np.isclose(fs, float(zb["fs"])):
            raise ValueError("两份记录的分析采样率不同。")
        labels = [za["labels"].tolist(), zb["labels"].tolist()]
        common = [c for c in labels[0] if c in labels[1]]
        if len(common) < 3:
            raise ValueError("共同保留通道不足3个。")
        selected = channel or common[0]
        if selected not in common:
            raise ValueError("指定通道不在共同保留通道中。")
        size = int(round(2 * fs))
        count = min(za["data"].shape[1], zb["data"].shape[1], int(max_seconds * fs)) // size * size
        if count < size:
            raise ValueError("共同可用时长不足2秒。")
        settings = AnalysisSettings(**json.loads(str(za["settings_json"])))
        prepared = []
        for z, names in zip((za, zb), labels):
            if any(str(u).lower() not in ("µv", "μv", "uv") for u in z["units"]):
                raise ValueError("论文方案缓存单位须为µV，请重新分析。")
            prepared.append(
                {
                    "data": z["data"][[names.index(c) for c in common], :count].copy(),
                    "fs": fs,
                    "channel_names": common,
                    "kept_epoch_indices": z["kept_epoch_indices"][: count // size].copy(),
                    "original_epoch_samples": int(z["original_epoch_samples"]),
                }
            )
    results = [extract_collection([pre], settings) for pre in prepared]
    series = [r[1].set_index("特征")["数值"] for r in results]
    merged = pd.concat(series, axis=1, keys=["记录A", "记录B"])
    merged["变化量_B减A"] = merged["记录B"] - merged["记录A"]
    merged["相对变化_pct"] = np.where(
        merged["记录A"].abs() > 1e-8, merged["变化量_B减A"] / merged["记录A"].abs() * 100, np.nan
    )
    merged.insert(0, "特征域", [k.split("__")[0] for k in merged.index])
    reasons = [{row["特征"]: row["缺失原因"] for row in r[2]["dictionary"]} for r in results]
    merged["A缺失原因"] = [reasons[0][k] for k in merged.index]
    merged["B缺失原因"] = [reasons[1][k] for k in merged.index]
    merged.index.name = "特征"
    rows = merged.reset_index()
    csv = output / "论文方案_配对特征变化.csv"
    rows.to_csv(csv, index=False, encoding="utf-8-sig")
    # The phases are labels supplied by the user, not inferred from file clocks.
    wide = {
        f"{phase}__{key}": value
        for phase, values in [
            ("pre", series[0]),
            ("post", series[1]),
            ("delta", series[1] - series[0]),
        ]
        for key, value in values.items()
    }
    assert len(wide) == 861
    wide_path = output / "论文方案_镇痛前后及差值_861列.csv"
    pd.DataFrame([wide]).to_csv(wide_path, index=False, encoding="utf-8-sig")
    files = [str(csv), str(wide_path)]
    for r, prefix in zip(results, ("A", "B")):
        files.extend(export_profile(r[1], r[2], output, "论文方案_" + prefix))
    from eeg_research_appendix import make_paired_appendix, write_appendix

    research_appendix = make_paired_appendix(
        rows.to_dict("records"), results[0][2], results[1][2], label_a, label_b
    )
    files.extend(write_appendix(research_appendix, output, "论文方案_配对"))
    plt = plotting()

    def save(fig, name):
        path = output / f"论文方案_对比_{name}.png"
        fig.savefig(path, dpi=140, bbox_inches="tight")
        plt.close(fig)
        files.append(str(path))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    xx = np.arange(5)
    for i, (table, _, metadata) in enumerate(results):
        f, psd = metadata["_spectra"]
        title = (label_a, label_b)[i]
        axes[0].semilogy(f, np.maximum(psd.mean(axis=0), 1e-20), label=title)
        axes[1].bar(
            xx + (i - 0.5) * 0.35,
            [table[f"频域__{band}_relative_power"].mean() for band in BANDS],
            width=0.35,
            label=title,
        )
    axes[0].set(xlim=(0.5, 45), xlabel="Hz", ylabel="PSD (µV²/Hz)", title="相同参数Welch功率谱")
    axes[0].legend()
    axes[1].set_xticks(xx, list(BANDS))
    axes[1].set(ylabel="相对功率 (0–1)")
    axes[1].legend()
    save(fig, "频域")
    idx = common.index(selected)
    all_power = [
        10 * np.log10(np.maximum(power[idx, f <= 45], 1e-20)).ravel()
        for _, _, meta in results
        for f, _, power in meta["_transforms"]
    ]
    lo, hi = np.percentile(np.concatenate(all_power), [2, 98])
    hi = max(hi, lo + 1)
    fig, axes = plt.subplots(
        2, 1, figsize=(12, 7), sharex=True, sharey=True, constrained_layout=True
    )
    for ax, (_, _, meta), title in zip(axes, results, (label_a, label_b)):
        cursor = 0
        for f, t, power in meta["_transforms"]:
            image = ax.pcolormesh(
                np.r_[t - 0.5, t[-1] + 0.5] + cursor,
                np.r_[f - 0.25, f[-1] + 0.25],
                10 * np.log10(np.maximum(power[idx], 1e-20)),
                shading="flat",
                cmap="magma",
                vmin=lo,
                vmax=hi,
            )
            cursor += t[-1] + 1
            ax.axvline(cursor, color="cyan", linestyle="--", linewidth=0.6)
        ax.set(ylim=(0.5, 45), ylabel="Hz", title=title + " · " + selected)
    axes[-1].set_xlabel("独立连续段展示坐标 (s)，不表示事件同步")
    fig.colorbar(image, ax=axes, label="PSD (dB re 1 µV²/Hz)，统一色标")
    save(fig, "时频")
    fig, axes = plt.subplots(
        2, 1, figsize=(12, 5), sharex=True, sharey=True, constrained_layout=True
    )
    for ax, pre, title in zip(axes, prepared, (label_a, label_b)):
        cursor = 0
        for block in continuous_blocks(pre):
            step = max(1, block.shape[1] // 5000)
            ax.plot(cursor + np.arange(0, block.shape[1], step) / fs, block[idx, ::step], lw=0.7)
            cursor += block.shape[1] / fs
        ax.set(title=title + " · " + selected, ylabel="µV")
    axes[-1].set_xlabel("有效时间展示坐标 (s)，各段独立绘制")
    save(fig, "时域")
    for band in BANDS:
        matrices = [result[2]["_matrices"].get(band) for result in results]
        if any(m is None for m in matrices):
            continue
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
        for i, (ax, matrix, title) in enumerate(
            zip(axes, [*matrices, matrices[1] - matrices[0]], [label_a, label_b, "B−A"])
        ):
            image = ax.imshow(
                matrix, vmin=-1 if i == 2 else 0, vmax=1, cmap="RdBu_r" if i == 2 else "viridis"
            )
            ax.set_xticks(range(len(common)), common, rotation=90, fontsize=7)
            ax.set_yticks(range(len(common)), common, fontsize=7)
            ax.set_title(band + " PLV · " + title)
            fig.colorbar(image, ax=ax)
        save(fig, "网络_" + band)
    return {
        "kind": "comparison",
        "metrics": {
            "common_channels": len(common),
            "duration_seconds": count / fs,
            "sampling_rate": fs,
            "channel": selected,
        },
        "labels": common,
        "label_a": label_a,
        "label_b": label_b,
        "rows": rows.to_dict("records"),
        "files": files,
        "feature_definition": METHOD,
        "research_appendix": research_appendix,
        "warnings": [
            "论文方案固定287项配对；导出pre/post/delta共861列；缺失值保留，不删除整行。pre/post按你指定的A/B标签解释。",
            "每份取相同数量的完整有效2秒片段，保留各自断点；不是按事件匹配的同步试次。",
        ],
    }
