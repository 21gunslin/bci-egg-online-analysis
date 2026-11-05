"""第4站：把波形变成特征数值。

时域看波形起伏，频域看不同频率的功率，时频看功率随时间变化，网络看通道相位同步。"""

from __future__ import annotations

from eeg_mne_ops import welch_psd

import math
from typing import Any, Sequence
import numpy as np
import pandas as pd
from scipy import signal
from scipy.stats import kurtosis, skew
from eeg_models import (
    AnalysisSettings,
    DEFAULT_BANDS,
    ROI_ALIASES,
    FEATURE_DESCRIPTIONS,
    ProgressCallback,
    default_progress,
)
from eeg_readers import _normalize_channel_name
from eeg_preprocessing import _bandpass


def _windowed_features(data: np.ndarray) -> dict[str, np.ndarray]:
    """时域指标：直接对波形算均值、方差、RMS和Hjorth，不先做频率变换。标准方案和论文方案的公式口径见各自说明。"""
    diff = np.diff(data, axis=1)
    second_diff = np.diff(diff, axis=1)
    variance = np.var(data, axis=1)
    diff_variance = np.var(diff, axis=1)
    mobility = np.sqrt(diff_variance / (variance + 1e-12))
    complexity = np.sqrt(np.var(second_diff, axis=1) / (diff_variance + 1e-12)) / (mobility + 1e-12)
    return {
        "mean": np.mean(data, axis=1),
        "std": np.std(data, axis=1),
        "variance": variance,
        "skewness": skew(data, axis=1),
        "kurtosis": kurtosis(data, axis=1),
        "zero_crossing_rate": np.mean(np.diff(np.signbit(data), axis=1), axis=1),
        "line_length": np.mean(np.abs(diff), axis=1),
        "rms": np.sqrt(np.mean(data**2, axis=1)),
        "peak_to_peak": np.max(data, axis=1) - np.min(data, axis=1),
        "hjorth_activity": variance,
        "hjorth_mobility": mobility,
        "hjorth_complexity": complexity,
    }


def _frequency_features(
    data: np.ndarray, fs: float, bands: dict[str, tuple[float, float]]
) -> dict[str, np.ndarray]:
    """频域指标：先用MNE Welch得到“频率→功率”的表，再对各频段累加。PSD乘频率间隔才是频段功率。"""
    nperseg = int(min(1024, 2 ** math.floor(math.log2(max(16, data.shape[1])))))
    frequencies, psd = welch_psd(data, fs=fs, nperseg=nperseg, axis=1)
    df = float(frequencies[1] - frequencies[0])
    analysis_mask = (frequencies >= 0.5) & (frequencies <= min(45.0, fs / 2))
    total = np.sum(psd[:, analysis_mask], axis=1) * df + 1e-20
    psd_mass = np.sum(psd, axis=1) + 1e-20
    features: dict[str, np.ndarray] = {
        "total_power": total,
        "mean_frequency": np.sum(frequencies[None, :] * psd, axis=1) / psd_mass,
        "spectral_entropy": -np.sum(
            (psd / psd_mass[:, None]) * np.log(psd / psd_mass[:, None] + 1e-20), axis=1
        )
        / np.log(psd.shape[1]),
    }
    for name, (low, high) in bands.items():
        mask = (frequencies >= low) & (frequencies < high)
        if not np.any(mask):
            features[f"{name}_absolute_power"] = np.zeros(data.shape[0])
            features[f"{name}_relative_power"] = np.zeros(data.shape[0])
            continue
        band_power = np.sum(psd[:, mask], axis=1) * df
        features[f"{name}_absolute_power"] = band_power
        features[f"{name}_relative_power"] = band_power / total
    for name in bands:
        if f"{name}_absolute_power" not in features:
            continue
        features[f"{name}_power_ratio"] = features[f"{name}_absolute_power"] / (
            total - features[f"{name}_absolute_power"] + 1e-20
        )

    alpha_mask = (frequencies >= 8.0) & (frequencies < 13.0)
    if np.any(alpha_mask):
        alpha_psd = psd[:, alpha_mask]
        features["peak_alpha_frequency"] = frequencies[alpha_mask][np.argmax(alpha_psd, axis=1)]
    if "theta_absolute_power" in features and "beta_absolute_power" in features:
        features["theta_beta_ratio"] = features["theta_absolute_power"] / (
            features["beta_absolute_power"] + 1e-20
        )
    if "alpha_absolute_power" in features and "beta_absolute_power" in features:
        features["alpha_beta_ratio"] = features["alpha_absolute_power"] / (
            features["beta_absolute_power"] + 1e-20
        )
    return features


def _time_frequency_features(
    data: np.ndarray,
    fs: float,
    bands: dict[str, tuple[float, float]],
) -> dict[str, np.ndarray]:
    """时频指标：使用SciPy的短时傅里叶变换观察频段功率随时间的变化；不把它替换成参数不同的其他变换。"""
    window_size = int(min(512, 2 ** math.floor(math.log2(max(32, data.shape[1] // 8)))))
    overlap = int(window_size * 0.75)
    if window_size >= data.shape[1]:
        window_size = int(2 ** math.floor(math.log2(max(32, data.shape[1] // 2))))
        overlap = window_size // 2
    if window_size <= 4:
        return {}

    frequencies, times, transform = signal.stft(
        data,
        fs=fs,
        window="hann",
        nperseg=window_size,
        noverlap=overlap,
        axis=1,
        boundary=None,
        padded=False,
    )
    power = np.abs(transform) ** 2
    features: dict[str, np.ndarray] = {}
    for name, (low, high) in bands.items():
        mask = (frequencies >= low) & (frequencies < high)
        if not np.any(mask):
            continue
        band_time = np.sum(power[:, mask, :], axis=1)
        mean = np.mean(band_time, axis=1)
        std = np.std(band_time, axis=1)
        features[f"{name}_tf_mean"] = mean
        features[f"{name}_tf_std"] = std
        features[f"{name}_tf_cv"] = std / (np.abs(mean) + 1e-20)
        features[f"{name}_tf_range"] = np.max(band_time, axis=1) - np.min(band_time, axis=1)
        threshold = np.percentile(band_time, 90, axis=1, keepdims=True)
        features[f"{name}_tf_burst_ratio"] = np.mean(band_time >= threshold, axis=1)
    if power.shape[2] >= 2:
        log_power = np.log(power + 1e-20)
        features["tf_spectral_flux"] = np.mean(np.abs(np.diff(log_power, axis=2)), axis=(1, 2))
    return features


def plv_matrix(
    data: np.ndarray,
    fs: float,
    low_hz: float,
    high_hz: float,
    max_samples: int = 60_000,
) -> np.ndarray:
    """PLV相位同步：每对通道比较相位差是否稳定，结果是通道×通道的0到1矩阵，不是脑源定位。"""
    if data.shape[0] < 2:
        return np.zeros((data.shape[0], data.shape[0]), dtype=float)
    nyquist = fs / 2.0
    high = min(high_hz, nyquist * 0.98)
    if low_hz >= high:
        return np.zeros((data.shape[0], data.shape[0]), dtype=float)
    filtered = _bandpass(data, fs, low_hz, high)
    analytic = signal.hilbert(filtered, axis=1)
    magnitude = np.abs(analytic) + 1e-20
    phases = analytic / magnitude
    if phases.shape[1] > max_samples:
        # Subsample phases after the Hilbert transform, never alias the voltage signal.
        phases = phases[:, np.linspace(0, phases.shape[1] - 1, max_samples, dtype=int)]
    plv = np.abs(phases @ phases.conj().T) / phases.shape[1]
    np.fill_diagonal(plv, 0.0)
    return np.clip(np.real(plv), 0.0, 1.0)


def graph_metrics(weights: np.ndarray, density: float = 0.25) -> dict[str, float]:
    """网络指标：把通道看成节点、同步强度看成边，按写明的阈值和距离定义计算；不额外引入图论库。"""
    n = weights.shape[0]
    if n < 2:
        return {
            "weighted_density": 0.0,
            "density": 0.0,
            "mean_strength": 0.0,
            "strength_std": 0.0,
            "mean_degree": 0.0,
            "degree_std": 0.0,
            "clustering": 0.0,
            "global_efficiency": 0.0,
            "characteristic_path_length": 0.0,
            "small_worldness": 0.0,
        }
    adjacency = weights.copy()
    np.fill_diagonal(adjacency, 0.0)
    upper = adjacency[np.triu_indices(n, 1)]
    positive = upper[np.isfinite(upper) & (upper > 0)]
    weighted_density = float(np.mean(positive)) if positive.size else 0.0

    threshold = float(np.quantile(upper, 1.0 - density)) if upper.size else 0.0
    binary = (adjacency >= threshold).astype(float)
    np.fill_diagonal(binary, 0.0)
    degree = np.sum(binary, axis=1)
    clustering = np.zeros(n, dtype=float)
    for node in range(n):
        neighbors = np.flatnonzero(binary[node])
        neighbor_count = len(neighbors)
        if neighbor_count > 1:
            subgraph = binary[np.ix_(neighbors, neighbors)]
            clustering[node] = np.sum(subgraph) / (neighbor_count * (neighbor_count - 1))

    distance = np.full((n, n), np.inf, dtype=float)
    np.fill_diagonal(distance, 0.0)
    edges = binary > 0
    distance[edges] = 1.0 / (adjacency[edges] + 1e-12)
    distance = np.minimum(distance, distance.T)
    for pivot in range(n):
        distance = np.minimum(distance, distance[:, pivot, None] + distance[pivot, :][None, :])
    finite = np.isfinite(distance) & (distance > 0)
    efficiency = float(np.mean(1.0 / distance[finite])) if np.any(finite) else 0.0
    path_length = float(np.mean(distance[finite])) if np.any(finite) else 0.0
    clustering_mean = float(np.mean(clustering))
    random_clustering = weighted_density
    small_world = clustering_mean / random_clustering if random_clustering > 0 else 0.0

    return {
        "weighted_density": weighted_density,
        "density": float(np.mean(binary[np.triu_indices(n, 1)])),
        "mean_strength": float(np.mean(adjacency)),
        "strength_std": float(np.std(adjacency)),
        "mean_degree": float(np.mean(degree)),
        "degree_std": float(np.std(degree)),
        "clustering": clustering_mean,
        "global_efficiency": efficiency,
        "characteristic_path_length": path_length,
        "small_worldness": small_world,
    }


def _network_features(
    data: np.ndarray,
    fs: float,
    bands: dict[str, tuple[float, float]],
    settings: AnalysisSettings,
    progress: ProgressCallback,
    base_progress: float,
    span: float,
) -> dict[str, float]:
    features: dict[str, float] = {}
    usable_bands = [name for name in settings.network_bands if name in bands]
    for index, name in enumerate(usable_bands):
        low, high = bands[name]
        matrix = plv_matrix(data, fs, low, high, settings.max_network_samples)
        for metric, value in graph_metrics(matrix, settings.density).items():
            features[f"{name}_plv_{metric}"] = float(value)
        progress(
            base_progress + span * (index + 1) / max(1, len(usable_bands)),
            f"计算 {name} 频段脑网络",
        )
    return features


def aggregate_features(
    channel_features: dict[str, np.ndarray],
    channel_names: Sequence[str],
    network_features: dict[str, float] | None = None,
    global_features: dict[str, float] | None = None,
) -> dict[str, float]:
    """按全脑、额区等范围汇总通道指标。脑区里没有可用电极时保留缺失，不虚构数值。"""
    summary: dict[str, float] = {}
    labels = [_normalize_channel_name(name) for name in channel_names]
    positions = {label: index for index, label in enumerate(labels)}

    for feature_name, values in channel_features.items():
        array = np.asarray(values, dtype=float)
        summary[f"{feature_name}__全脑均值"] = float(np.nanmean(array))
        summary[f"{feature_name}__全脑标准差"] = float(np.nanstd(array))
        summary[f"{feature_name}__全脑中位数"] = float(np.nanmedian(array))
        for region, aliases in ROI_ALIASES.items():
            indices = [positions[alias] for alias in aliases if alias in positions]
            if indices:
                summary[f"{feature_name}__{region}_均值"] = float(np.nanmean(array[indices]))

    pairs = (
        ("Fp1", "Fp2"),
        ("F3", "F4"),
        ("F7", "F8"),
        ("C3", "C4"),
        ("T3", "T4"),
        ("T5", "T6"),
        ("O1", "O2"),
    )
    for power_name in ("alpha_absolute_power", "alpha_relative_power", "alpha__全脑均值"):
        if power_name not in channel_features:
            continue
        array = np.asarray(channel_features[power_name], dtype=float)
        for left, right in pairs:
            if left in positions and right in positions:
                summary[f"FAA__{left}_{right}"] = float(
                    np.log(array[positions[right]] + 1e-20) - np.log(array[positions[left]] + 1e-20)
                )

    if network_features:
        summary.update({f"网络__{name}": float(value) for name, value in network_features.items()})
    if global_features:
        summary.update({f"全局__{name}": float(value) for name, value in global_features.items()})
    return summary


def extract_features(
    preprocessed: dict[str, Any],
    settings: AnalysisSettings,
    progress: ProgressCallback = default_progress,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """选择当前特征方案并依次计算各域，返回通道级表、汇总表和计算说明。"""
    if settings.feature_profile == "thesis_v1":
        from eeg_thesis_features import extract_collection

        return extract_collection([preprocessed], settings, progress)
    data = np.asarray(preprocessed["data"], dtype=float)
    fs = float(preprocessed["fs"])
    channel_names = list(preprocessed["channel_names"])
    bands = settings.bands

    progress(0.38, "计算时域特征")
    time_features = _windowed_features(data)
    progress(0.48, "计算频域特征")
    frequency_features = _frequency_features(data, fs, bands)
    progress(0.60, "计算时频域特征")
    time_frequency_features = _time_frequency_features(data, fs, bands)

    channel_features: dict[str, np.ndarray] = {}
    channel_features.update({f"时域__{key}": value for key, value in time_features.items()})
    channel_features.update({f"频域__{key}": value for key, value in frequency_features.items()})
    channel_features.update(
        {f"时频域__{key}": value for key, value in time_frequency_features.items()}
    )

    channel_table = pd.DataFrame({"通道": channel_names})
    for name, values in channel_features.items():
        channel_table[name] = np.asarray(values, dtype=float)
    channel_table["单位"] = preprocessed.get("channel_units", [""] * len(channel_names))
    channel_table["分析采样率_Hz"] = fs
    channel_table["使用时长_s"] = preprocessed.get("duration_seconds", math.nan)

    progress(0.72, "计算脑网络特征")
    network_features = _network_features(
        data, fs, bands, settings, progress, base_progress=0.72, span=0.20
    )

    global_features = {
        "平均标准差": float(np.mean(channel_table["时域__std"])),
        "中位标准差": float(np.median(channel_table["时域__std"])),
        "平均谱熵": float(np.mean(channel_table["频域__spectral_entropy"])),
        "通道间谱熵变异": float(np.std(channel_table["频域__spectral_entropy"])),
        "平均峰值Alpha频率": float(
            np.mean(channel_table.get("频域__peak_alpha_frequency", pd.Series([np.nan])))
        ),
        "平均Hjorth复杂度": float(np.mean(channel_table["时域__hjorth_complexity"])),
        "平均ThetaBeta比": float(
            np.mean(channel_table.get("频域__theta_beta_ratio", pd.Series([np.nan])))
        ),
    }

    summary = aggregate_features(channel_features, channel_names, network_features, global_features)
    summary = {
        "分析采样率_Hz": fs,
        "使用时长_s": float(preprocessed.get("duration_seconds", math.nan)),
        "保留通道数": float(len(channel_names)),
        "片段保留比例": float(preprocessed.get("keep_ratio", math.nan)),
        "干净片段数": float(preprocessed.get("epoch_count", 0)),
        "总片段数": float(preprocessed.get("epoch_total", 0)),
        **summary,
    }
    summary_table = pd.DataFrame([summary]).T.reset_index()
    summary_table.columns = ["特征", "数值"]

    metadata = {
        "channels": channel_names,
        "bands": bands,
        "network_features": network_features,
        "global_features": global_features,
        "feature_categories": {
            "时域": [name for name in channel_features if name.startswith("时域__")],
            "频域": [name for name in channel_features if name.startswith("频域__")],
            "时频域": [name for name in channel_features if name.startswith("时频域__")],
            "脑网络": [f"网络__{name}" for name in network_features],
            "汇总": list(summary.keys()),
        },
    }
    progress(0.95, "整理特征表")
    return channel_table, summary_table, metadata
