"""第3站：预处理和质控。先统一单位，再滤波，最后挑出可用通道与时间片段。

输入和输出都保留时间索引，不能把拼接后的时间当成原始记录时间。"""

from __future__ import annotations

import numpy as np
from scipy import signal
from typing import Any, Iterable, Sequence
from eeg_models import (
    AnalysisSettings,
    HeaderInfo,
    ProgressCallback,
    default_progress,
    EEGQualityError,
    NoUsableEpochsError,
)
from eeg_readers import voltage_to_uv, _normalize_channel_name


def infer_sampling_rate(header: HeaderInfo) -> float:
    if header.sampling_rate > 0:
        return float(header.sampling_rate)
    if header.record_duration > 0:
        return 0.0
    return 0.0


def _estimate_sampling_rate(
    data: np.ndarray,
    header: HeaderInfo,
    explicit_fs: float | None,
) -> tuple[float, list[str]]:
    notes: list[str] = []
    if explicit_fs and explicit_fs > 0:
        return float(explicit_fs), ["采样率由用户指定。"]

    if header.sampling_rate > 0:
        return float(header.sampling_rate), [
            f"采样率由文件头各通道采样点数推算：{header.sampling_rate:g} Hz。"
        ]

    if header.data_records > 0 and header.duration_seconds > 0:
        samples = data.shape[1]
        fs = samples / header.duration_seconds
        return float(fs), ["采样率由总时长与采样点数推算。"]

    raise ValueError("无法确定采样率；请在分析设置中手动指定。")


def _downsample(
    data: np.ndarray, source_fs: float, target_fs: float
) -> tuple[np.ndarray, float, list[str]]:
    """降低分析采样率前先做抗混叠处理；输入通道×时间，输出同形状含义的数据与实际采样率。"""
    if source_fs <= 0:
        raise ValueError("源采样率无效。")
    target = min(target_fs, source_fs)
    if np.isclose(source_fs, target, rtol=1e-6, atol=1e-6):
        return data.astype(np.float64, copy=False), source_fs, []

    integer_ratio = source_fs / target
    rounded_ratio = int(round(integer_ratio))
    if rounded_ratio >= 2 and np.isclose(integer_ratio, rounded_ratio, rtol=1e-6, atol=1e-6):
        decimated = signal.decimate(data, rounded_ratio, axis=1, ftype="fir", zero_phase=True)
        return (
            decimated,
            source_fs / rounded_ratio,
            [f"按 {rounded_ratio}:1 抗混叠抽取至 {source_fs / rounded_ratio:g} Hz。"],
        )

    number_samples = int(round(data.shape[1] * target / source_fs))
    if number_samples < 8:
        raise ValueError("数据过短，无法重采样。")
    resampled = signal.resample_poly(
        data, up=int(round(target * 1000)), down=int(round(source_fs * 1000)), axis=1
    )[:, :number_samples]
    return resampled, target, [f"使用 resample_poly 重采样至约 {target:g} Hz。"]


def _bandpass(data, fs, low_hz, high_hz):
    """带通滤波的兼容名字；具体MNE调用只维护在eeg_mne_ops中。"""
    from eeg_mne_ops import bandpass_zero_phase

    return bandpass_zero_phase(data, fs, low_hz, high_hz)


def _apply_notches(data, fs, notches):
    """按表单选择逐个抑制工频；超过Nyquist的频率跳过并保留原有规则。"""
    from eeg_mne_ops import notch_zero_phase

    result = data
    notes = []
    for frequency in notches:
        if frequency <= 0 or frequency >= fs / 2:
            continue
        result = notch_zero_phase(result, fs, frequency)
        notes.append(f"已施加 {frequency:g} Hz 工频陷波（Q=30，双向零相位）。")
    return result, notes


def preprocess_eeg(
    data: np.ndarray,
    header: HeaderInfo,
    settings: AnalysisSettings,
    explicit_fs: float | None = None,
    selected_channels: Iterable[str] | None = None,
    progress: ProgressCallback = default_progress,
) -> dict[str, Any]:
    """预处理主线：单位换算→滤波→通道质控→时间分段。返回数据和原始时间索引，便于追溯被保留的片段。"""
    progress(0.02, "读取文件并识别通道")
    source_fs, fs_notes = _estimate_sampling_rate(data, header, explicit_fs)
    labels, units = list(header.channel_labels or []), list(header.channel_units or [])
    if not labels or not units or len(labels) != data.shape[0]:
        raise ValueError("文件头中的分析通道标签不完整。")
    if selected_channels:
        wanted = {_normalize_channel_name(item) for item in selected_channels}
        indices = [i for i, label in enumerate(labels) if _normalize_channel_name(label) in wanted]
        if indices:
            data, labels, units = (
                data[indices],
                [labels[i] for i in indices],
                [units[i] for i in indices],
            )
        else:
            raise ValueError("指定通道不存在。")

    data = voltage_to_uv(data, units)
    source_units = units.copy()
    units = ["µV"] * len(labels)

    if data.shape[1] < int(source_fs):
        raise EEGQualityError(f"有效数据不足 1 秒（{data.shape[1]} 采样点 @ {source_fs:g} Hz）。")

    notes = list(fs_notes) + [
        "电压按文件头声明单位统一为 µV；未推断设备量纲误标或施加额外倍率。",
        "保留采集参考；文件头未验证参考电极，未自动重参考或推断电极坐标。",
    ]
    progress(0.10, "重采样与带通滤波")
    valid_samples = None
    if header.format == "FIF" and header.marked_segments:
        from eeg_fif import preprocess_marked

        data, fs, valid_samples, resample_notes = preprocess_marked(
            data, header, settings, source_fs
        )
    else:
        data, fs, resample_notes = _downsample(data, source_fs, settings.target_fs)
    notes.extend(resample_notes)
    filter_high_hz = min(settings.lowpass_hz, 45.0)
    if valid_samples is None:
        data = _bandpass(data, fs, 0.5, filter_high_hz)
    notes.append(
        f"已施加四阶 Butterworth 带通滤波（0.5–{filter_high_hz:g} Hz），"
        "采用 SOS 双向零相位滤波；N=4 原型的带通单程为8阶，双向有效16阶；合成幅频响应为 |H|²。"
    )
    if valid_samples is None:
        data, notch_notes = _apply_notches(data, fs, settings.notches)
    else:
        notch_notes = ["FIF未标记连续段分别执行相同的带通与所选陷波，BAD不参与滤波。"]
    if header.format == "FIF":
        notes.extend(header.notes)
        notes.append("FIF原有坏道：" + (", ".join(header.bad_channels or []) or "无"))
    notes.extend(notch_notes)

    progress(0.18, "通道质量控制")
    keep, channel_metrics = channel_qc(
        data if valid_samples is None else data[:, valid_samples], labels, settings
    )
    if len(keep) < settings.min_channels:
        raise EEGQualityError(
            f"有效通道仅 {len(keep)} 个，低于最小要求 {settings.min_channels}；"
            "请检查电极脱落、单位或预处理设置。"
        )
    data = data[keep]
    kept_labels = [labels[index] for index in keep]
    kept_units = [units[index] for index in keep]

    progress(0.28, "分段与伪迹剔除")
    clean_epochs_data, epochs = clean_epochs(data, settings, fs=fs, valid_samples=valid_samples)
    keep_ratio = float(np.mean(epochs)) if epochs.size else 0.0
    if clean_epochs_data.shape[2] == 0 or keep_ratio < settings.min_keep_ratio:
        guidance = (
            "有效数据不足 2 秒，请缩短片段长度或检查记录时长。"
            if epochs.size == 0
            else "请放宽伪迹阈值或检查原始数据。"
        )
        error_type = NoUsableEpochsError if clean_epochs_data.shape[2] == 0 else EEGQualityError
        raise error_type(
            f"自动质控后可用片段比例仅 {keep_ratio:.1%}，低于最小要求 "
            f"{settings.min_keep_ratio:.1%}；{guidance}"
        )
    # Collapse retained clean epochs back to channel × time for the feature
    # routines. The earlier 3-D array leaked an epoch dimension into Welch and
    # other feature calculations, causing incorrect shapes or broadcasting.
    clean = clean_epochs_data.reshape(clean_epochs_data.shape[0], -1)
    kept_epoch_indices = np.flatnonzero(epochs)
    original_epoch_samples = int(round(settings.epoch_seconds * fs))
    notes.append(
        "保留片段拼接用于探索性特征；跨剔除间隙的谱/时频/相位及导数可能受边界影响，不能视为连续原始时间。"
    )

    channel_records = []
    for original_index in keep:
        channel_records.append(
            {
                "原始序号": original_index + 1,
                "通道": labels[original_index],
                "单位": units[original_index],
                "状态": "保留",
                "标准差": float(channel_metrics["std"][original_index]),
                "是否饱和或异常": bool(channel_metrics["saturated"][original_index]),
            }
        )
    for original_index, label in enumerate(labels):
        if original_index not in keep:
            channel_records.append(
                {
                    "原始序号": original_index + 1,
                    "通道": label,
                    "单位": units[original_index],
                    "状态": "剔除",
                    "标准差": float(channel_metrics["std"][original_index]),
                    "是否饱和或异常": bool(channel_metrics["saturated"][original_index]),
                }
            )

    return {
        "data": clean,
        "fs": float(fs),
        "source_fs": float(source_fs),
        "channel_names": kept_labels,
        "channel_units": kept_units,
        "source_channel_units": source_units,
        "reference": "acquisition_reference_unverified",
        "kept_epoch_indices": kept_epoch_indices,
        "original_epoch_samples": original_epoch_samples,
        "raw_channel_names": labels,
        "channel_records": channel_records,
        "keep_indices": keep,
        "keep_ratio": keep_ratio,
        "epoch_count": int(epochs.sum()) if epochs.size else 0,
        "epoch_total": int(epochs.size),
        "notes": notes,
        "duration_seconds": float(clean.shape[1] / fs),
    }


def channel_qc(
    data: np.ndarray,
    channel_names: Sequence[str],
    settings: AnalysisSettings,
) -> tuple[list[int], dict[str, np.ndarray]]:
    """检查平线、极大波动和高度重复通道，返回保留序号以及每个通道的检查数值。"""
    std = np.std(data, axis=1)
    centered = data - np.mean(data, axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        correlation = np.corrcoef(centered)
    saturated = ~np.isfinite(std) | (std > settings.max_std_uv) | (std < settings.flat_threshold_uv)
    keep: list[int] = []
    for index in range(len(channel_names)):
        if saturated[index]:
            continue
        duplicate = False
        for other in keep:
            if (
                np.isfinite(correlation[index, other])
                and correlation[index, other] > settings.duplicate_correlation
            ):
                duplicate = True
                break
        if not duplicate:
            keep.append(index)
    return keep, {"std": std, "saturated": saturated, "correlation": correlation}


def clean_epochs(
    data: np.ndarray, settings: AnalysisSettings, fs: float | None = None, valid_samples=None
) -> tuple[np.ndarray, np.ndarray]:
    """把长波形切成等长小片段，再按幅值、方差和已有BAD标记选择可用片段；布尔值True表示保留。"""
    samples_per_epoch = max(
        1, int(round(settings.epoch_seconds * (fs if fs is not None else settings.target_fs)))
    )
    epoch_count = data.shape[1] // samples_per_epoch
    if epoch_count <= 0:
        return np.empty((data.shape[0], 0, samples_per_epoch)), np.array([], dtype=bool)
    epochs = data[:, : epoch_count * samples_per_epoch].reshape(
        data.shape[0], epoch_count, samples_per_epoch
    )
    keep = np.ones(epoch_count, dtype=bool)
    if valid_samples is not None:
        keep &= (
            valid_samples[: epoch_count * samples_per_epoch]
            .reshape(epoch_count, samples_per_epoch)
            .all(axis=1)
        )
    amplitude = np.max(np.abs(epochs), axis=(0, 2))
    keep &= amplitude < settings.amplitude_threshold_uv
    variance = np.var(epochs, axis=2)
    eligible = variance[:, keep] if valid_samples is not None else variance
    z = (
        (variance - np.mean(eligible)) / (np.std(eligible) + 1e-12)
        if eligible.size
        else np.full_like(variance, np.inf)
    )
    keep &= np.all(np.abs(z) < settings.variance_z_threshold, axis=0)
    return epochs[:, keep, :], keep
