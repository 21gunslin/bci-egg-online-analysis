"""常用MNE操作集中放在这里，避免各模块各写一遍参数。

这里是“翻译层”：把网站约定的窗长、重叠和滤波参数交给MNE公开接口。
只使用MNE、NumPy和SciPy，不增加额外的脑电分析依赖。
原始BDF查看/导出流程只能调用PSD，不能调用这里的滤波函数。
"""

from __future__ import annotations

import numpy as np
from scipy import signal

from eeg_mne_runtime import load_mne


def bandpass_zero_phase(data, fs, low_hz, high_hz):
    """四阶Butterworth带通，正向和反向各滤一次，避免相位延迟。

    fs是“每秒采多少个点”。data可以是一条波形，也可以是多通道数值表。
    显式给出27点边缘填充，保持原有SciPy SOS滤波的边缘口径。
    """
    if not 0 < low_hz < high_hz < fs / 2:
        raise ValueError(f"采样率{fs:g}Hz无法实现{low_hz:g}–{high_hz:g}Hz带通。")
    mne = load_mne()
    return mne.filter.filter_data(
        np.asarray(data, dtype=np.float64),
        sfreq=fs,
        l_freq=low_hz,
        h_freq=high_hz,
        method="iir",
        phase="zero",
        copy=True,
        iir_params={"order": 4, "ftype": "butter", "output": "sos", "padlen": 27},
        verbose="ERROR",
    )


def notch_zero_phase(data, fs, frequency):
    """抑制一个工频尖峰；Q=30越大，抑制的频率范围越窄。

    SciPy只负责给出这条既定滤波器的系数，实际双向滤波由MNE执行。
    不改用MNE的默认陷波参数，以免同一份数据重构后悄悄换了处理方法。
    """
    mne = load_mne()
    numerator, denominator = signal.iirnotch(frequency, Q=30.0, fs=fs)
    return mne.filter.notch_filter(
        np.asarray(data, dtype=np.float64),
        Fs=fs,
        freqs=[frequency],
        method="iir",
        phase="zero",
        copy=True,
        iir_params={"b": numerator, "a": denominator, "padlen": 9},
        verbose="ERROR",
    )


def welch_psd(
    data,
    fs,
    window="hann",
    nperseg=256,
    noverlap=None,
    nfft=None,
    detrend="constant",
    scaling="density",
    axis=-1,
):
    """Welch功率谱：分成小窗、每窗算频谱，最后平均，让曲线更稳定。

    返回顺序是“频率，功率”，方便与图表横纵轴对应。
    原始质控用detrend=False保留直流；特征分析用constant逐窗减均值。
    Hann窗、重叠点数、FFT长度全部显式传入，不能依赖MNE的默认Hamming窗。
    """
    if scaling != "density" or detrend not in (False, "constant"):
        raise ValueError("本项目Welch仅使用density及False/constant去趋势。")
    values = np.moveaxis(np.asarray(data, dtype=np.float64), axis, -1)
    samples = min(int(nperseg), values.shape[-1])
    if samples < 2:
        raise ValueError("至少需要2个采样点才能计算PSD。")
    overlap = samples // 2 if noverlap is None else int(noverlap)
    fft_samples = samples if nfft is None else int(nfft)
    mne = load_mne()
    power, frequencies = mne.time_frequency.psd_array_welch(
        values,
        sfreq=fs,
        fmin=0.0,
        fmax=np.inf,
        n_fft=fft_samples,
        n_overlap=overlap,
        n_per_seg=samples,
        window=window,
        remove_dc=detrend is not False,
        average="mean",
        output="power",
        n_jobs=1,
        verbose="ERROR",
    )
    return frequencies, np.moveaxis(power, -1, axis)


def power_spectrum_windows(data, fs, window_samples, overlap_samples):
    """保留每个Hann窗口的PSD，不先平均，供时频波动统计。

    MNE的average=None返回“通道 × 频率 × 窗口”。
    每窗PSD等价于同参数STFT的模平方密度；窗中心就是横轴时间。
    不补零，不跨文件或BAD间隙拼接窗口。
    """
    mne = load_mne()
    power, frequencies = mne.time_frequency.psd_array_welch(
        np.asarray(data, dtype=np.float64),
        sfreq=fs,
        n_fft=window_samples,
        n_per_seg=window_samples,
        n_overlap=overlap_samples,
        window="hann",
        remove_dc=True,
        average=None,
        output="power",
        n_jobs=1,
        verbose="ERROR",
    )
    hop = window_samples - overlap_samples
    centers = (np.arange(power.shape[-1]) * hop + window_samples / 2) / fs
    return frequencies, centers, power
