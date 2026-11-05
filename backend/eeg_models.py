"""第1站：数据说明书和参数。只定义数据的样子，不读取文件、不计算特征。

HeaderInfo像文件的身份证；AnalysisSettings像一张分析参数表。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

SUPPORTED_EXTENSIONS = (".bdf", ".edf", ".gdf", ".fif")
PIPELINE_VERSION = "0.11.0"
PathLike = str | Path
ProgressCallback = Callable[[float, str], None]

DEFAULT_BANDS = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 45.0),
}

ROI_ALIASES = {
    "frontal": ("Fp1", "Fp2", "F3", "F4", "F7", "F8", "Fz", "AF3", "AF4"),
    "central": ("C3", "C4", "Cz", "C1", "C2"),
    "parietal": ("P3", "P4", "Pz", "P1", "P2"),
    "temporal": ("T3", "T4", "T5", "T6", "T7", "T8", "P7", "P8", "FT9", "FT10"),
    "occipital": ("O1", "O2", "Oz"),
}

NON_EEG_LABELS = {
    "STATUS",
    "MARKER",
    "MARKERS",
    "TRIGGER",
    "TRIGGERS",
    "EOG",
    "EMG",
    "EKG",
    "ECG",
    "RESP",
    "RESPIRATION",
    "SPO2",
    "PULSE",
    "TEMPERATURE",
    "TEMP",
    "ACCELEROMETER",
    "GYROSCOPE",
    "ADC",
    "DIGITAL",
    "ACTIVITY",
    "SNORING",
}

FEATURE_DESCRIPTIONS = {
    "时域": "信号幅值分布、波动、过零率、线长度及 Hjorth 活动度/移动度/复杂度。",
    "频域": "Welch 功率谱、频段绝对/相对功率、谱熵、峰值频率和频段比值。",
    "时频域": "短时傅里叶变换下各频段功率的时间波动、动态范围和高频事件占比。",
    "脑网络": "基于频段带通 PLV 相位锁值矩阵的加权/二值网络拓扑指标。",
    "质量控制": "有效通道、干净片段、幅值、方差、重复工频及自动伪迹剔除情况。",
    "汇总": "全脑与解剖 ROI 汇总、左右半球不对称和跨频段比值。",
}


@dataclass
class HeaderInfo:
    """文件头信息表：记录采样率、时长、通道和注释，避免在函数间传一串没有名字的数字。"""

    file_name: str
    file_path: str
    format: str
    version: str
    patient: str
    recording: str
    start_date: str
    start_time: str
    channel_count: int
    signal_count: int
    data_records: int
    record_duration: float
    duration_seconds: float
    physical_minimum: float
    physical_maximum: float
    digital_minimum: int
    digital_maximum: int
    annotations: int
    reader: str
    notes: list[str]
    analysis_channel_count: int = 0
    channel_labels: list[str] | None = None
    channel_units: list[str] | None = None
    sampling_rate: float = 0.0
    bad_channels: list[str] | None = None
    marked_segments: list[dict] | None = None
    annotation_records: list[dict] | None = None
    start_offset_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AnalysisSettings:
    """分析参数表：例如200Hz表示每秒200点，2秒片段就是400点；阈值不是临床诊断标准。"""

    target_fs: float = 250.0
    lowpass_hz: float = 45.0
    notches: tuple[float, ...] = (50.0, 100.0)
    epoch_seconds: float = 2.0
    amplitude_threshold_uv: float = 150.0
    variance_z_threshold: float = 4.0
    flat_threshold_uv: float = 0.8
    max_std_uv: float = 300.0
    duplicate_correlation: float = 0.998
    min_channels: int = 3
    min_keep_ratio: float = 0.10
    density: float = 0.25
    network_bands: tuple[str, ...] = ("delta", "theta", "alpha", "beta", "gamma")
    max_network_samples: int = 60_000
    max_channels: int = 128
    feature_profile: str = "standard"

    def __post_init__(self):
        if self.feature_profile not in ("standard", "thesis_v1"):
            raise ValueError("未知特征方案。")
        if self.feature_profile == "thesis_v1" and self.epoch_seconds != 2:
            raise ValueError("论文方案要求2秒质控片段。")

    @property
    def bands(self) -> dict[str, tuple[float, float]]:
        if self.feature_profile == "thesis_v1":
            from eeg_thesis_features import BANDS

            return dict(BANDS)
        upper = min(self.lowpass_hz, 45.0)
        return {
            name: (lo, min(hi, upper)) for name, (lo, hi) in DEFAULT_BANDS.items() if lo < upper
        }


def default_progress(_: float, __: str) -> None:
    return None


class EEGQualityError(ValueError):
    """质控未通过时的专用错误，便于网页展示原因而不是继续算出误导结果。"""


class NoUsableEpochsError(EEGQualityError):
    """没有可用时间片段；多份入口可据此保留原始波形供复核并跳过特征汇总。"""
