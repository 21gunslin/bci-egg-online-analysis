# -*- coding: utf-8 -*-
"""脑电特征提取引擎。

支持 BDF/EDF/GDF 读取，并从时域、频域、时频域和脑网络四个角度自动提取特征。
主要面向单文件、小规模研究数据分析；所有计算仅依赖 NumPy/SciPy/Matplotlib/Pandas。
"""

from __future__ import annotations

import json
import math
import re
import platform
from importlib.metadata import version as package_version
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np
import pandas as pd
from scipy import signal
from scipy.stats import kurtosis, skew


SUPPORTED_EXTENSIONS = (".bdf", ".edf", ".gdf")
PIPELINE_VERSION = "0.5.0"
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
    "STATUS", "MARKER", "MARKERS", "TRIGGER", "TRIGGERS", "EOG", "EMG", "EKG",
    "ECG", "RESP", "RESPIRATION", "SPO2", "PULSE", "TEMPERATURE", "TEMP",
    "ACCELEROMETER", "GYROSCOPE", "ADC", "DIGITAL", "ACTIVITY", "SNORING",
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

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AnalysisSettings:
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

    @property
    def bands(self) -> dict[str, tuple[float, float]]:
        upper = min(self.lowpass_hz, 45.0)
        return {name: (lo, min(hi, upper)) for name, (lo, hi) in DEFAULT_BANDS.items() if lo < upper}


def default_progress(_: float, __: str) -> None:
    return None


def _clean_text(value: Any) -> str:
    text = "" if value is None else str(value)
    return text.replace("\x00", " ").strip()


def _parse_ascii(raw: bytes) -> str:
    return raw.decode("latin-1", errors="replace").replace("\x00", " ").strip()


def _parse_number(value: Any, default: float = 0.0) -> float:
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return default


def _as_float_array(data: Any) -> np.ndarray:
    arr = np.asarray(data, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.ndim != 2:
        raise ValueError("信号数据必须是二维数组（通道 × 采样点）。")
    return arr


def _normalize_channel_name(label: str) -> str:
    name = _clean_text(label).upper().replace("EEG", "").replace("REF", "")
    name = re.sub(r"[\s\-_]+", "", name)
    name = re.sub(r"\(.*?\)|\[.*?\]", "", name)
    return name.strip() or "UNKNOWN"


def _is_annotation_label(label: str) -> bool:
    normalized = _normalize_channel_name(label)
    return (
        normalized in {"EDFANNOTATIONS", "BDFANNOTATIONS", "GDFANNOTATIONS", "ANNOTIONS"}
        or normalized.endswith("ANNOTATIONS")
    )


def _is_eeg_like(label: str, physical_unit: str) -> bool:
    normalized = _normalize_channel_name(label)
    if not normalized or normalized in NON_EEG_LABELS:
        return False
    if normalized.startswith(("X", "EMG", "EOG", "ECG", "EKG", "ACC", "GYRO")):
        return False
    # Voltage units alone cannot establish that a channel is EEG (e.g. EHG).
    return bool(re.fullmatch(
        r"(?:FP[12Z]|AF[1-8Z]|F[1-9Z]|FC[1-6Z]|FT[7-9]|C[1-6Z]|"
        r"CP[1-6Z]|TP[7-9]|P[1-9Z]|PO[1-8Z]|O[12Z]|T[1-9])", normalized
    ))


def voltage_to_uv(data: np.ndarray, units: Sequence[str]) -> np.ndarray:
    """Convert declared voltage units to µV; never infer a device correction."""
    factors = {"v": 1e6, "mv": 1e3, "uv": 1.0, "µv": 1.0, "μv": 1.0,
               "nv": 1e-3, "microvolt": 1.0, "microvolts": 1.0}
    values = _as_float_array(data).copy()
    if len(units) != len(values):
        raise ValueError("通道数量与电压单位数量不一致。")
    for index, unit in enumerate(units):
        key = _clean_text(unit).lower()
        if key not in factors:
            raise ValueError(f"未知电压单位 {unit!r}；请确认文件单位，不能按 µV 猜测。")
        values[index] *= factors[key]
    return values


def _choose_channels(
    labels: Sequence[str],
    units: Sequence[str],
    auto_eeg_only: bool,
    selected: Iterable[str] | None = None,
) -> tuple[list[int], list[str]]:
    selected_set = {_normalize_channel_name(item) for item in (selected or []) if _clean_text(item)}
    if selected_set:
        indices = [
            index for index, label in enumerate(labels)
            if _normalize_channel_name(label) in selected_set and not _is_annotation_label(label)
        ]
        return indices, [labels[index] for index in indices]

    if auto_eeg_only:
        indices = [
            index for index, label in enumerate(labels)
            if not _is_annotation_label(label) and _is_eeg_like(label, units[index])
        ]
        return indices, [labels[index] for index in indices]

    indices = [index for index, label in enumerate(labels) if not _is_annotation_label(label)]
    return indices, [labels[index] for index in indices]


def _record_duration_to_samples(record_duration: float, nominal_fs: float) -> int:
    if record_duration > 0:
        return max(1, int(round(record_duration * nominal_fs)))
    return max(1, int(round(nominal_fs)))


class _EDFReader:
    """EDF/EDF+/BDF/BDF+ 读取器；GDF 若与 EDF 结构兼容时也可读取。"""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.is_bdf = self.path.suffix.lower() == ".bdf"
        self.sample_width = 3 if self.is_bdf else 2
        self.digitally_signed = True
        self.header_bytes = 0
        self.signal_count = 0
        self.annotation_indices: list[int] = []
        self.reader_name = "EDF/BDF结构读取器"

    @staticmethod
    def _fits_standard_edf(path: Path) -> bool:
        try:
            with path.open("rb") as handle:
                fixed = handle.read(256)
                if len(fixed) < 256:
                    return False
                version = _parse_ascii(fixed[0:8])
                is_bdf_signature = fixed[:1] == b"\xff"
                if not is_bdf_signature and not version.startswith(("0", "BDF", "B+B")):
                    return False
                try:
                    signal_count = int(_parse_ascii(fixed[252:256]) or 0)
                except ValueError:
                    return False
                if signal_count <= 0 or signal_count > 10_000:
                    return False
                signal_header = handle.read(256 * signal_count)
                if len(signal_header) != 256 * signal_count:
                    return False
                label_block = signal_header[:16 * signal_count]
                labels = [
                    _parse_ascii(label_block[index * 16:(index + 1) * 16])
                    for index in range(signal_count)
                ]
                if not any(label for label in labels):
                    return False
                return True
        except OSError:
            return False
        return True

    @classmethod
    def can_read(cls, path: Path) -> bool:
        return cls._fits_standard_edf(path)

    @staticmethod
    def _parse_int_field(raw: bytes, fallback: int) -> int:
        text = _parse_ascii(raw)
        match = re.search(r"[-+]?\d+", text)
        if not match:
            return fallback
        try:
            return int(match.group())
        except ValueError:
            return fallback

    @staticmethod
    def _parse_float_field(raw: bytes, fallback: float) -> float:
        text = _parse_ascii(raw)
        try:
            return float(text)
        except ValueError:
            return fallback

    def read(self, max_channels: int = 128) -> tuple[np.ndarray, list[str], list[str], HeaderInfo, list[str]]:
        with self.path.open("rb") as handle:
            fixed = handle.read(256)
            if len(fixed) < 256:
                raise ValueError("文件头不足 256 字节，无法按 EDF/BDF 结构读取。")

            reserved = _parse_ascii(fixed[192:236])
            if reserved.startswith(("EDF+D", "BDF+D")):
                raise ValueError("不连续 EDF/BDF 需要保留时间注释，请使用 MNE 读取；不能按连续信号拼接。")
            version_raw = fixed[0:8]
            version = _parse_ascii(version_raw)
            is_bdf = (
                self.is_bdf
                or version_raw[:1] == b"\xff"
                or version.upper().startswith("BDF")
                or "24BIT" in reserved.upper()
            )
            self.is_bdf = is_bdf
            self.sample_width = 3 if is_bdf else 2
            patient = _parse_ascii(fixed[8:88])
            recording = _parse_ascii(fixed[88:168])
            start_date = _parse_ascii(fixed[168:176])
            start_time = _parse_ascii(fixed[176:184])
            try:
                header_bytes = int(_parse_ascii(fixed[184:192]) or 0)
            except ValueError:
                header_bytes = 0
            header_bytes = self._parse_int_field(fixed[184:192], header_bytes)
            try:
                data_records = int(float(_parse_ascii(fixed[236:244]) or 0))
            except ValueError:
                data_records = 0
            data_records = self._parse_int_field(fixed[236:244], data_records)
            try:
                record_duration = float(_parse_ascii(fixed[244:252]) or 0)
            except ValueError:
                record_duration = 0.0
            record_duration = self._parse_float_field(fixed[244:252], record_duration)
            if not np.isfinite(record_duration) or record_duration <= 0:
                raise ValueError("EDF/BDF 记录时长无效，不能猜测采样率。")
            try:
                signal_count = int(_parse_ascii(fixed[252:256]) or 0)
            except ValueError as error:
                raise ValueError("EDF/BDF 头部的信号通道数无效。") from error
            signal_count = self._parse_int_field(fixed[252:256], signal_count)
            if signal_count <= 0:
                raise ValueError("EDF/BDF 头部没有有效信号通道。")

            signal_header = handle.read(256 * signal_count)
            if len(signal_header) != 256 * signal_count:
                raise ValueError("EDF/BDF 信号头被截断。")

            fields = [
                ("label", 16), ("transducer", 80), ("physical_dimension", 8),
                ("physical_minimum", 8), ("physical_maximum", 8),
                ("digital_minimum", 8), ("digital_maximum", 8),
                ("prefilter", 80), ("samples_per_record", 8), ("reserved", 32),
            ]
            parsed: dict[str, list[str]] = {}
            field_offset = 0
            for name, width in fields:
                block = signal_header[field_offset:field_offset + width * signal_count]
                parsed[name] = [
                    _parse_ascii(block[index * width:(index + 1) * width])
                    for index in range(signal_count)
                ]
                field_offset += width * signal_count

            labels = [_clean_text(value) for value in parsed["label"]]
            dimensions = [_clean_text(value) for value in parsed["physical_dimension"]]
            physical_min = np.array([_parse_number(value, math.nan) for value in parsed["physical_minimum"]], dtype=float)
            physical_max = np.array([_parse_number(value, math.nan) for value in parsed["physical_maximum"]], dtype=float)
            digital_min = np.array([_parse_number(value, math.nan) for value in parsed["digital_minimum"]], dtype=float)
            digital_max = np.array([_parse_number(value, math.nan) for value in parsed["digital_maximum"]], dtype=float)
            samples_per_record = np.array([int(_parse_number(value, 1)) for value in parsed["samples_per_record"]], dtype=int)
            if np.any(samples_per_record <= 0):
                raise ValueError("存在每记录采样点数无效的通道。")

            if header_bytes <= 0:
                header_bytes = 256 + signal_count * 256
            if header_bytes < 256 + signal_count * 256:
                raise ValueError("EDF/BDF 主头长度与信号头长度不一致。")

            handle.seek(header_bytes)
            body = handle.read()
            expected_record_bytes = int(samples_per_record.sum()) * self.sample_width
            if expected_record_bytes <= 0:
                raise ValueError("EDF/BDF 每记录字节数无效。")
            available_records = len(body) // expected_record_bytes
            if data_records > 0 and available_records > 0:
                available_records = min(available_records, data_records)
            if available_records <= 0:
                raise ValueError("未读取到有效数据记录，请检查文件是否完整。")

            record_count = available_records
            body = np.frombuffer(
                body[: record_count * expected_record_bytes], dtype=np.uint8
            ).reshape(record_count, -1)

            if self.sample_width == 3:
                raw = (
                    body[:, 0::3].astype(np.int32)
                    | (body[:, 1::3].astype(np.int32) << 8)
                    | (body[:, 2::3].astype(np.int32) << 16)
                )
                raw = np.where((raw & 0x80_0000) != 0, raw - 0x100_0000, raw)
            else:
                raw = body.view("<i2").reshape(record_count, -1)

            annotation_indices = [
                index for index, label in enumerate(labels) if _is_annotation_label(label)
            ]
            annotations = self._decode_annotations(
                raw, annotation_indices, samples_per_record
            )

            eeg_indices = [index for index in range(signal_count) if index not in annotation_indices]
            if not eeg_indices:
                raise ValueError("EDF/BDF 文件没有可用于脑电分析的数据通道。")
            if len(eeg_indices) > max_channels:
                eeg_indices = eeg_indices[:max_channels]

            offsets = np.concatenate(([0], np.cumsum(samples_per_record)))
            mode_samples = int(np.bincount(samples_per_record[eeg_indices]).argmax())
            notes: list[str] = []
            channel_data: list[np.ndarray] = []
            eeg_labels: list[str] = []
            eeg_units: list[str] = []
            gains: list[float] = []
            for index in eeg_indices:
                start = offsets[index]
                end = offsets[index + 1]
                values = raw[:, start:end].reshape(-1)
                scale_range = digital_max[index] - digital_min[index]
                if abs(scale_range) < 1e-15:
                    gain = 1.0
                    offset = 0.0
                else:
                    gain = (physical_max[index] - physical_min[index]) / scale_range
                    offset = physical_min[index] - digital_min[index] * gain
                if samples_per_record[index] != mode_samples:
                    notes.append(
                        f"通道 {labels[index] or f'CH{index + 1}'} 每记录采样点数 "
                        f"{samples_per_record[index]} 与主要通道 {mode_samples} 不一致；"
                        "为保持时间轴对齐，已跳过该通道。"
                    )
                    continue
                channel_data.append(values.astype(np.float64) * gain + offset)
                eeg_labels.append(labels[index] or f"CH{index + 1}")
                eeg_units.append(dimensions[index])
                gains.append(float(gain))

            duration_records = record_duration if record_duration > 0 else 1.0
            duration_seconds = record_count * duration_records
            signal_count_eeg = len(eeg_indices)
            if reserved:
                notes.append(f"保留字段: {reserved}")
            if data_records > 0 and record_count < data_records:
                notes.append(f"仅读取完整记录 {record_count}/{data_records}，文件可能被截断。")
            data = np.vstack(channel_data)
            self.signal_count = signal_count
            self.annotation_indices = annotation_indices
            header = HeaderInfo(
                file_name=self.path.name,
                file_path=str(self.path),
                format="BDF" if self.is_bdf else "EDF",
                version=version,
                patient=patient,
                recording=recording,
                start_date=start_date,
                start_time=start_time,
                channel_count=signal_count,
                signal_count=signal_count_eeg,
                data_records=record_count,
                record_duration=duration_records,
                duration_seconds=float(duration_seconds),
                physical_minimum=float(np.nanmin(physical_min[eeg_indices])) if len(eeg_indices) else math.nan,
                physical_maximum=float(np.nanmax(physical_max[eeg_indices])) if len(eeg_indices) else math.nan,
                digital_minimum=int(np.nanmin(digital_min[eeg_indices])) if len(eeg_indices) else 0,
                digital_maximum=int(np.nanmax(digital_max[eeg_indices])) if len(eeg_indices) else 0,
                annotations=len(annotations),
                reader=self.reader_name,
                notes=notes,
                sampling_rate=(mode_samples / duration_records) if duration_records > 0 else 0.0,
            )
            return data, eeg_labels, eeg_units, header, annotations

    @staticmethod
    def _decode_annotations(raw: np.ndarray, indices: list[int], samples: np.ndarray) -> list[str]:
        if not indices:
            return []
        events: list[str] = []
        offsets = np.concatenate(([0], np.cumsum(samples)))
        for index in indices:
            start, end = offsets[index], offsets[index + 1]
            for record in raw:
                payload = record[start:end]
                if payload.size == 0:
                    continue
                byte_values = payload.astype(np.uint8) if payload.dtype != np.uint8 else payload
                if byte_values.size == 0:
                    continue
                text = bytes(byte_values).decode("latin-1", errors="ignore").replace("\x00", " ")
                for match in re.finditer(r"(\d+[.,]\d+)\x14([^\x14\x00]{0,80})", text):
                    events.append(f"{match.group(1)}: {match.group(2).strip()}")
        return events


class _GDFReader:
    """Delegate GDF binary headers, units and event timing to MNE's public reader."""

    def __init__(self, path: Path):
        self.path = Path(path)

    @classmethod
    def can_read(cls, path: Path) -> bool:
        with Path(path).open("rb") as handle:
            return handle.read(8).startswith(b"GDF")

    def read(self, max_channels: int = 128):
        try:
            import mne
        except ImportError as error:
            raise ValueError("GDF 读取需要可选 MNE 依赖：python -m pip install -r requirements-mne.txt") from error
        raw = mne.io.read_raw_gdf(self.path, preload=False, verbose="ERROR")
        try:
            indices = [i for i, kind in enumerate(raw.get_channel_types()) if kind == "eeg"]
            if not indices:
                raise ValueError("MNE 没有识别出 GDF EEG 通道，请核对通道类型。")
            if len(indices) > max_channels:
                raise ValueError("GDF EEG 通道数超过配置上限，不能静默截断。")
            # MNE returns EEG voltages in SI volts, regardless of on-disk unit.
            data = raw.get_data(picks=indices) * 1e6
            labels = [raw.ch_names[i] for i in indices]
            units = ["µV"] * len(labels)
            fs = float(raw.info["sfreq"])
            annotations = [f"{onset:g}s / {duration:g}s: {description}" for onset, duration, description
                           in zip(raw.annotations.onset, raw.annotations.duration, raw.annotations.description)]
            date = raw.info.get("meas_date")
            header = HeaderInfo(
                file_name=self.path.name, file_path=str(self.path), format="GDF",
                version="MNE reader", patient="", recording="",
                start_date=date.strftime("%d.%m.%y") if date else "",
                start_time=date.strftime("%H.%M.%S") if date else "",
                channel_count=len(raw.ch_names), signal_count=len(labels), data_records=0,
                record_duration=0.0, duration_seconds=raw.n_times/fs,
                physical_minimum=float("nan"), physical_maximum=float("nan"),
                digital_minimum=0, digital_maximum=0, annotations=len(annotations),
                reader=f"MNE {mne.__version__} read_raw_gdf",
                notes=["GDF EEG 单位由 MNE 转为 V 后统一换算为 µV。", "事件保留为带时间文本；未自动建立 ERP epochs。"],
                sampling_rate=fs,
            )
            return data, labels, units, header, annotations
        finally:
            raw.close()


def read_eeg_file(
    path: PathLike,
    selected_channels: Iterable[str] | None = None,
    auto_eeg_only: bool = True,
    max_channels: int = 128,
) -> tuple[np.ndarray, list[str], list[str], HeaderInfo, list[str]]:
    """读取 BDF/EDF/GDF 文件，返回数据、通道名、单位、头信息和注释。"""
    file_path = Path(path).expanduser().resolve()
    if not file_path.is_file():
        raise FileNotFoundError(f"找不到输入文件：{file_path}")
    if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"暂不支持 {file_path.suffix or '未知'} 格式；支持 .bdf/.edf/.gdf。")

    reader: _EDFReader | _GDFReader
    if _EDFReader.can_read(file_path):
        reader = _EDFReader(file_path)
    elif _GDFReader.can_read(file_path):
        reader = _GDFReader(file_path)
    else:
        raise ValueError(
            "无法识别文件结构。GDF 需为标准 GDF 2.x；若为其他 GDF 扩展，"
            "请先转换为 EDF/BDF 后再导入。"
        )

    data, labels, units, header, annotations = reader.read(max_channels=max_channels)
    indices, chosen_labels = _choose_channels(labels, units, auto_eeg_only, selected_channels)
    if not indices:
        raise ValueError("没有找到可分析的 EEG 通道。")
    if len(indices) > max_channels:
        indices = indices[:max_channels]
        chosen_labels = chosen_labels[:max_channels]
    data = data[indices]
    units = [units[index] for index in indices]
    header.analysis_channel_count = len(chosen_labels)
    header.channel_labels = chosen_labels
    header.channel_units = units
    if not np.isfinite(data).all():
        raise ValueError("记录包含非有限采样值，请先标记坏段；不能静默置零后继续分析。")
    return data, chosen_labels, units, header, annotations


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
        return float(header.sampling_rate), [f"采样率由文件头各通道采样点数推算：{header.sampling_rate:g} Hz。"]

    if header.data_records > 0 and header.duration_seconds > 0:
        samples = data.shape[1]
        fs = samples / header.duration_seconds
        return float(fs), ["采样率由总时长与采样点数推算。"]

    raise ValueError("无法确定采样率；请在分析设置中手动指定。")


def _downsample(data: np.ndarray, source_fs: float, target_fs: float) -> tuple[np.ndarray, float, list[str]]:
    if source_fs <= 0:
        raise ValueError("源采样率无效。")
    target = min(target_fs, source_fs)
    if np.isclose(source_fs, target, rtol=1e-6, atol=1e-6):
        return data.astype(np.float64, copy=False), source_fs, []

    integer_ratio = source_fs / target
    rounded_ratio = int(round(integer_ratio))
    if rounded_ratio >= 2 and np.isclose(integer_ratio, rounded_ratio, rtol=1e-6, atol=1e-6):
        decimated = signal.decimate(data, rounded_ratio, axis=1, ftype="fir", zero_phase=True)
        return decimated, source_fs / rounded_ratio, [f"按 {rounded_ratio}:1 抗混叠抽取至 {source_fs / rounded_ratio:g} Hz。"]

    number_samples = int(round(data.shape[1] * target / source_fs))
    if number_samples < 8:
        raise ValueError("数据过短，无法重采样。")
    resampled = signal.resample_poly(
        data, up=int(round(target * 1000)), down=int(round(source_fs * 1000)), axis=1
    )[:, :number_samples]
    return resampled, target, [f"使用 resample_poly 重采样至约 {target:g} Hz。"]


def _bandpass(data: np.ndarray, fs: float, low_hz: float, high_hz: float) -> np.ndarray:
    nyquist = fs / 2.0
    if low_hz >= high_hz or high_hz >= nyquist:
        raise ValueError(
            f"采样率 {fs:g} Hz 无法实现 {low_hz:g}–{high_hz:g} Hz 带通；"
            "分析采样率必须高于最高截止频率的 2 倍。"
        )
    # SOS is numerically better conditioned than a high-order b/a polynomial.
    # N=4 is the Butterworth prototype order. Band-pass transformation gives
    # single-pass order 8; forward-backward filtering gives effective order 16.
    # The composite amplitude response is |H|² (cutoffs are about -6 dB).
    sos = signal.butter(4, [low_hz, high_hz], btype="bandpass", fs=fs, output="sos")
    return signal.sosfiltfilt(sos, data, axis=1)


def _apply_notches(data: np.ndarray, fs: float, notches: Sequence[float]) -> tuple[np.ndarray, list[str]]:
    applied: list[str] = []
    result = data
    nyquist = fs / 2.0
    for frequency in notches:
        if frequency <= 0 or frequency >= nyquist:
            continue
        b, a = signal.iirnotch(frequency, Q=30.0, fs=fs)
        result = signal.filtfilt(b, a, result, axis=1)
        applied.append(f"{frequency:g} Hz")
    if applied:
        return result, [f"已施加工频/谐波陷波：{', '.join(applied)}。"]
    return result, []


def preprocess_eeg(
    data: np.ndarray,
    header: HeaderInfo,
    settings: AnalysisSettings,
    explicit_fs: float | None = None,
    selected_channels: Iterable[str] | None = None,
    progress: ProgressCallback = default_progress,
) -> dict[str, Any]:
    """读取后的质量控制与预处理，返回字典而不是多套位置参数。"""
    progress(0.02, "读取文件并识别通道")
    source_fs, fs_notes = _estimate_sampling_rate(data, header, explicit_fs)
    labels, units = list(header.channel_labels or []), list(header.channel_units or [])
    if not labels or not units or len(labels) != data.shape[0]:
        raise ValueError("文件头中的分析通道标签不完整。")
    if selected_channels:
        wanted = {_normalize_channel_name(item) for item in selected_channels}
        indices = [i for i, label in enumerate(labels) if _normalize_channel_name(label) in wanted]
        if indices:
            data, labels, units = data[indices], [labels[i] for i in indices], [units[i] for i in indices]
        else:
            raise ValueError("指定通道不存在。")

    data = voltage_to_uv(data, units)
    source_units = units.copy()
    units = ["µV"] * len(labels)

    if data.shape[1] < int(source_fs):
        raise ValueError(f"有效数据不足 1 秒（{data.shape[1]} 采样点 @ {source_fs:g} Hz）。")

    notes = list(fs_notes) + [
        "电压按文件头声明单位统一为 µV；未推断设备量纲误标或施加额外倍率。",
        "保留采集参考；文件头未验证参考电极，未自动重参考或推断电极坐标。",
    ]
    progress(0.10, "重采样与带通滤波")
    data, fs, resample_notes = _downsample(data, source_fs, settings.target_fs)
    notes.extend(resample_notes)
    filter_high_hz = min(settings.lowpass_hz, 45.0)
    data = _bandpass(data, fs, 0.5, filter_high_hz)
    notes.append(
        f"已施加四阶 Butterworth 带通滤波（0.5–{filter_high_hz:g} Hz），"
        "采用 SOS 双向零相位滤波；N=4 原型的带通单程为8阶，双向有效16阶；合成幅频响应为 |H|²。"
    )
    data, notch_notes = _apply_notches(data, fs, settings.notches)
    notes.extend(notch_notes)

    progress(0.18, "通道质量控制")
    keep, channel_metrics = channel_qc(data, labels, settings)
    if len(keep) < settings.min_channels:
        raise ValueError(
            f"有效通道仅 {len(keep)} 个，低于最小要求 {settings.min_channels}；"
            "请检查电极脱落、单位或预处理设置。"
        )
    data = data[keep]
    kept_labels = [labels[index] for index in keep]
    kept_units = [units[index] for index in keep]

    progress(0.28, "分段与伪迹剔除")
    clean_epochs_data, epochs = clean_epochs(data, settings, fs=fs)
    keep_ratio = float(np.mean(epochs)) if epochs.size else 0.0
    if clean_epochs_data.shape[2] == 0 or keep_ratio < settings.min_keep_ratio:
        guidance = (
            "有效数据不足 2 秒，请缩短片段长度或检查记录时长。"
            if epochs.size == 0
            else "请放宽伪迹阈值或检查原始数据。"
        )
        raise ValueError(
            f"自动质控后可用片段比例仅 {keep_ratio:.1%}，低于最小要求 "
            f"{settings.min_keep_ratio:.1%}；{guidance}"
        )
    # Collapse retained clean epochs back to channel × time for the feature
    # routines. The earlier 3-D array leaked an epoch dimension into Welch and
    # other feature calculations, causing incorrect shapes or broadcasting.
    clean = clean_epochs_data.reshape(clean_epochs_data.shape[0], -1)
    kept_epoch_indices = np.flatnonzero(epochs)
    original_epoch_samples = int(round(settings.epoch_seconds * fs))
    notes.append("保留片段拼接用于探索性特征；跨剔除间隙的谱/时频/相位及导数可能受边界影响，不能视为连续原始时间。")

    channel_records = []
    for original_index in keep:
        channel_records.append({
            "原始序号": original_index + 1,
            "通道": labels[original_index],
            "单位": units[original_index],
            "状态": "保留",
            "标准差": float(channel_metrics["std"][original_index]),
            "是否饱和或异常": bool(channel_metrics["saturated"][original_index]),
        })
    for original_index, label in enumerate(labels):
        if original_index not in keep:
            channel_records.append({
                "原始序号": original_index + 1,
                "通道": label,
                "单位": units[original_index],
                "状态": "剔除",
                "标准差": float(channel_metrics["std"][original_index]),
                "是否饱和或异常": bool(channel_metrics["saturated"][original_index]),
            })

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
    """剔除平坦、饱和以及高度重复的通道。"""
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
            if np.isfinite(correlation[index, other]) and correlation[index, other] > settings.duplicate_correlation:
                duplicate = True
                break
        if not duplicate:
            keep.append(index)
    return keep, {"std": std, "saturated": saturated, "correlation": correlation}


def clean_epochs(data: np.ndarray, settings: AnalysisSettings, fs: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    """2 秒分段，按幅值与方差异常剔除伪迹，返回保留掩码。"""
    samples_per_epoch = max(1, int(round(settings.epoch_seconds * (fs if fs is not None else settings.target_fs))))
    epoch_count = data.shape[1] // samples_per_epoch
    if epoch_count <= 0:
        return np.empty((data.shape[0], 0, samples_per_epoch)), np.array([], dtype=bool)
    epochs = data[:, : epoch_count * samples_per_epoch].reshape(data.shape[0], epoch_count, samples_per_epoch)
    keep = np.ones(epoch_count, dtype=bool)
    amplitude = np.max(np.abs(epochs), axis=(0, 2))
    keep &= amplitude < settings.amplitude_threshold_uv
    variance = np.var(epochs, axis=2)
    z = (variance - np.mean(variance)) / (np.std(variance) + 1e-12)
    keep &= np.all(np.abs(z) < settings.variance_z_threshold, axis=0)
    return epochs[:, keep, :], keep


def _windowed_features(data: np.ndarray) -> dict[str, np.ndarray]:
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
        "rms": np.sqrt(np.mean(data ** 2, axis=1)),
        "peak_to_peak": np.max(data, axis=1) - np.min(data, axis=1),
        "hjorth_activity": variance,
        "hjorth_mobility": mobility,
        "hjorth_complexity": complexity,
    }


def _frequency_features(data: np.ndarray, fs: float, bands: dict[str, tuple[float, float]]) -> dict[str, np.ndarray]:
    nperseg = int(min(1024, 2 ** math.floor(math.log2(max(16, data.shape[1])))))
    frequencies, psd = signal.welch(data, fs=fs, nperseg=nperseg, axis=1)
    df = float(frequencies[1] - frequencies[0])
    analysis_mask = (frequencies >= 0.5) & (frequencies <= min(45.0, fs / 2))
    total = np.sum(psd[:, analysis_mask], axis=1) * df + 1e-20
    psd_mass = np.sum(psd, axis=1) + 1e-20
    features: dict[str, np.ndarray] = {
        "total_power": total,
        "mean_frequency": np.sum(frequencies[None, :] * psd, axis=1) / psd_mass,
        "spectral_entropy": -np.sum((psd / psd_mass[:, None]) * np.log(psd / psd_mass[:, None] + 1e-20), axis=1)
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
    window_size = int(min(512, 2 ** math.floor(math.log2(max(32, data.shape[1] // 8)))))
    overlap = int(window_size * 0.75)
    if window_size >= data.shape[1]:
        window_size = int(2 ** math.floor(math.log2(max(32, data.shape[1] // 2))))
        overlap = window_size // 2
    if window_size <= 4:
        return {}

    frequencies, times, transform = signal.stft(
        data, fs=fs, window="hann", nperseg=window_size, noverlap=overlap, axis=1, boundary=None, padded=False
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
    n = weights.shape[0]
    if n < 2:
        return {
            "weighted_density": 0.0, "density": 0.0, "mean_strength": 0.0,
            "strength_std": 0.0, "mean_degree": 0.0, "degree_std": 0.0,
            "clustering": 0.0, "global_efficiency": 0.0, "characteristic_path_length": 0.0,
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
        progress(base_progress + span * (index + 1) / max(1, len(usable_bands)), f"计算 {name} 频段脑网络")
    return features


def aggregate_features(
    channel_features: dict[str, np.ndarray],
    channel_names: Sequence[str],
    network_features: dict[str, float] | None = None,
    global_features: dict[str, float] | None = None,
) -> dict[str, float]:
    """通道级特征 -> 全脑/ROI/不对称汇总；避免把数组误写入 CSV 单元格。"""
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
        ("Fp1", "Fp2"), ("F3", "F4"), ("F7", "F8"),
        ("C3", "C4"), ("T3", "T4"), ("T5", "T6"), ("O1", "O2"),
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
    """提取四类特征，返回（通道级表，汇总表，附加元数据）。"""
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
    channel_features.update({f"时频域__{key}": value for key, value in time_frequency_features.items()})

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
        "平均峰值Alpha频率": float(np.mean(channel_table.get("频域__peak_alpha_frequency", pd.Series([np.nan])))),
        "平均Hjorth复杂度": float(np.mean(channel_table["时域__hjorth_complexity"])),
        "平均ThetaBeta比": float(np.mean(channel_table.get("频域__theta_beta_ratio", pd.Series([np.nan])))),
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


def _safe_slug(value: str) -> str:
    cleaned = re.sub(r"[^\w\u4e00-\u9fff-]+", "_", _clean_text(value)).strip("_")
    return cleaned or "eeg"


def save_figures(
    preprocessed: dict[str, Any],
    channel_table: pd.DataFrame,
    metadata: dict[str, Any],
    output_dir: PathLike,
    slug: str,
    progress: ProgressCallback = default_progress,
) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import font_manager, rcParams
    import matplotlib.pyplot as plt

    for font_family in ("Microsoft YaHei UI", "Microsoft YaHei", "SimHei", "Noto Sans CJK SC"):
        try:
            font_manager.findfont(font_family, fallback_to_default=False)
        except ValueError:
            continue
        rcParams["font.sans-serif"] = [font_family]
        break
    rcParams["axes.unicode_minus"] = False

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    figure_paths: list[Path] = []

    data = np.asarray(preprocessed["data"])
    fs = float(preprocessed["fs"])
    labels = list(preprocessed["channel_names"])
    bands = metadata["bands"]

    progress(0.96, "绘制波形与通道质控图")
    fig, axes = plt.subplots(2, 1, figsize=(13, 9), constrained_layout=True)
    duration = min(10.0, data.shape[1] / fs)
    samples = int(duration * fs)
    time_axis = np.arange(samples) / fs
    for index, label in enumerate(labels):
        offset = index * (np.std(data[index, :samples]) * 6 + 1e-9)
        axes[0].plot(time_axis, data[index, :samples] - offset, linewidth=0.7)
    axes[0].set_title("预处理后波形（前 10 秒，垂直偏移显示）")
    axes[0].set_xlabel("有效片段拼接时间 (s)，非原始连续时间")
    axes[0].set_ylabel("通道")
    axes[0].grid(alpha=0.25)

    sorted_table = channel_table.sort_values("时域__std")
    axes[1].barh(sorted_table["通道"], sorted_table["时域__std"], color="#4f8cc9")
    axes[1].set_title("通道标准差（自动质控保留通道）")
    axes[1].set_xlabel("标准差")
    axes[1].grid(axis="x", alpha=0.25)
    waveform_path = output / f"{slug}_波形与通道质控.png"
    fig.savefig(waveform_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    figure_paths.append(waveform_path)

    progress(0.97, "绘制功率谱与频段特征")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    nperseg = int(min(1024, 2 ** math.floor(math.log2(max(16, data.shape[1])))))
    frequencies, psd = signal.welch(data, fs=fs, nperseg=nperseg, axis=1)
    mean_psd = np.mean(psd, axis=0)
    axes[0].semilogy(frequencies, mean_psd, color="#3b76c1", linewidth=1.5)
    for low, high in bands.values():
        axes[0].axvspan(low, high, alpha=0.10, color="#e97397")
    axes[0].set_xlim(0, min(fs / 2, 60))
    axes[0].set_title("平均功率谱密度")
    axes[0].set_xlabel("频率 (Hz)")
    axes[0].set_ylabel("PSD (µV²/Hz)")
    axes[0].grid(alpha=0.25)

    band_names = list(bands)
    relative_columns = [f"频域__{name}_relative_power" for name in band_names]
    if all(column in channel_table for column in relative_columns):
        relative = channel_table[relative_columns].to_numpy(dtype=float)
        mean_relative = np.mean(relative, axis=0) * 100
        std_relative = np.std(relative, axis=0) * 100
        axes[1].bar(band_names, mean_relative, yerr=std_relative, color="#77b26c", capsize=4)
        axes[1].set_title("各频段相对功率（全通道均值 ± 标准差）")
        axes[1].set_ylabel("相对功率 (%)")
        axes[1].grid(axis="y", alpha=0.25)
    else:
        axes[1].text(0.5, 0.5, "频段特征不可用", ha="center", va="center")
        axes[1].axis("off")
    spectrum_path = output / f"{slug}_频域特征.png"
    fig.savefig(spectrum_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    figure_paths.append(spectrum_path)

    progress(0.98, "绘制时频图与脑网络图")
    channel_index = int(np.argmax(np.var(data, axis=1)))
    stft_size = int(min(512, 2 ** math.floor(math.log2(max(32, data.shape[1] // 8)))))
    stft_overlap = int(stft_size * 0.75)
    if stft_size >= data.shape[1]:
        stft_size = int(2 ** math.floor(math.log2(max(32, data.shape[1] // 2))))
        stft_overlap = stft_size // 2
    if stft_size > 4:
        stft_frequencies, stft_times, transform = signal.stft(
            data[channel_index], fs=fs, nperseg=stft_size, noverlap=stft_overlap,
            boundary=None, padded=False
        )
        power = np.abs(transform) ** 2
        fig, ax = plt.subplots(figsize=(11, 5), constrained_layout=True)
        mesh = ax.pcolormesh(
            stft_times, stft_frequencies, 10 * np.log10(power + 1e-20),
            shading="auto", cmap="magma"
        )
        ax.set_ylim(0, min(45, fs / 2))
        ax.set_title(f"代表性通道 {labels[channel_index]} 的 STFT 时频图")
        ax.set_xlabel("有效片段拼接时间 (s)，跨剔除边界需谨慎解释")
        ax.set_ylabel("频率 (Hz)")
        fig.colorbar(mesh, ax=ax, label="功率 (dB)")
        tf_path = output / f"{slug}_时频图.png"
        fig.savefig(tf_path, dpi=160, bbox_inches="tight")
        plt.close(fig)
        figure_paths.append(tf_path)

    network_band = next((name for name in settings_network_bands(metadata) if name in bands), None)
    if network_band:
        low, high = bands[network_band]
        matrix = plv_matrix(data, fs, low, high, 60_000)
        fig, ax = plt.subplots(figsize=(6.5, 5.5), constrained_layout=True)
        image = ax.imshow(matrix, cmap="viridis", vmin=0, vmax=1)
        ax.set_xticks(range(len(labels)), labels, rotation=90, fontsize=8)
        ax.set_yticks(range(len(labels)), labels, fontsize=8)
        ax.set_title(f"{network_band} 频段 PLV 脑网络矩阵")
        fig.colorbar(image, ax=ax, label="PLV")
        network_path = output / f"{slug}_脑网络_{network_band}.png"
        fig.savefig(network_path, dpi=160, bbox_inches="tight")
        plt.close(fig)
        figure_paths.append(network_path)

    return figure_paths


def settings_network_bands(metadata: dict[str, Any]) -> list[str]:
    """从 metadata 中取得网络频段，避免绘图函数额外依赖全局状态。"""
    network_features = metadata.get("network_features", {})
    names: list[str] = []
    for key in network_features:
        match = re.match(r"(.+?)_plv_", key)
        if match and match.group(1) not in names:
            names.append(match.group(1))
    return names


def build_feature_workbook(
    channel_table: pd.DataFrame,
    summary_table: pd.DataFrame,
    header: HeaderInfo,
    preprocessed: dict[str, Any],
    metadata: dict[str, Any],
    output_path: PathLike,
    settings: AnalysisSettings,
    annotations: Sequence[str] | None = None,
) -> Path:
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
    from openpyxl.utils import get_column_letter

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    workbook = Workbook()
    header_fill = PatternFill("solid", fgColor="3B76C1")
    section_fill = PatternFill("solid", fgColor="DDEBF7")
    title_font = Font(name="Microsoft YaHei", size=14, bold=True, color="FFFFFF")
    normal_font = Font(name="Microsoft YaHei", size=10)
    thin = Side(style="thin", color="D9E2F3")
    border = Border(left=thin, right=thin, top=thin, bottom=thin)

    info = workbook.active
    info.title = "文件信息"
    info.append(["项目", "值"])
    info_values = {
        "文件名": header.file_name,
        "文件路径": header.file_path,
        "格式": header.format,
        "版本": header.version,
        "患者字段": header.patient,
        "记录字段": header.recording,
        "开始日期": header.start_date,
        "开始时间": header.start_time,
        "原始通道数": header.channel_count,
        "分析通道数": header.analysis_channel_count or len(preprocessed["channel_names"]),
        "数据记录数": header.data_records,
        "记录时长_s": header.record_duration,
        "原始时长_s": header.duration_seconds,
        "分析采样率_Hz": preprocessed["fs"],
        "原始采样率_Hz": preprocessed["source_fs"],
        "使用时长_s": preprocessed["duration_seconds"],
        "片段保留比例": preprocessed["keep_ratio"],
        "干净片段数": preprocessed["epoch_count"],
        "总片段数": preprocessed["epoch_total"],
        "读取器": header.reader,
        "注释数量": header.annotations,
    }
    for key, value in info_values.items():
        info.append([key, value])
    if header.notes:
        info.append(["读取提示", "；".join(header.notes)])
    if preprocessed.get("notes"):
        info.append(["预处理提示", "；".join(preprocessed["notes"])])

    settings_sheet = workbook.create_sheet("分析设置")
    settings_sheet.append(["参数", "值"])
    for key, value in asdict(settings).items():
        settings_sheet.append([key, str(value)])

    summary_sheet = workbook.create_sheet("特征汇总")
    summary_sheet.append(["特征", "数值"])
    for row in summary_table.itertuples(index=False):
        summary_sheet.append([row[0], row[1]])

    channel_sheet = workbook.create_sheet("通道级特征")
    for row in channel_table.itertuples(index=False, name=None):
        channel_sheet.append(list(row))

    qc_sheet = workbook.create_sheet("通道质控")
    qc_columns = ["原始序号", "通道", "单位", "状态", "标准差", "是否饱和或异常"]
    qc_sheet.append(qc_columns)
    for record in preprocessed.get("channel_records", []):
        qc_sheet.append([record.get(column, "") for column in qc_columns])

    network_sheet = workbook.create_sheet("脑网络指标")
    network_sheet.append(["指标", *settings.network_bands])
    network_features = metadata.get("network_features", {})
    network_metrics = (
        "mean_strength", "strength_std", "mean_degree", "degree_std",
        "clustering", "global_efficiency", "characteristic_path_length", "small_worldness",
    )
    for metric in network_metrics:
        network_sheet.append([
            metric,
            *[network_features.get(f"{band}_plv_{metric}", math.nan) for band in settings.network_bands],
        ])

    description_sheet = workbook.create_sheet("字段说明")
    description_sheet.append(["特征类别", "说明"])
    for category, description in FEATURE_DESCRIPTIONS.items():
        description_sheet.append([category, description])
    for category, names in metadata.get("feature_categories", {}).items():
        description_sheet.append([f"{category}（{len(names)} 项）", "；".join(names)])

    annotation_sheet = workbook.create_sheet("注释")
    annotation_sheet.append(["序号", "注释"])
    for index, annotation in enumerate(annotations or [], start=1):
        annotation_sheet.append([index, annotation])
    if not annotations:
        annotation_sheet.append([0, "文件未包含可解析注释。"])

    for sheet in workbook.worksheets:
        sheet.freeze_panes = "A2"
        for cell in sheet[1]:
            cell.fill = header_fill
            cell.font = title_font
            cell.alignment = Alignment(horizontal="center", vertical="center")
            cell.border = border
        for row in sheet.iter_rows(min_row=2):
            for cell in row:
                cell.font = normal_font
                cell.border = border
                if isinstance(cell.value, float):
                    cell.number_format = "0.000000"
        for column_cells in sheet.columns:
            max_length = 0
            letter = get_column_letter(column_cells[0].column)
            for cell in column_cells[:200]:
                if cell.value is not None:
                    max_length = max(max_length, len(str(cell.value)))
            sheet.column_dimensions[letter].width = min(max(max_length + 2, 12), 48)
    info["A1"].fill = header_fill
    workbook.save(path)
    return path


def analyze_eeg_file(
    input_path: PathLike,
    output_dir: PathLike,
    settings: AnalysisSettings | None = None,
    explicit_fs: float | None = None,
    selected_channels: Iterable[str] | None = None,
    auto_eeg_only: bool = True,
    include_time: bool = True,
    include_frequency: bool = True,
    include_time_frequency: bool = True,
    include_network: bool = True,
    save_preprocessed: bool = False,
    progress: ProgressCallback = default_progress,
) -> dict[str, Any]:
    """一次性执行读取、预处理、四类特征提取、绘图与 Excel/CSV 导出。"""
    settings = settings or AnalysisSettings()
    drop_prefixes: list[str] = []
    source = Path(input_path).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    slug = _safe_slug(source.stem)

    progress(0.01, "读取脑电文件")
    data, labels, units, header, annotations = read_eeg_file(
        source,
        selected_channels=selected_channels,
        auto_eeg_only=auto_eeg_only,
        max_channels=settings.max_channels,
    )
    header.channel_labels = labels  # type: ignore[attr-defined]
    header.channel_units = units    # type: ignore[attr-defined]

    preprocessed = preprocess_eeg(
        data, header, settings,
        explicit_fs=explicit_fs,
        selected_channels=None,
        progress=progress,
    )
    provenance = {
        "pipeline_version": PIPELINE_VERSION,
        "python": platform.python_version(),
        "packages": {name: package_version(name) for name in ("numpy", "scipy", "pandas", "matplotlib", "openpyxl")},
        "settings": asdict(settings),
        "reference": preprocessed["reference"],
        "source_channel_units": preprocessed["source_channel_units"],
        "analysis_voltage_unit": "µV",
        "kept_epoch_indices": preprocessed["kept_epoch_indices"].tolist(),
        "original_epoch_samples": preprocessed["original_epoch_samples"],
        "annotation_policy": "timed text only; BAD annotations not automatically excluded",
        "boundary_policy": "retained epochs concatenated; exploratory features may contain join effects",
    }
    channel_table, summary_table, metadata = extract_features(preprocessed, settings, progress)

    if not include_time:
        drop_prefixes.append("时域__")
    if not include_frequency:
        drop_prefixes.append("频域__")
    if not include_time_frequency:
        drop_prefixes.append("时频域__")
    if not include_network:
        drop_prefixes.append("网络__")
        settings = AnalysisSettings(
            target_fs=settings.target_fs,
            lowpass_hz=settings.lowpass_hz,
            notches=settings.notches,
            epoch_seconds=settings.epoch_seconds,
            amplitude_threshold_uv=settings.amplitude_threshold_uv,
            variance_z_threshold=settings.variance_z_threshold,
            flat_threshold_uv=settings.flat_threshold_uv,
            max_std_uv=settings.max_std_uv,
            duplicate_correlation=settings.duplicate_correlation,
            min_channels=settings.min_channels,
            min_keep_ratio=settings.min_keep_ratio,
            density=settings.density,
            network_bands=(),
            max_network_samples=settings.max_network_samples,
            max_channels=settings.max_channels,
        )
    if drop_prefixes:
        summary_table = summary_table.loc[
            ~summary_table["特征"].astype(str).str.startswith(tuple(drop_prefixes))
        ].reset_index(drop=True)
        metadata["feature_categories"] = {
            category: [
                name for name in names
                if not any(name.startswith(prefix) for prefix in drop_prefixes)
            ]
            for category, names in metadata["feature_categories"].items()
        }

    feature_csv = output / f"{slug}_特征_通道级.csv"
    summary_csv = output / f"{slug}_特征_汇总.csv"
    channel_table.to_csv(feature_csv, index=False, encoding="utf-8-sig")
    summary_table.to_csv(summary_csv, index=False, encoding="utf-8-sig")

    figure_paths = save_figures(preprocessed, channel_table, metadata, output, slug, progress)
    workbook_path = build_feature_workbook(
        channel_table, summary_table, header, preprocessed, metadata,
        output / f"{slug}_脑电特征分析.xlsx", settings, annotations,
    )

    report_path = output / f"{slug}_分析报告.txt"
    with report_path.open("w", encoding="utf-8") as handle:
        handle.write("脑电特征自动分析报告\n")
        handle.write("=" * 52 + "\n")
        handle.write("方法与软件参数：\n" + json.dumps(provenance, ensure_ascii=False, indent=2) + "\n")
        handle.write(f"输入文件：{source}\n")
        handle.write(f"分析时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        handle.write(f"原始格式/时长：{header.format} / {header.duration_seconds:.2f} s\n")
        handle.write(f"原始采样率：{preprocessed['source_fs']:.6g} Hz；分析采样率：{preprocessed['fs']:.6g} Hz\n")
        handle.write(f"通道：{len(preprocessed['channel_names'])} / {header.analysis_channel_count or header.signal_count}\n")
        handle.write(f"片段保留比例：{preprocessed['keep_ratio']:.2%}（{preprocessed['epoch_count']}/{preprocessed['epoch_total']}）\n")
        handle.write("特征角度：时域、频域、时频域、脑网络、质量控制、汇总\n")
        if preprocessed.get("notes"):
            handle.write("\n预处理提示：\n")
            for note in preprocessed["notes"]:
                handle.write(f"- {note}\n")
        if header.notes:
            handle.write("\n读取提示：\n")
            for note in header.notes:
                handle.write(f"- {note}\n")
        handle.write("\n输出文件：\n")
        for item in [feature_csv, summary_csv, workbook_path, *figure_paths]:
            handle.write(f"- {item.name}\n")

    cache_path = None
    if save_preprocessed:
        cache_path = output / "comparison_cache.npz"
        np.savez_compressed(
            cache_path, data=preprocessed["data"], fs=preprocessed["fs"],
            labels=np.asarray(preprocessed["channel_names"], dtype=str),
            units=np.asarray(preprocessed["channel_units"], dtype=str),
            pipeline_version=PIPELINE_VERSION,
            kept_epoch_indices=preprocessed["kept_epoch_indices"],
            original_epoch_samples=preprocessed["original_epoch_samples"],
        )
    progress(1.0, "分析完成")
    return {
        "input": str(source),
        "output_dir": str(output),
        "cache_path": str(cache_path) if cache_path else None,
        "pipeline_version": PIPELINE_VERSION,
        "provenance": provenance,
        "header": header.to_dict(),
        "preprocessed": {
            "fs": preprocessed["fs"],
            "source_fs": preprocessed["source_fs"],
            "channels": preprocessed["channel_names"],
            "keep_ratio": preprocessed["keep_ratio"],
            "epoch_count": preprocessed["epoch_count"],
            "epoch_total": preprocessed["epoch_total"],
            "notes": preprocessed["notes"],
        },
        "metadata": {
            "bands": metadata["bands"],
            "feature_counts": {key: len(value) for key, value in metadata["feature_categories"].items()},
        },
        "outputs": {
            "channel_features_csv": str(feature_csv),
            "summary_features_csv": str(summary_csv),
            "workbook": str(workbook_path),
            "report": str(report_path),
            "figures": [str(item) for item in figure_paths],
        },
        "feature_rows": int(len(channel_table)),
        "feature_columns": int(len(channel_table.columns)),
        "summary_rows": int(len(summary_table)),
    }


def make_example_bdf(path: PathLike, seconds: float = 20.0, fs: float = 500.0) -> Path:
    """生成用于界面/流水线验证的示例 BDF；不用于任何科研结论。"""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    labels = ["Fp1", "Fp2", "F3", "F4", "C3", "C4", "Cz", "P3", "P4", "O1", "O2"]
    channels = len(labels)
    samples_per_record = int(fs)
    record_duration = 1.0
    record_count = int(seconds)
    start = datetime(2024, 1, 1, 8, 30, 0)

    fixed = bytearray(256)
    fixed[0:8] = b"\xffBIOSEMI"
    # Keep every fixed-header field exactly 80 bytes; assigning a longer or
    # shorter byte string to a bytearray slice resizes the whole EDF/BDF header.
    fixed[8:88] = b"Study-001".ljust(80, b" ")
    fixed[88:168] = b"Startdate 01-JAN-2024".ljust(80, b" ")
    fixed[168:176] = start.strftime("%d.%m.%y").encode("ascii")
    fixed[176:184] = start.strftime("%H.%M.%S").encode("ascii")
    fixed[184:192] = f"{256 + (channels + 1) * 256:>8}".encode("ascii")
    fixed[192:236] = "24BIT+EDF Annotations".encode("ascii").ljust(44)
    fixed[236:244] = f"{record_count:>8}".encode("ascii")
    fixed[244:252] = f"{record_duration:>8}".encode("ascii")
    fixed[252:256] = f"{channels + 1:>4}".encode("ascii")

    signal_header = bytearray()
    all_labels = labels + ["Status"]
    transducers = ["Ag/AgCl EEG"] * channels + ["Digital"]
    dimensions = ["µV"] * channels + [""]
    physical_min = [-750.0] * channels + [0.0]
    physical_max = [750.0] * channels + [0.0]
    digital_min = [-8388608] * channels + [0]
    digital_max = [8388607] * channels + [65535]
    prefilter = ["HP:0.5Hz LP:45Hz"] * channels + [""]
    samples = [samples_per_record] * channels + [samples_per_record]
    reserved = [""] * (channels + 1)

    fields = [
        (all_labels, 16), (transducers, 80), (dimensions, 8),
        (physical_min, 8), (physical_max, 8), (digital_min, 8), (digital_max, 8),
        (prefilter, 80), (samples, 8), (reserved, 32),
    ]
    for values, width in fields:
        for value in values:
            text = str(value)
            signal_header.extend(text.encode("latin-1", errors="replace")[:width].ljust(width))

    rng = np.random.default_rng(20240101)
    time_axis = np.arange(samples_per_record) / fs
    channels_data: list[np.ndarray] = []
    for index in range(channels):
        alpha = 35.0 * (1.0 + 0.15 * np.sin(2 * np.pi * (0.2 + index * 0.03) * np.arange(record_count)[:, None]))
        theta = 18.0
        beta = 8.0
        noise = rng.normal(0, 8.0, size=(record_count, samples_per_record))
        theta_wave = theta * np.sin(2 * np.pi * 6.0 * time_axis[None, :] + rng.uniform(0, 2 * np.pi))
        alpha_wave = alpha * np.sin(2 * np.pi * (10.0 + index * 0.08) * time_axis[None, :] + rng.uniform(0, 2 * np.pi))
        beta_wave = beta * np.sin(2 * np.pi * 20.0 * time_axis[None, :] + rng.uniform(0, 2 * np.pi))
        line_noise = 4.0 * np.sin(2 * np.pi * 50.0 * time_axis[None, :])
        signal_uv = noise + theta_wave + alpha_wave + beta_wave + line_noise
        signal_uv -= np.mean(signal_uv)
        channels_data.append(signal_uv)

    annotation_signal = np.zeros((record_count, samples_per_record), dtype=np.float64)
    for record in range(record_count):
        if record in {5, 12} and record < record_count:
            annotation_signal[record, int(0.5 * fs):int(1.0 * fs)] = 999

    def encode_24bit(values: np.ndarray) -> np.ndarray:
        clipped = np.clip(np.rint(values), -8388608, 8388607).astype(np.int64)
        unsigned = clipped.astype(np.int64) & 0xFFFFFF
        bytes_view = np.empty((unsigned.size, 3), dtype=np.uint8)
        bytes_view[:, 0] = unsigned & 0xFF
        bytes_view[:, 1] = (unsigned >> 8) & 0xFF
        bytes_view[:, 2] = (unsigned >> 16) & 0xFF
        return bytes_view

    body = bytearray()
    gains = [
        (physical_max[index] - physical_min[index]) / (digital_max[index] - digital_min[index])
        for index in range(channels)
    ]
    for record in range(record_count):
        for index in range(channels):
            digital = (channels_data[index][record] - physical_min[index]) / gains[index] + digital_min[index]
            body.extend(encode_24bit(digital).reshape(-1))
        status = annotation_signal[record].astype(np.int64)
        body.extend(encode_24bit(status).reshape(-1))

    # BDF 的 Status/注释通道必须是无符号 24 位；上面的通用编码按有符号处理。
    # 这里仅重建状态通道，以保证 BDF+C 注释可被读取器识别。
    status_records = annotation_signal.astype(np.int64)
    unsigned = status_records.reshape(-1) & 0xFFFFFF
    status_bytes = np.empty((unsigned.size, 3), dtype=np.uint8)
    status_bytes[:, 0] = unsigned & 0xFF
    status_bytes[:, 1] = (unsigned >> 8) & 0xFF
    status_bytes[:, 2] = (unsigned >> 16) & 0xFF
    eeg_bytes_per_record = channels * samples_per_record * 3
    for record in range(record_count):
        start = record * (eeg_bytes_per_record + samples_per_record * 3) + eeg_bytes_per_record
        end = start + samples_per_record * 3
        body[start:end] = status_bytes[record * samples_per_record:(record + 1) * samples_per_record].tobytes()

    output.write_bytes(bytes(fixed) + bytes(signal_header) + bytes(body))
    return output


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser(description="BDF/EDF/GDF 脑电特征提取")
    parser.add_argument("input", help="输入 .bdf/.edf/.gdf 文件")
    parser.add_argument("--out", default="results", help="输出目录")
    parser.add_argument("--fs", type=float, default=None, help="手动采样率（Hz）")
    parser.add_argument("--example", action="store_true", help="生成示例 BDF")
    args = parser.parse_args()

    if args.example:
        target = make_example_bdf(args.input)
        print(target)
    else:
        result = analyze_eeg_file(args.input, args.out, explicit_fs=args.fs)
        print(json.dumps(result, ensure_ascii=False, indent=2))
