"""第2站：把文件读成“通道 × 采样点”的数值表。

标准信号交给MNE；单位换算和文件完整性检查在这里明确说明。"""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any, Iterable
import numpy as np
from eeg_models import HeaderInfo, SUPPORTED_EXTENSIONS, NON_EEG_LABELS, PathLike
from eeg_mne_runtime import load_mne


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
    """统一通道名的大小写和常见前后缀，用来匹配名称，不修改波形。"""
    name = _clean_text(label).upper().replace("EEG", "").replace("REF", "")
    name = re.sub(r"[\s\-_]+", "", name)
    name = re.sub(r"\(.*?\)|\[.*?\]", "", name)
    return name.strip() or "UNKNOWN"


def _is_annotation_label(label: str) -> bool:
    normalized = _normalize_channel_name(label)
    return normalized in {
        "EDFANNOTATIONS",
        "BDFANNOTATIONS",
        "GDFANNOTATIONS",
        "ANNOTIONS",
    } or normalized.endswith("ANNOTATIONS")


def _is_eeg_like(label: str, physical_unit: str) -> bool:
    normalized = _normalize_channel_name(label)
    if not normalized or normalized in NON_EEG_LABELS:
        return False
    if normalized.startswith(("X", "EMG", "EOG", "ECG", "EKG", "ACC", "GYRO")):
        return False
    # Voltage units alone cannot establish that a channel is EEG (e.g. EHG).
    return bool(
        re.fullmatch(
            r"(?:FP[12Z]|AF[1-8Z]|F[1-9Z]|FC[1-6Z]|FT[7-9]|C[1-6Z]|"
            r"CP[1-6Z]|TP[7-9]|P[1-9Z]|PO[1-8Z]|O[12Z]|T[1-9])",
            normalized,
        )
    )


def voltage_to_uv(data: np.ndarray, units: Sequence[str]) -> np.ndarray:
    """把明确声明的V/mV/µV/nV换成µV。未知单位报错，不猜设备倍率。"""
    factors = {
        "v": 1e6,
        "mv": 1e3,
        "uv": 1.0,
        "µv": 1.0,
        "μv": 1.0,
        "nv": 1e-3,
        "microvolt": 1.0,
        "microvolts": 1.0,
    }
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
    """根据已选名称和单位挑出EEG候选；不能把参考、Status或spike计数当成EEG补位。"""
    selected_set = {_normalize_channel_name(item) for item in (selected or []) if _clean_text(item)}
    if selected_set:
        indices = [
            index
            for index, label in enumerate(labels)
            if _normalize_channel_name(label) in selected_set and not _is_annotation_label(label)
        ]
        return indices, [labels[index] for index in indices]

    if auto_eeg_only:
        indices = [
            index
            for index, label in enumerate(labels)
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
    """旧导入名的兼容入口，不再自行解码标准EEG采样值。

    老测试或外部脚本可能仍导入这个名字，所以保留它。
    真正读取信号时统一走下面的MNE入口。
    """

    def __init__(self, path):
        self.path = Path(path)

    @staticmethod
    def _parse_float_field(raw, fallback):
        return _parse_number(_parse_ascii(raw), fallback)

    @staticmethod
    def _parse_int_field(raw, fallback):
        try:
            return int(_parse_ascii(raw))
        except ValueError:
            return fallback

    @classmethod
    def can_read(cls, path):
        with Path(path).open("rb") as stream:
            signature = stream.read(8)
        return signature.startswith((b"0", b"\xff"))

    def read(self, max_channels=128):
        return read_eeg_file(self.path, auto_eeg_only=False, max_channels=max_channels)


class _GDFReader(_EDFReader):
    """保留旧名称，GDF二进制读取实际交给mne.io.read_raw_gdf。"""


def _read_bdf_with_mne(path, selected_channels, auto_eeg_only, max_channels):
    """MNE负责EEG采样；原生头信息用来阻止隐式上采样和猜测单位。"""
    from collections import Counter
    from eeg_raw_bdf import NativeBDF

    source = NativeBDF(path)
    source_labels = [row["name"] for row in source.signals]
    source_units = [row["unit"] for row in source.signals]
    indices, _ = _choose_channels(source_labels, source_units, auto_eeg_only, selected_channels)
    if not indices:
        raise ValueError("没有找到可分析的EEG通道；不会改用Status或spike计数冒充脑电。")
    selected = [source.signals[i] for i in indices]

    # 一个二维数值表必须使用同一个采样率。先选主要采样率，再交给MNE读，
    # 这样MNE不会把低速辅助通道自动插值到高速EEG上。
    main_samples = Counter(row["spr"] for row in selected).most_common(1)[0][0]
    rows = [row for row in selected if row["spr"] == main_samples]
    omitted = [row["name"] for row in selected if row["spr"] != main_samples]
    if len(rows) > max_channels:
        raise ValueError("EEG通道超过配置上限，不能静默截断。")
    names = [row["name"] for row in rows]
    mne = load_mne()
    raw = mne.io.read_raw_bdf(
        path, include=names, stim_channel=None, infer_types=False, preload=True, verbose="ERROR"
    )
    try:
        data = raw.get_data(picks=names)
        units = []
        for index, row in enumerate(rows):
            if row["scale"] is not None:
                # MNE给出的电压是V。乘10的6次方只是把单位改成µV，
                # 不是设备校准，更不是额外除以1000。
                data[index] *= 1e6
                units.append("µV")
            else:
                # 非电压计数不能按微伏解释。该小分支保留其原生数值，
                # 供RecordInfo核对后排除；不会交给EEG特征冒充电压。
                data[index] = source.read(row)
                units.append(row["unit"])
        events = source.annotations()
        annotations = [f"{e['onset']:g}s / {e['duration']:g}s: {e['description']}" for e in events]
        notes = [
            "标准EEG采样由MNE read_raw_bdf读取；电压由SI伏特明确换算为µV。",
            "原始质控/FIF导出仍使用独立的保真入口，保留所有原采样率组与完整Status。",
        ]
        if omitted:
            notes.append("特征分析跳过不同采样率通道：" + ", ".join(omitted) + "；未隐式上采样。")
        with path.open("rb") as stream:
            fixed = stream.read(256)
        header = HeaderInfo(
            file_name=path.name,
            file_path=str(path),
            format="BDF",
            version=_parse_ascii(fixed[:8]),
            patient=_parse_ascii(fixed[8:88]),
            recording=_parse_ascii(fixed[88:168]),
            start_date=source.local_date,
            start_time=source.local_time,
            channel_count=len(source.channels),
            signal_count=len(rows),
            data_records=source.records,
            record_duration=source.record_seconds,
            duration_seconds=source.duration,
            physical_minimum=min(r["pmin"] for r in rows),
            physical_maximum=max(r["pmax"] for r in rows),
            digital_minimum=int(min(r["dmin"] for r in rows)),
            digital_maximum=int(max(r["dmax"] for r in rows)),
            annotations=len(events),
            reader=f"MNE {mne.__version__} read_raw_bdf",
            notes=notes,
            sampling_rate=rows[0]["fs"],
        )
        return data, names, units, header, annotations
    finally:
        raw.close()


def _read_edf_or_gdf_with_mne(path, selected_channels, auto_eeg_only, max_channels):
    """EDF/GDF也调用MNE公开读取器，不另写一套二进制解码器。"""
    mne = load_mne()
    reader = mne.io.read_raw_edf if path.suffix.lower() == ".edf" else mne.io.read_raw_gdf
    raw = reader(path, preload=True, verbose="ERROR")
    try:
        kinds = raw.get_channel_types()
        allowed = [i for i, kind in enumerate(kinds) if kind in ("eeg", "eog", "ecg", "emg")]
        labels = [raw.ch_names[i] for i in allowed]
        indices, _ = _choose_channels(
            labels, ["µV"] * len(labels), auto_eeg_only, selected_channels
        )
        picks = [allowed[i] for i in indices]
        if not picks:
            raise ValueError("MNE未识别到可用于分析的EEG电压通道。")
        if len(picks) > max_channels:
            raise ValueError("EEG通道数超过配置上限。")
        names = [raw.ch_names[i] for i in picks]
        data = raw.get_data(picks=picks) * 1e6
        units = ["µV"] * len(picks)
        fs = float(raw.info["sfreq"])
        date = raw.info.get("meas_date")
        annotations = [
            f"{a:g}s / {b:g}s: {label}"
            for a, b, label in zip(
                raw.annotations.onset, raw.annotations.duration, raw.annotations.description
            )
        ]
        header = HeaderInfo(
            file_name=path.name,
            file_path=str(path),
            format=path.suffix[1:].upper(),
            version="MNE",
            patient="",
            recording="",
            start_date=date.strftime("%d.%m.%y") if date else "",
            start_time=date.strftime("%H.%M.%S") if date else "",
            channel_count=len(raw.ch_names),
            signal_count=len(names),
            data_records=0,
            record_duration=0.0,
            duration_seconds=raw.n_times / fs,
            physical_minimum=0.0,
            physical_maximum=0.0,
            digital_minimum=0,
            digital_maximum=0,
            annotations=len(annotations),
            reader=f"MNE {mne.__version__} {reader.__name__}",
            notes=["EDF/GDF由MNE公开接口读取，电压统一为µV；其余采集信息不猜测。"],
            sampling_rate=fs,
        )
        return data, names, units, header, annotations
    finally:
        raw.close()


def read_eeg_file(path, selected_channels=None, auto_eeg_only=True, max_channels=128):
    """统一入口：输入文件路径，返回数据表、通道名、单位、头信息和注释。

    例如data.shape=(5,4000)，表示5个通道，每个通道4000个采样点。
    先用后缀分派，再让MNE检查实际内容；不能把改后缀当成格式转换。
    """
    file_path = Path(path).expanduser().resolve()
    if not file_path.is_file():
        raise FileNotFoundError(f"找不到输入文件：{file_path}")
    if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError("支持BDF/FIF/EDF/GDF文件。")
    if file_path.suffix.lower() == ".fif":
        from eeg_fif import read_fif

        data, labels, units, header, annotations = read_fif(file_path, max_channels)
        if selected_channels:
            wanted = {_normalize_channel_name(name) for name in selected_channels}
            keep = [i for i, name in enumerate(labels) if _normalize_channel_name(name) in wanted]
            if not keep:
                raise ValueError("FIF指定通道不存在或已标为坏道。")
            data = data[keep]
            labels = [labels[i] for i in keep]
            units = [units[i] for i in keep]
    elif file_path.suffix.lower() == ".bdf":
        data, labels, units, header, annotations = _read_bdf_with_mne(
            file_path, selected_channels, auto_eeg_only, max_channels
        )
    else:
        data, labels, units, header, annotations = _read_edf_or_gdf_with_mne(
            file_path, selected_channels, auto_eeg_only, max_channels
        )
    if not np.isfinite(data).all():
        raise ValueError("采样值包含NaN或无穷大，不能静默置零。")
    header.analysis_channel_count = len(labels)
    header.channel_labels = labels
    header.channel_units = units
    return data, labels, units, header, annotations
