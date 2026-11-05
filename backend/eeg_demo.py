"""教学用合成数据。只生成虚构的测试波形，不读取任何产妇资料。"""

from __future__ import annotations

import math
from datetime import datetime
from pathlib import Path
import numpy as np
from eeg_models import PathLike


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
        (all_labels, 16),
        (transducers, 80),
        (dimensions, 8),
        (physical_min, 8),
        (physical_max, 8),
        (digital_min, 8),
        (digital_max, 8),
        (prefilter, 80),
        (samples, 8),
        (reserved, 32),
    ]
    for values, width in fields:
        for value in values:
            text = str(value)
            signal_header.extend(text.encode("latin-1", errors="replace")[:width].ljust(width))

    rng = np.random.default_rng(20240101)
    time_axis = np.arange(samples_per_record) / fs
    channels_data: list[np.ndarray] = []
    for index in range(channels):
        alpha = 35.0 * (
            1.0 + 0.15 * np.sin(2 * np.pi * (0.2 + index * 0.03) * np.arange(record_count)[:, None])
        )
        theta = 18.0
        beta = 8.0
        noise = rng.normal(0, 8.0, size=(record_count, samples_per_record))
        theta_wave = theta * np.sin(
            2 * np.pi * 6.0 * time_axis[None, :] + rng.uniform(0, 2 * np.pi)
        )
        alpha_wave = alpha * np.sin(
            2 * np.pi * (10.0 + index * 0.08) * time_axis[None, :] + rng.uniform(0, 2 * np.pi)
        )
        beta_wave = beta * np.sin(2 * np.pi * 20.0 * time_axis[None, :] + rng.uniform(0, 2 * np.pi))
        line_noise = 4.0 * np.sin(2 * np.pi * 50.0 * time_axis[None, :])
        signal_uv = noise + theta_wave + alpha_wave + beta_wave + line_noise
        signal_uv -= np.mean(signal_uv)
        channels_data.append(signal_uv)

    annotation_signal = np.zeros((record_count, samples_per_record), dtype=np.float64)
    for record in range(record_count):
        if record in {5, 12} and record < record_count:
            annotation_signal[record, int(0.5 * fs) : int(1.0 * fs)] = 999

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
            digital = (channels_data[index][record] - physical_min[index]) / gains[
                index
            ] + digital_min[index]
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
        body[start:end] = status_bytes[
            record * samples_per_record : (record + 1) * samples_per_record
        ].tobytes()

    output.write_bytes(bytes(fixed) + bytes(signal_header) + bytes(body))
    return output
