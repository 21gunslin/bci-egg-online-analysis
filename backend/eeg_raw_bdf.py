"""最后再读的保真兼容层：逐字节保留BDF原始采样率、数字计数和Status。普通EEG特征入口使用MNE；这里只服务原始检视与无损导出。

Lossless calibrated BDF signal access, without MNE's mixed-rate upsampling.

Annotation bytes are metadata, not sampled EEG. All other signals (including
status words) are decoded without trigger bit masks, type-dependent gains,
channel selection, detrending, or any signal processing.
"""

from pathlib import Path
import math
import re

import numpy as np


VOLTAGE = {"v": 1.0, "mv": 1e-3, "uv": 1e-6, "nv": 1e-9}
TAL = re.compile(rb"([+-]?(?:\d+(?:\.\d*)?|\.\d+))(?:\x15([\d.]+))?\x14(.*)", re.S)


def unit_scale(unit):
    return VOLTAGE.get(unit.strip().lower().replace("µ", "u").replace("μ", "u"))


class NativeBDF:
    """Strict continuous 24-bit reader; malformed/truncated inputs are rejected."""

    def __init__(self, path):
        self.path = Path(path)
        with self.path.open("rb") as stream:
            fixed = stream.read(256)
            if len(fixed) != 256 or fixed[:1] != b"\xff":
                raise ValueError("需要标准24位BDF文件头。")
            self.reserved = fixed[192:236].decode("latin1").strip()
            if "BDF+D" in self.reserved or "EDF+D" in self.reserved:
                raise ValueError(
                    "本入口暂不接收不连续BDF+D；不能把时间缺口压缩后宣称保留原始时间。"
                )
            self.header_bytes = int(fixed[184:192])
            self.records = int(fixed[236:244])
            self.record_seconds = float(fixed[244:252])
            count = int(fixed[252:256])
            if (
                not 0 < count <= 1024
                or not math.isfinite(self.record_seconds)
                or self.record_seconds <= 0
            ):
                raise ValueError("BDF通道数或记录时长无效。")
            if self.header_bytes < 256 * (count + 1):
                raise ValueError("BDF头长度不足。")
            header = stream.read(count * 256)
            if len(header) != count * 256:
                raise ValueError("BDF信号头被截断。")
        fields = [
            ("name", 16),
            ("transducer", 80),
            ("unit", 8),
            ("pmin", 8),
            ("pmax", 8),
            ("dmin", 8),
            ("dmax", 8),
            ("prefilter", 80),
            ("spr", 8),
            ("reserved", 32),
        ]
        decoded = {}
        offset = 0
        for key, width in fields:
            decoded[key] = [
                header[offset + i * width : offset + (i + 1) * width]
                .decode("latin1")
                .strip(" \x00")
                for i in range(count)
            ]
            offset += count * width
        self.channels = []
        seen = set()
        sample_offset = 0
        for i in range(count):
            row = {key: decoded[key][i] for key, _ in fields}
            row["spr"] = int(row["spr"])
            row["offset"] = sample_offset * 3
            sample_offset += row["spr"]
            if row["spr"] <= 0:
                raise ValueError("BDF每记录采样点数无效。")
            row["annotation"] = row["name"].upper() in ("BDF ANNOTATIONS", "EDF ANNOTATIONS")
            for key in ("pmin", "pmax", "dmin", "dmax"):
                row[key] = float(row[key])
            if not row["annotation"]:
                if not row["name"] or row["name"] in seen:
                    raise ValueError("存在空名或重复通道名；请先核对采集导联，不能静默改名。")
                seen.add(row["name"])
                if (
                    not all(math.isfinite(row[k]) for k in ("pmin", "pmax", "dmin", "dmax"))
                    or row["dmax"] <= row["dmin"]
                ):
                    raise ValueError("BDF幅值标定范围无效，不能猜测倍率。")
                if row["dmin"] < -8388608 or row["dmax"] > 8388607:
                    raise ValueError("BDF数字范围超出有符号24位范围。")
            row["fs"] = row["spr"] / self.record_seconds
            row["scale"] = unit_scale(row["unit"])
            row["digital_only"] = not row["annotation"] and row["pmax"] == row["pmin"]
            if row["digital_only"] and row["scale"] is not None:
                raise ValueError("电压通道物理标定退化，不能猜测倍率。")
            self.channels.append(row)
        self.record_bytes = sample_offset * 3
        actual, tail = divmod(self.path.stat().st_size - self.header_bytes, self.record_bytes)
        if tail or actual <= 0 or (self.records != -1 and self.records != actual):
            raise ValueError("BDF长度与声明记录数不一致，或末尾不完整；不静默截断导出。")
        self.records = actual
        self.duration = actual * self.record_seconds
        self.local_date = fixed[168:176].decode("ascii").strip()
        self.local_time = fixed[176:184].decode("ascii").strip()
        self.signals = [r for r in self.channels if not r["annotation"]]
        if not self.signals:
            raise ValueError("该BDF只有事件元数据，没有采样信号。")

    def read(self, row, start=0, stop=None, *, digital=False):
        """Read native samples. Physical voltage channels are converted to SI V."""
        total = self.records * row["spr"]
        stop = total if stop is None else min(total, int(stop))
        start = int(start)
        if not 0 <= start <= stop:
            raise ValueError("原始读取样本范围无效。")
        if start == stop:
            return np.array([], dtype=float)
        first = start // row["spr"]
        end = (stop + row["spr"] - 1) // row["spr"]
        mapped = np.memmap(
            self.path,
            dtype="u1",
            mode="r",
            offset=self.header_bytes,
            shape=(self.records, self.record_bytes),
        )
        block = np.array(
            mapped[first:end, row["offset"] : row["offset"] + row["spr"] * 3], copy=True
        ).reshape(-1, 3)
        del mapped
        values = (
            block[:, 0].astype("i4")
            | (block[:, 1].astype("i4") << 8)
            | (block[:, 2].astype("i4") << 16)
        )
        values = np.where(values & 0x800000, values - 0x1000000, values)
        values = values[start - first * row["spr"] : stop - first * row["spr"]]
        if digital:
            return values
        if row["digital_only"]:
            return values.astype("f8")
        gain = (row["pmax"] - row["pmin"]) / (row["dmax"] - row["dmin"])
        physical = (values.astype("f8") - row["dmin"]) * gain + row["pmin"]
        return physical * (row["scale"] if row["scale"] is not None else 1.0)

    def annotations(self):
        """Keep all UTF-8 TAL labels/durations and check continuous timekeeping."""
        events = []
        origin = None
        annotations = [r for r in self.channels if r["annotation"]]
        if not annotations:
            return events
        mapped = np.memmap(
            self.path,
            dtype="u1",
            mode="r",
            offset=self.header_bytes,
            shape=(self.records, self.record_bytes),
        )
        for rec in range(self.records):
            timekeeper = None
            for row in annotations:
                payload = mapped[rec, row["offset"] : row["offset"] + row["spr"] * 3].tobytes()
                for entry in payload.split(b"\x00"):
                    if not entry:
                        continue
                    match = TAL.fullmatch(entry)
                    if not match:
                        raise ValueError("BDF注释TAL格式无效，不能有损猜测事件。")
                    onset = float(match[1])
                    duration = float(match[2] or b"0")
                    if not math.isfinite(onset) or not math.isfinite(duration) or duration < 0:
                        raise ValueError("BDF注释时间无效。")
                    labels = [value for value in match[3].split(b"\x14") if value]
                    if not labels and timekeeper is None:
                        timekeeper = onset
                    for value in labels:
                        events.append(
                            {
                                "onset": onset,
                                "duration": duration,
                                "description": value.decode("utf8"),
                            }
                        )
            if timekeeper is not None:
                if origin is None:
                    origin = timekeeper - rec * self.record_seconds
                if abs(timekeeper - origin - rec * self.record_seconds) > 1e-6:
                    raise ValueError("发现BDF注释时间缺口；本入口不压缩不连续时间。")
        del mapped
        self.annotation_origin = origin or 0.0
        # FIF relative time starts at its first sample; exact original offsets are audited.
        return [{**e, "onset": e["onset"] - (origin or 0.0)} for e in events]


def waveform(path, options, mapping=None):
    """读取指定通道和时间窗。长窗仅为绘图保留极值包络，原始文件和FIF不抽稀。"""
    source = NativeBDF(path)
    names = options.get("channels", [r["name"] for r in source.signals[:12]])
    if not isinstance(names, list) or not 1 <= len(names) <= 64 or len(set(names)) != len(names):
        raise ValueError("波形检视需选择1–64个不同通道。")
    from eeg_montage import montage_mapping
    from eeg_raw_display import ordered_channels

    mapping = mapping or montage_mapping([r["name"] for r in source.signals])
    order = ordered_channels(names, mapping)
    start = float(options.get("start", 0))
    seconds = float(options.get("seconds", 10))
    if (
        not math.isfinite(start)
        or not math.isfinite(seconds)
        or not 0 <= start < source.duration
        or not min(0.01, source.duration) <= seconds <= max(0.01, source.duration)
    ):
        raise ValueError("波形检视起点或窗长无效；窗长应在0.01秒与本记录总时长之间。")
    rows = []
    lookup = {r["name"]: r for r in source.signals}
    for display_row in order:
        name = display_row["name"]
        if name not in lookup:
            raise ValueError("波形通道不存在。")
        row = lookup[name]
        fs = row["fs"]
        begin = math.floor(start * fs)
        data = source.read(row, begin, math.ceil(min(source.duration, start + seconds) * fs))
        if row["scale"] is not None:
            data = data * 1e6
        stats = {
            "min": float(np.min(data)),
            "max": float(np.max(data)),
            "median": float(np.median(data)),
            "mean": float(np.mean(data)),
            "std": float(np.std(data)),
            "ptp": float(np.ptp(data)),
            "n": int(len(data)),
        }
        stride = max(1, math.ceil(len(data) / 1400))
        points = []
        for index in range(0, len(data), stride):
            block = data[index : index + stride]
            picks = sorted(set([int(np.argmin(block)), int(np.argmax(block))]))
            points.extend([[(begin + index + p) / fs, float(block[p])] for p in picks])
        rows.append(
            {
                **display_row,
                "fs": fs,
                "unit": (
                    "µV"
                    if row["scale"] is not None
                    else (
                        "digital counts"
                        if row["digital_only"]
                        else row["unit"] or "header-unit unspecified"
                    )
                ),
                "points": points,
                "stats": stats,
            }
        )
    return {
        "start": start,
        "stop": min(source.duration, start + seconds),
        "rows": rows,
        "display_only": True,
    }
