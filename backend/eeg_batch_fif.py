"""批量BDF→FIF的保真转换：只转换格式，不滤波、不猜坏道、不删样本。

与网页的原始导出使用相同的NativeBDF标定、MNE装配和FIF读回核验。
批处理不画几千张重复质控图；保留每份转换记录，后续特征从FIF重新读取。
"""

from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import shutil
import uuid

import numpy as np

from eeg_mne_runtime import load_mne
from eeg_montage import template
from eeg_raw_bdf import NativeBDF
from eeg_raw_qc import assemble_mapping, roundtrip


CONVERTER_VERSION = "batch_native_fif_v1"


def file_sha256(path):
    """像文件指纹一样核对内容。只读，不改写源BDF。"""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def convert_bdf_to_fif(source_path, recordinfo=None, destination_dir=None):
    """保存同目录fdata.fif，并检查所有原始样本、坐标和注释。

    已有且有同版本可信清单的文件可断点续跑；其他已有FIF不自动覆盖。
    不同原采样率分组保存，EEG最多的一组作为主fdata.fif。
    """
    source_path = Path(source_path).resolve()
    destination = Path(destination_dir).resolve() if destination_dir else source_path.parent
    destination.mkdir(parents=True, exist_ok=True)
    source = NativeBDF(source_path)
    before = source_path.stat()
    source_hash = file_sha256(source_path)
    manifest_path = destination / "fdata_batch_manifest.json"
    mapping = assemble_mapping(source, recordinfo)
    mapping_hash = hashlib.sha256(
        json.dumps(mapping["rows"], ensure_ascii=False, sort_keys=True).encode("utf8")
    ).hexdigest()
    if manifest_path.is_file():
        previous = json.loads(manifest_path.read_text(encoding="utf8"))
        if (
            previous.get("version") == CONVERTER_VERSION
            and previous.get("source_sha256") == source_hash
            and previous.get("mapping_sha256") == mapping_hash
        ):
            if all(
                (destination / name).is_file() for r in previous["groups"] for name in r["files"]
            ):
                return {**previous, "reused": True, "mapping": mapping}
    if (destination / "fdata.fif").exists():
        raise ValueError("已有FIF缺少相符的批次清单；为保留可能的人工标记，本轮不覆盖。")

    mne = load_mne()
    from mne._fiff.constants import FIFF

    by_name = {row["原始通道"]: row for row in mapping["rows"]}
    groups = defaultdict(list)
    for channel in source.signals:
        groups[channel["spr"]].append(channel)
    ordered = sorted(
        groups.values(),
        key=lambda rows: sum(by_name[r["name"]]["MNE类型"] == "eeg" for r in rows),
        reverse=True,
    )
    events = source.annotations()
    stage = destination / (".fdata_batch_tmp_" + uuid.uuid4().hex)
    stage.mkdir()
    published = []
    try:
        group_rows = []
        for number, channels in enumerate(ordered, 1):
            names = [row["name"] for row in channels]
            fs = channels[0]["fs"]
            # 这里保留原始采样值。MNE电压使用V，非电压信号保留原标定数值。
            values = np.stack([source.read(row) for row in channels])
            info = mne.create_info(names, fs, [by_name[name]["MNE类型"] for name in names])
            for entry, channel in zip(info["chs"], channels):
                entry["unit"] = (
                    FIFF.FIFF_UNIT_V if channel["scale"] is not None else FIFF.FIFF_UNIT_NONE
                )
                entry["unit_mul"] = FIFF.FIFF_UNITM_NONE
            info["description"] = (
                "Unfiltered calibrated BDF; no rereference/resample/notch; template head coordinates, not individual measurements; "
                "original local clock (timezone unverified): "
                + source.local_date
                + " "
                + source.local_time
            )
            raw = mne.io.RawArray(values, info, copy="auto", verbose="ERROR")
            positions = {
                name: [by_name[name][key] for key in ("x_m", "y_m", "z_m")]
                for name in names
                if by_name[name]["可装配"]
            }
            if positions:
                fiducials = template()["fiducials"]
                montage = mne.channels.make_dig_montage(
                    ch_pos=positions,
                    nasion=fiducials["nasion"],
                    lpa=fiducials["lpa"],
                    rpa=fiducials["rpa"],
                    coord_frame="head",
                )
                raw.set_montage(montage, on_missing="ignore", verbose="ERROR")
            descriptions = [
                (
                    "BAD_source_acquisition_skip"
                    if e["description"].lower().startswith("bad_acq_skip")
                    else e["description"].encode("latin1", "xmlcharrefreplace").decode("latin1")
                )
                for e in events
            ]
            raw.set_annotations(
                mne.Annotations(
                    [e["onset"] for e in events],
                    [e["duration"] for e in events],
                    descriptions,
                    extras=[
                        {
                            "original_description": e["description"],
                            "exact_onset_s": e["onset"],
                            "exact_duration_s": e["duration"],
                            "source": "BDF",
                        }
                        for e in events
                    ],
                )
            )
            name = "fdata.fif" if number == 1 else f"fdata_group{number}.fif"
            paths, error = roundtrip(raw, stage / name, mapping)
            group_rows.append(
                {
                    "file": name,
                    "files": [Path(p).name for p in paths],
                    "fs": fs,
                    "samples": int(raw.n_times),
                    "channels": names,
                    "max_roundtrip_error": error,
                }
            )
            raw.close()
            del values, raw
        after = source_path.stat()
        if before.st_size != after.st_size or before.st_mtime_ns != after.st_mtime_ns:
            raise ValueError("转换期间源BDF发生变化，取消发布FIF。")
        # Windows的rename遇到同名目标会报错，不会悄悄覆盖别人的导出。
        for group in group_rows:
            for name in group["files"]:
                target = destination / name
                os.rename(stage / name, target)
                published.append(target)
        result = {
            "version": CONVERTER_VERSION,
            "source_sha256": source_hash,
            "source_bytes": before.st_size,
            "source_mtime_ns": before.st_mtime_ns,
            "mapping_sha256": mapping_hash,
            "duration_seconds": source.duration,
            "groups": group_rows,
            "mne_version": mne.__version__,
            "filtering": "none",
            "automatic_bad_channel_marking": False,
            "automatic_eye_labeling": False,
            "source_local_date": source.local_date,
            "source_local_time": source.local_time,
            "fif_path": str(destination / "fdata.fif"),
        }
        manifest_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf8")
        return {**result, "reused": False, "mapping": mapping}
    except Exception:
        # 只撤销本函数刚刚新建的派生文件，不碰原始BDF或其他既有FIF。
        for path in published:
            path.unlink(missing_ok=True)
        raise
    finally:
        if stage.resolve().parent == destination and stage.name.startswith(".fdata_batch_tmp_"):
            shutil.rmtree(stage)
