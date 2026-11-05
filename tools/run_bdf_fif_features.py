"""第1步：全量BDF→FIF，再逐份提取论文特征。患者数据只写本机results。

每份文件独立处理。--workers 1就是最容易跟读的顺序循环；默认2进程可同时处理两份。
支持断点续跑：已完成文件不会因为重新启动而重复转换或提取。
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time
import traceback

# 防止两个工作进程各自占满全部数学运算线程。
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"
BACKEND = Path(__file__).resolve().parents[1] / "backend"
sys.path.insert(0, str(BACKEND))

import numpy as np
import pandas as pd

from eeg_batch_fif import convert_bdf_to_fif
from eeg_metadata import decode_metadata
from eeg_models import AnalysisSettings, PIPELINE_VERSION
from eeg_preprocessing import preprocess_eeg
from eeg_readers import read_eeg_file, _is_eeg_like
from eeg_thesis_features import extract_collection


FOLDER_PATTERN = re.compile(r"^(\d+)([^_]*)_(\d{4}_\d{2}_\d{2}_\d{2}_\d{2}_\d{2})_")
SETTINGS = AnalysisSettings(feature_profile="thesis_v1", target_fs=250, notches=(50.0, 100.0))


def save_json(path, value):
    """先写临时文件，再替换，避免中断后留半个JSON。NaN改成null便于阅读。"""

    def clean(x):
        if isinstance(x, dict):
            return {str(k): clean(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [clean(v) for v in x]
        if isinstance(x, np.generic):
            return clean(x.item())
        if isinstance(x, float) and not np.isfinite(x):
            return None
        return x

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(clean(value), ensure_ascii=False, indent=2), encoding="utf8")
    temporary.replace(path)


def inventory(input_root):
    """按一次记录的文件夹时间分成镇痛前/后；不把evt或spike计数当EEG。"""
    sessions = []
    subjects = {}
    for folder in sorted(Path(input_root).iterdir()):
        match = FOLDER_PATTERN.match(folder.name)
        if not folder.is_dir() or not match:
            continue
        sid = int(match[1])
        name_key = re.sub(r"\s+", "", match[2])
        subject_key = hashlib.sha256(name_key.encode("utf8")).hexdigest()
        row = {
            "id": sid,
            "identity_hash": subject_key,
            "timestamp": match[3],
            "folder": str(folder),
        }
        subjects.setdefault(sid, []).append(row)
    entries = []
    for sid, records in sorted(subjects.items()):
        records.sort(key=lambda r: r["timestamp"])
        identity_conflict = len({r["identity_hash"] for r in records}) != 1
        for order, record in enumerate(records):
            phase = (
                "pre"
                if order == 0
                else "post" if len(records) == 2 and order == 1 else "unresolved"
            )
            files = sorted(
                Path(record["folder"]).glob("*/data.bdf"),
                key=lambda p: int(p.parent.name) if p.parent.name.isdigit() else 10**9,
            )
            session_key = f'subject_{sid:04d}_{record["timestamp"]}'
            for path in files:
                if not path.parent.name.isdigit():
                    continue
                file_key = session_key + "_part_" + path.parent.name
                entries.append(
                    {
                        **record,
                        "phase": phase,
                        "session_key": session_key,
                        "file_key": file_key,
                        "path": str(path),
                        "part": int(path.parent.name),
                        "bytes": path.stat().st_size,
                        "identity_conflict": identity_conflict,
                        "session_count": len(records),
                    }
                )
    return entries


def process_one(entry, output):
    """一份文件的完整流程：转换→重新读FIF→质控→287项特征→保存结果。"""
    output = Path(output)
    directory = output / "files" / entry["file_key"]
    directory.mkdir(parents=True, exist_ok=True)
    result_path = directory / "result.json"
    if result_path.is_file():
        previous = json.loads(result_path.read_text(encoding="utf8"))
        if (
            previous.get("pipeline_version") == PIPELINE_VERSION
            and previous.get("source_mtime_ns") == Path(entry["path"]).stat().st_mtime_ns
            and previous.get("source_bytes") == entry["bytes"]
            and previous.get("features_status") == "ok"
            and Path(previous.get("fif_path", "_missing")).is_file()
        ):
            return previous
    result = {
        **entry,
        "pipeline_version": PIPELINE_VERSION,
        "source_bytes": entry["bytes"],
        "source_mtime_ns": Path(entry["path"]).stat().st_mtime_ns,
        "conversion": "not_started",
        "features_status": "not_started",
    }
    converted = None
    needs_copy = False
    try:
        recordinfo_path = Path(entry["folder"]) / "RecordInfo.json"
        recordinfo = (
            decode_metadata(recordinfo_path.read_bytes(), "recordinfo")
            if recordinfo_path.is_file()
            else None
        )
        # 当前电脑对Python写入原始盘有限制：先在本机结果目录转换并核验，
        # 特征也从这个已核验FIF读取。独立PowerShell复制器再按原路径发布。
        source_parent = Path(entry["path"]).parent
        if (source_parent / "fdata_batch_manifest.json").is_file():
            converted = convert_bdf_to_fif(entry["path"], recordinfo)
        else:
            converted = convert_bdf_to_fif(entry["path"], recordinfo, directory / "fif_work")
            needs_copy = True
        result.update(
            conversion="ok",
            fif_path=converted["fif_path"],
            source_sha256=converted["source_sha256"],
            duration_seconds=converted["duration_seconds"],
            groups=converted["groups"],
        )
        # 同时要求：记录配置允许EEG，且符合原网站的头皮通道名称规则。
        # 这避免把PG、A1/A2等未知角色通道因标为EEG就自动纳入模型。
        selected = [
            r["原始通道"]
            for r in converted["mapping"]["rows"]
            if r["MNE类型"] == "eeg" and _is_eeg_like(r["原始通道"], "")
        ]
        data, labels, units, header, _ = read_eeg_file(
            converted["fif_path"], selected_channels=selected
        )
        pre = preprocess_eeg(data, header, SETTINGS)
        channel_table, summary, metadata = extract_collection([pre], SETTINGS)
        summary.to_csv(directory / "features_287.csv", index=False, encoding="utf-8-sig")
        channel_table.to_csv(directory / "channel_features.csv", index=False, encoding="utf-8-sig")
        np.savez_compressed(
            directory / "prepared.npz",
            data=pre["data"],
            fs=pre["fs"],
            source_fs=pre["source_fs"],
            names=np.asarray(pre["channel_names"], dtype=str),
            kept=pre["kept_epoch_indices"],
            epoch_samples=pre["original_epoch_samples"],
            epoch_count=pre["epoch_count"],
            epoch_total=pre["epoch_total"],
        )
        save_json(directory / "feature_definition.json", metadata.get("dictionary", []))
        result.update(
            features_status="ok",
            kept_channels=pre["channel_names"],
            epoch_count=pre["epoch_count"],
            epoch_total=pre["epoch_total"],
            kept_ratio=pre["keep_ratio"],
            effective_seconds=pre["duration_seconds"],
            feature_count=len(summary),
            notes=pre["notes"],
        )
    except Exception as error:
        result["features_status"] = "failed" if result["conversion"] == "ok" else "not_available"
        if result["conversion"] != "ok":
            result["conversion"] = "failed"
        result["reason"] = f"{type(error).__name__}: {error}"
        (directory / "error_private.log").write_text(traceback.format_exc(), encoding="utf8")
    if converted and needs_copy:
        try:
            publish_with_copy_worker(entry, converted, output, directory)
            result["fif_path"] = str(Path(entry["path"]).parent / "fdata.fif")
        except Exception as error:
            result["conversion"] = "failed"
            result["publish_error"] = f"{type(error).__name__}: {error}"
            if result["features_status"] == "ok":
                result["features_status"] = "computed_but_not_published"
    save_json(result_path, result)
    return result


def publish_with_copy_worker(entry, converted, output, directory):
    """交给原生文件复制器，不改目录权限；核对SHA-256后才算转换完成。"""
    queue = output / "copy_queue"
    queue.mkdir(exist_ok=True)
    stage = directory / "fif_work"
    target = Path(entry["path"]).parent
    manifest = stage / "fdata_batch_manifest.json"
    saved = json.loads(manifest.read_text(encoding="utf8"))
    saved["fif_path"] = str(target / "fdata.fif")
    save_json(manifest, saved)
    names = [name for group in converted["groups"] for name in group["files"]] + [
        "fdata_batch_manifest.json"
    ]
    save_json(
        queue / (entry["file_key"] + ".ready.json"),
        {
            "source_bdf": entry["path"],
            "source_sha256": converted["source_sha256"],
            "files": [
                {"source": str((stage / name).resolve()), "target": str(target / name)}
                for name in names
            ],
        },
    )
    done = queue / (entry["file_key"] + ".done.json")
    failed = queue / (entry["file_key"] + ".failed.json")
    start = time.time()
    while not done.is_file():
        if failed.is_file():
            raise RuntimeError(json.loads(failed.read_text(encoding="utf-8-sig"))["error"])
        if time.time() - start > 1200:
            raise TimeoutError("等待同目录复制超时；临时FIF保留，可继续复制。")
        time.sleep(0.5)
    # 复制器已核对内容指纹。只清理当前文件的C盘暂存FIF，保留特征与准备数据。
    for name in names:
        (stage / name).unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description="逐份转换并提取EEG特征（只读源BDF）")
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--limit", type=int, default=0, help="仅调试时处理前N份；0表示全部")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    entries = inventory(args.input_root)
    pd.DataFrame(entries).to_csv(
        args.output / "inventory_private.csv", index=False, encoding="utf-8-sig"
    )
    selected = entries[: args.limit] if args.limit else entries
    (args.output / "files_complete.json").unlink(missing_ok=True)
    save_json(
        args.output / "protocol.json",
        {
            "pipeline_version": PIPELINE_VERSION,
            "input_root": str(args.input_root),
            "files": len(entries),
            "selected_files": len(selected),
            "settings": SETTINGS.__dict__,
            "conversion": "unfiltered native samples, double precision, standard template coordinates, no automatic BAD",
            "raw_source": "read only",
            "feature_input": "read exported FIF; exclude non-EEG by both type and default scalp-name rule",
        },
    )
    start = time.time()
    results = []

    def record(result):
        results.append(result)
        summary = {
            "total": len(selected),
            "completed": len(results),
            "converted": sum(r["conversion"] == "ok" for r in results),
            "features_ok": sum(r["features_status"] == "ok" for r in results),
            "conversion_failed": sum(r["conversion"] == "failed" for r in results),
            "features_failed": sum(r["features_status"] == "failed" for r in results),
            "elapsed_seconds": round(time.time() - start, 1),
            "updated": datetime.now().isoformat(timespec="seconds"),
        }
        save_json(args.output / "progress.json", summary)
        print(json.dumps(summary, ensure_ascii=True), flush=True)

    if args.workers == 1:
        for entry in selected:
            record(process_one(entry, args.output))
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            tasks = [pool.submit(process_one, entry, str(args.output)) for entry in selected]
            for task in as_completed(tasks):
                record(task.result())
    pd.DataFrame(
        [{k: v for k, v in r.items() if not isinstance(v, (list, dict))} for r in results]
    ).sort_values(["id", "timestamp", "part"]).to_csv(
        args.output / "file_status_private.csv", index=False, encoding="utf-8-sig"
    )
    save_json(
        args.output / "files_complete.json",
        {"completed": True, "files": len(results), "seconds": time.time() - start},
    )


if __name__ == "__main__":
    main()
