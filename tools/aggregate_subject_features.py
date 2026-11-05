"""第2步：把逐文件特征整理成“每位产妇一行”，准备建立随机森林。

为什么不能直接把664份文件当664个人？同一个人的脑电很相似。
如果她的一份文件用来训练、另一份用来测试，模型可能只记住了这个人。
本程序先合并一次记录的合格片段，再配对镇痛前后，最后才连接PPD标签。
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
from pathlib import Path
import sys

for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

import numpy as np
import pandas as pd

from eeg_thesis_features import extract_collection
from eeg_batch_fif import file_sha256
from run_bdf_fif_features import SETTINGS, save_json


def merge_one_record(rows, output):
    """一次记录可能分成几个data文件；只汇集合格数据，不跨缺口计算。"""
    output = Path(output)
    first = rows[0]
    destination = output / "sessions" / first["session_key"]
    destination.mkdir(parents=True, exist_ok=True)
    result = {
        "id": int(first["id"]),
        "session_key": first["session_key"],
        "timestamp": first["timestamp"],
        "phase": first["phase"],
        "identity_conflict": bool(first["identity_conflict"]),
        "file_total": len(rows),
        "status": "failed",
    }
    try:
        usable = []
        for row in sorted(rows, key=lambda r: r["part"]):
            path = output / "files" / row["file_key"] / "result.json"
            status = json.loads(path.read_text(encoding="utf8"))
            if status["conversion"] == "ok" and status["features_status"] == "ok":
                usable.append(row)
        result["file_usable"] = len(usable)
        if not usable:
            raise ValueError("本次记录没有通过现有质控的data文件")

        # 不填补缺失电极：取本次记录各合格文件共同保留的通道。
        # 这样一次记录的脑区平均和网络始终使用同一组实际电极。
        channel_lists = []
        for row in usable:
            with np.load(output / "files" / row["file_key"] / "prepared.npz") as cached:
                channel_lists.append(cached["names"].tolist())
        common = [
            name for name in channel_lists[0] if all(name in names for names in channel_lists)
        ]
        if len(common) < SETTINGS.min_channels:
            raise ValueError(f"合格文件的共同保留通道只有{len(common)}个，不足3个")

        prepared = []
        for row, names in zip(usable, channel_lists):
            with np.load(output / "files" / row["file_key"] / "prepared.npz") as cached:
                indices = [names.index(name) for name in common]
                prepared.append(
                    {
                        "data": cached["data"][indices].copy(),
                        "fs": float(cached["fs"]),
                        "channel_names": common,
                        "kept_epoch_indices": cached["kept"].copy(),
                        "original_epoch_samples": int(cached["epoch_samples"]),
                    }
                )

        # 不能简单把每个文件的特征再平均：例如谱熵和网络不是线性量。
        # 调用网站相同函数，汇集独立窗口后重新计算；不把断点当连续脑电。
        channels, summary, metadata = extract_collection(prepared, SETTINGS)
        summary.to_csv(destination / "features_287.csv", index=False, encoding="utf-8-sig")
        channels.to_csv(destination / "channel_features.csv", index=False, encoding="utf-8-sig")
        save_json(destination / "feature_dictionary.json", metadata["dictionary"])
        save_json(destination / "coverage.json", metadata["coverage"])
        result.update(
            status="ok",
            channels=common,
            channel_count=len(common),
            feature_count=len(summary),
            clean_seconds=sum(p["data"].shape[1] / p["fs"] for p in prepared),
            usable_file_keys=[row["file_key"] for row in usable],
        )
    except Exception as error:
        result["reason"] = f"{type(error).__name__}: {error}"
    save_json(destination / "result.json", result)
    return result


def read_labels(baseline):
    """只取编号、结局和错误记录检查字段，不把随访信息当预测特征。"""
    table = pd.read_excel(baseline)
    required = ["id", "PPD", "PPD time"]
    if any(name not in table for name in required):
        raise ValueError("基线表缺少id、PPD或PPD time列")
    labels = table[required].apply(pd.to_numeric, errors="coerce")
    labels = labels.dropna(subset=["id"]).copy()  # 表头说明等非患者行不计数。
    if (labels["id"] % 1 != 0).any() or labels["id"].duplicated().any():
        raise ValueError("基线编号不是唯一整数，请先核对，不能自动挑一条")
    labels["id"] = labels["id"].astype(int)
    return labels.set_index("id")


def build_subject_table(inventory, sessions, labels, output):
    """先检查身份与结局，再把pre/post/delta拼成861列。差值=镇痛后−镇痛前。"""
    by_session = {r["session_key"]: r for r in sessions}
    subjects = []
    exclusions = []
    for sid, rows in inventory.groupby("id", sort=True):
        sid = int(sid)
        reason = ""
        keys = rows["session_key"].drop_duplicates().tolist()
        if rows["identity_conflict"].any():
            reason = "同一编号的姓名信息不一致，身份待核对"
        elif len(keys) != 2:
            reason = "记录数不是两次，镇痛前后无法唯一配对"
        elif sid not in labels.index:
            reason = "基线表没有匹配编号"
        elif labels.loc[sid, "PPD"] == 1 and labels.loc[sid, "PPD time"] == 0:
            reason = "PPD=1且PPD time=0，按用户确认删除错误记录"
        elif labels.loc[sid, "PPD"] not in (0, 1):
            reason = "没有有效的二分类PPD结局"
        records = [by_session[key] for key in keys]
        if not reason and ({r["phase"] for r in records} != {"pre", "post"}):
            reason = "镇痛前后时相不明确"
        if not reason and any(r["status"] != "ok" for r in records):
            reason = "至少一个时相无法生成合格记录特征"
        if reason:
            exclusions.append({"id": sid, "reason": reason})
            continue

        phase_values = {}
        for record in records:
            path = Path(output) / "sessions" / record["session_key"] / "features_287.csv"
            feature_table = pd.read_csv(path)
            if feature_table["特征"].duplicated().any() or len(feature_table) != 287:
                raise ValueError(f"编号{sid}的特征列不完整或重复")
            phase_values[record["phase"]] = feature_table.set_index("特征")["数值"]
        before, after = phase_values["pre"], phase_values["post"]
        if before.index.tolist() != after.index.tolist():
            raise ValueError("镇痛前后特征定义或顺序不一致")
        values = {"pre": before, "post": after, "delta": after - before}
        row = {"id": sid, "PPD": int(labels.loc[sid, "PPD"])}
        for phase, features in values.items():
            for feature, value in features.items():
                row[f"{phase}__{feature}"] = value
        subjects.append(row)

    table = pd.DataFrame(subjects)
    if table.empty or table["id"].duplicated().any():
        raise ValueError("没有可用的独立受试者，或编号重复")
    table.to_csv(Path(output) / "subject_features_private.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(exclusions).to_csv(
        Path(output) / "subject_exclusions_private.csv", index=False, encoding="utf-8-sig"
    )
    return table, exclusions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    inventory = pd.read_csv(args.output / "inventory_private.csv")
    complete = json.loads((args.output / "files_complete.json").read_text(encoding="utf8"))
    if not complete.get("completed") or complete["files"] != len(inventory):
        raise ValueError("逐文件任务尚未全部完成，不能把试跑子集当全量结果")
    groups = [part.to_dict("records") for _, part in inventory.groupby("session_key", sort=True)]
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(merge_one_record, group, args.output) for group in groups]
        for future in as_completed(futures):
            results.append(future.result())
            if len(results) % 20 == 0:
                print(f"已汇总记录：{len(results)}/{len(groups)}", flush=True)
    pd.DataFrame(results).to_csv(
        args.output / "session_status_private.csv", index=False, encoding="utf-8-sig"
    )
    baseline_hash = file_sha256(args.baseline)
    labels = read_labels(args.baseline)
    if file_sha256(args.baseline) != baseline_hash:
        raise ValueError("读取期间基线表发生变化，请重新汇总")
    table, exclusions = build_subject_table(inventory, results, labels, args.output)
    matched_labels = labels.reindex(inventory["id"].drop_duplicates().astype(int))
    # 覆盖情况单独保存给人工复核，不偷偷放进模型预测列。
    coverage = pd.DataFrame(results)
    coverage = coverage[coverage["id"].isin(table["id"])]
    coverage[["id", "phase", "clean_seconds", "channel_count", "file_total", "file_usable"]].to_csv(
        args.output / "subject_record_coverage_private.csv", index=False, encoding="utf-8-sig"
    )
    features = table.drop(columns=["id", "PPD"]).replace([np.inf, -np.inf], np.nan)
    pd.DataFrame(
        {
            "feature": features.columns,
            "missing_fraction": features.isna().mean().values,
            "finite_people": features.notna().sum().values,
        }
    ).to_csv(args.output / "subject_feature_missingness.csv", index=False, encoding="utf-8-sig")
    summary = {
        "input_subjects": int(inventory["id"].nunique()),
        "baseline_sha256": baseline_hash,
        "input_sessions": len(groups),
        "sessions_ok": sum(r["status"] == "ok" for r in results),
        "included_subjects": len(table),
        "PPD_positive": int(table["PPD"].sum()),
        "PPD_negative": int((table["PPD"] == 0).sum()),
        "input_PPD_positive_before_exclusions": int((matched_labels["PPD"] == 1).sum()),
        "input_PPD_negative_before_exclusions": int((matched_labels["PPD"] == 0).sum()),
        "feature_columns": len(table.columns) - 2,
        "all_missing_feature_columns": int(features.isna().all().sum()),
        "excluded_subjects": len(exclusions),
        "exclusion_counts": pd.Series([r["reason"] for r in exclusions]).value_counts().to_dict(),
        "session_merge": "all QC-passed files, common retained channels within each session, independent runs, no artificial continuity",
        "phase_interpretation": "folder clock order; analgesia pre/post interpretation previously confirmed by user",
        "duration_matching": "none; use all available clean data per phase, report unequal coverage",
    }
    save_json(args.output / "cohort_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
