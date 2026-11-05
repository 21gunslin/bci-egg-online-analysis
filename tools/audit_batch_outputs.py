"""批处理结束后复核：原BDF是否变化、FIF是否齐全、复制核验是否完成。"""

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
from eeg_batch_fif import file_sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    inventory = pd.read_csv(args.output / "inventory_private.csv")
    results = []
    issues = []
    fif_bytes = 0
    fif_files = 0
    max_roundtrip_error = 0.0
    hashed = 0
    for _, row in inventory.iterrows():
        result = json.loads(
            (args.output / "files" / row["file_key"] / "result.json").read_text(encoding="utf8")
        )
        results.append(result)
        source = Path(row["path"])
        stat = source.stat()
        if stat.st_size != result["source_bytes"] or stat.st_mtime_ns != result["source_mtime_ns"]:
            issues.append({"file_key": row["file_key"], "issue": "源BDF大小或修改时间变化"})
        if result["conversion"] != "ok":
            continue
        if file_sha256(source) != result["source_sha256"]:
            issues.append({"file_key": row["file_key"], "issue": "源BDF SHA-256变化"})
        hashed += 1
        manifest_path = source.parent / "fdata_batch_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf8"))
        ack_path = args.output / "copy_queue" / (row["file_key"] + ".done.json")
        if not ack_path.is_file() or not json.loads(ack_path.read_text(encoding="utf-8-sig")).get(
            "ok"
        ):
            issues.append({"file_key": row["file_key"], "issue": "缺少复制后内容一致的核验记录"})
        for group in manifest["groups"]:
            max_roundtrip_error = max(max_roundtrip_error, group["max_roundtrip_error"])
            for name in group["files"]:
                target = source.parent / name
                if not target.is_file():
                    issues.append({"file_key": row["file_key"], "issue": "缺少FIF派生文件"})
                else:
                    fif_files += 1
                    fif_bytes += target.stat().st_size
        if hashed % 100 == 0:
            print(f"已复核源文件指纹：{hashed}", flush=True)
    summary = {
        "files_in_inventory": len(inventory),
        "converted": sum(r["conversion"] == "ok" for r in results),
        "features_ok": sum(r["features_status"] == "ok" for r in results),
        "conversion_failed": sum(r["conversion"] == "failed" for r in results),
        "source_stat_checks": len(inventory),
        "source_sha256_rechecks": hashed,
        "fif_files": fif_files,
        "fif_bytes": fif_bytes,
        "max_fif_roundtrip_error": max_roundtrip_error,
        "issue_count": len(issues),
        "issues": issues,
        "interpretation": "每个转换均已逐样本读回核验，复制时另做FIF SHA-256核对；本次审计重新核对源BDF内容指纹和目标文件存在性",
    }
    (args.output / "batch_audit.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    if issues:
        raise SystemExit("批处理核验发现问题，请检查batch_audit.json")


if __name__ == "__main__":
    main()
