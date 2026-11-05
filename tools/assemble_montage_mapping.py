"""Create an identity-free channel montage catalogue from RecordInfo files.

No waveforms are loaded; source files are read-only. Output includes a CSV and
template point diagram. Counts are configuration occurrences, not participant N.
"""

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
from eeg_metadata import decode_metadata
from eeg_montage import montage_mapping, mapping_summary, export_montage


def assemble(source, output):
    source = Path(source)
    paths = [source] if source.is_file() else sorted(source.glob("*/RecordInfo.json"))
    if source.is_dir() and (source / "RecordInfo.json").is_file():
        paths = [source / "RecordInfo.json"]
    if not paths:
        raise ValueError("没有找到RecordInfo.json。")
    counts = Counter()
    for path in paths:
        safe = decode_metadata(path.read_bytes(), "recordinfo")
        counts.update((row["通道名称"], row["配置类型"]) for row in safe["channels"])
    rows = []
    for (name, kind), count in sorted(counts.items()):
        single = {"channels": [{"通道名称": name, "配置类型": kind}]}
        row = montage_mapping([name], recordinfo=single)["rows"][0]
        rows.append({**row, "配置出现次数": count})
    info = mapping_summary(
        rows, "RecordInfo通道名称与类型汇集；各行是映射目录，不把不同记录混作一次采集"
    )
    files = export_montage(info, output, "RecordInfo汇集_Montage")
    summary = {
        "source_recordinfo_files": len(paths),
        "unique_name_type_pairs": len(rows),
        "mappable_entries": info["counts"]["mapped"],
        "unresolved_entries": info["counts"]["unresolved"],
        "template": info["template_resolved"],
        "mne_version": info["mne_version"],
        "coordinate_frame": info["coordinate_frame"],
        "unit": info["unit"],
        "unresolved_names": [r["原始通道"] for r in rows if not r["可装配"]],
        "privacy": "No source paths, identities, camera fields or EEG samples exported.",
    }
    path = Path(output) / "Montage装配核对摘要.txt"
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf8")
    return info, files + [str(path)], summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", help="RecordInfo文件、单次记录目录，或包含记录的上层目录")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    _, _, summary = assemble(args.source, args.output)
    print(json.dumps(summary, ensure_ascii=False))
