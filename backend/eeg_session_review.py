"""质控失败段的复核展示：保留原始位置和波形，明确它没有进入特征汇总。

Display failed EEG segments on their original clock, never in feature estimates."""

from collections import Counter
from decimal import Decimal
from pathlib import Path
import numpy as np
import pandas as pd

from eeg_events import align_events, read_event_file, recording_start
from eeg_feature_engine import voltage_to_uv, PIPELINE_VERSION
from eeg_pair_spike import plotting


def preview_waveform(data, header):
    """Peak-preserving display samples; these arrays are not analysis inputs."""
    fs = float(header.sampling_rate)
    labels = list(header.channel_labels or [])[:3]
    units = list(header.channel_units or [])[: len(labels)]
    data = voltage_to_uv(np.asarray(data[: len(labels)]), units)
    lines = []
    for x in data:
        step = max(1, int(np.ceil(len(x) / 3000)))
        indices = []
        for left in range(0, len(x), step):
            chunk = x[left : left + step]
            indices.extend(sorted({left + int(chunk.argmin()), left + int(chunk.argmax())}))
        indices = np.asarray(indices, dtype=int)
        lines.append((indices / fs, x[indices].copy()))
    return {"labels": labels, "lines": lines, "original_samples": data.shape[1], "source_fs": fs}


def build_review(headers, skipped, previews, evt_path, output, evt_offset):
    output = Path(output)
    failed = {r["顺序"]: r for r in skipped}
    starts = [(recording_start(h) - recording_start(headers[0])).total_seconds() for h in headers]
    rows = []
    for i, (header, start) in enumerate(zip(headers, starts), 1):
        excluded = i in failed
        rows.append(
            {
                "顺序": i,
                "标签": f"第{i}段",
                "文件头开始时间": header["start_date"] + " " + header["start_time"],
                "距首段起点_s": start,
                "原始时长_s": header["duration_seconds"],
                "原始采样率_Hz": header["sampling_rate"],
                "显示状态": "质控失败，仅供复核" if excluded else "质控通过，参与计算",
                "参与特征汇总": "否" if excluded else "是",
                "失败原因": failed[i]["原因"] if excluded else "",
                "有效时长_s": 0 if excluded else None,
                "有效拼接起点_s": None,
                "保留片段数": None,
                "总片段数": None,
                "共同通道数": 0 if excluded else None,
                "与前段间隔_s": (
                    0 if i == 1 else start - starts[i - 2] - headers[i - 2]["duration_seconds"]
                ),
            }
        )
    parsed = read_event_file(evt_path)
    failed_matches = {}
    for number in failed:
        header = headers[number - 1]
        aligned, _ = align_events(parsed, header, offset_seconds=evt_offset)
        for event_index, event in enumerate(aligned):
            if event["本段关系"] == "本段外":
                continue
            failed_matches.setdefault(event_index, []).append(
                {
                    "事件序号": event_index + 1,
                    "原始标签": event["原始标签"],
                    "evt原始起点_s": event["evt原始起点_s"],
                    "持续时间_s": event["持续时间_s"],
                    "距首段起点_s": event["相对本段起点_s"] + starts[number - 1],
                    "所属分段": number,
                    "分段标签": f"第{number}段",
                    "段内起点_s": event["相对本段起点_s"],
                    "段内原始样本索引_0基": event["原始样本索引_0基"],
                    "状态": event["本段关系"] + "；质控失败，仅供复核；不参与特征汇总",
                    "有效拼接时间_s": None,
                }
            )
    shift = (recording_start(parsed) - recording_start(headers[0])).total_seconds() + float(
        evt_offset
    )
    files = []

    def csv(frame, name):
        path = output / name
        frame.to_csv(path, index=False, encoding="utf-8-sig")
        files.append(str(path))

    csv(pd.DataFrame(rows), "多份_全部分段状态与原始时间.csv")
    plt = plotting()
    fig, ax = plt.subplots(figsize=(13, max(4, 0.45 * len(rows) + 1.8)), constrained_layout=True)
    for i, row in enumerate(rows):
        bad = row["顺序"] in failed
        ax.broken_barh(
            [(row["距首段起点_s"], row["原始时长_s"])],
            (i - 0.3, 0.6),
            facecolors="#c24d49" if bad else "#3c8f72",
            edgecolors="#8e302f" if bad else "#2b6955",
            hatch="///" if bad else None,
            alpha=0.8,
        )
    for event in parsed["events"]:
        ax.axvline(event["onset"] + shift, color="#366baf", alpha=0.35, lw=0.8)
    ax.set_yticks(
        range(len(rows)),
        [
            f"{r['标签']} · {'失败/仅复核' if r['顺序'] in failed else '通过/参与计算'}"
            for r in rows
        ],
    )
    ax.set(
        xlabel="距首段起点的原始时间 (s)，蓝线为evt；红色斜线为失败段",
        title="全部分段：原始时间位置与质控状态",
    )
    span = starts[-1] + headers[-1]["duration_seconds"]
    margin = max(0.5, span * 0.02)
    ax.set_xlim(-margin, span + margin)
    ax.grid(axis="x", alpha=0.2)
    path = output / "多份_全部分段质控状态与evt.png"
    fig.savefig(path, dpi=145, bbox_inches="tight")
    plt.close(fig)
    files.append(str(path))
    for number, preview in previews.items():
        count = len(preview["lines"])
        fig, axes = plt.subplots(
            max(1, count),
            1,
            figsize=(13, 2.1 * max(1, count) + 1.2),
            sharex=True,
            squeeze=False,
            constrained_layout=True,
        )
        for ax, (label, (time, x)) in zip(axes[:, 0], zip(preview["labels"], preview["lines"])):
            ax.plot(time + starts[number - 1], x, color="#ac3937", lw=0.65)
            ax.set(ylabel=label + " (µV)")
            ax.grid(alpha=0.2)
        axes[-1, 0].set_xlabel(
            "距首段起点的原始时间 (s)；未滤波；按文件头单位换算；显示抽稀保留每桶极值"
        )
        fig.suptitle(
            f"第{number}段 · 质控失败，仅供复核，不参与特征汇总\n" + failed[number]["原因"],
            fontsize=10,
            wrap=True,
        )
        path = output / f"复核_第{number}段_失败数据原始波形.png"
        fig.savefig(path, dpi=145, bbox_inches="tight")
        plt.close(fig)
        files.append(str(path))
    return {
        "rows": rows,
        "files": files,
        "parsed": parsed,
        "failed_matches": failed_matches,
        "shift": shift,
        "span": starts[-1] + headers[-1]["duration_seconds"],
    }


def review_only_result(review, headers, skipped, output, settings, metadata_info=None):
    parsed = review["parsed"]
    rows = []
    related = set()
    for index, event in enumerate(parsed["events"]):
        if index in review["failed_matches"]:
            related.add(index)
            rows.extend(review["failed_matches"][index])
            continue
        time = float(Decimal(str(event["onset"])) + Decimal(str(review["shift"])))
        rows.append(
            {
                "事件序号": index + 1,
                "原始标签": event["label"],
                "evt原始起点_s": event["onset"],
                "持续时间_s": event["duration"],
                "距首段起点_s": time,
                "所属分段": None,
                "分段标签": "",
                "段内起点_s": None,
                "段内原始样本索引_0基": None,
                "状态": (
                    "记录间隙（未采集/未选择）" if 0 <= time < review["span"] else "所选时间范围外"
                ),
                "有效拼接时间_s": None,
            }
        )
    files = list(review["files"])
    path = Path(output) / "多份_仅复核_evt事件映射.csv"
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")
    files.append(str(path))
    warning = "所有所选文件均未通过质控：当前仅显示原始时间、失败原因、原始波形及evt，不生成EEG特征或抑郁风险结论。"
    note = Path(output) / "多份_仅复核说明.txt"
    import json

    note.write_text(
        warning
        + "\n"
        + json.dumps(
            {
                "pipeline_version": PIPELINE_VERSION,
                "settings": settings.__dict__,
                "segments": review["rows"],
                "waveform_policy": "unfiltered original voltage; first 3 candidate EEG channels; peak-preserving display only",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf8",
    )
    files.append(str(note))
    return {
        "kind": "session",
        "review_only": True,
        "metrics": {
            "files": 0,
            "selected_files": len(headers),
            "skipped_files": len(skipped),
            "common_channels": 0,
            "sampling_rate": None,
            "recorded_seconds": sum(h["duration_seconds"] for h in headers),
            "span_seconds": review["span"],
            "effective_seconds": 0,
            "continuous_runs": 0,
        },
        "skipped_files": skipped,
        "segments": [],
        "review_segments": review["rows"],
        "common_labels": [],
        "rows": [],
        "table_truncated": False,
        "metadata_info": metadata_info,
        "warnings": [warning],
        "files": files,
        "event_info": {
            "total_events": len(parsed["events"]),
            "in_segment": len(related),
            "outside_segment": len(parsed["events"]) - len(related),
            "alignment_shift_seconds": review["shift"],
            "rows": rows[:1000],
            "table_truncated": len(rows) > 1000,
            "labels": [
                {"原始标签": k, "次数": v}
                for k, v in Counter(e["label"] for e in parsed["events"]).items()
            ],
            "warnings": ["失败段事件只保留在原始时间轴；有效拼接时间为空。"],
            "files": [],
        },
    }
