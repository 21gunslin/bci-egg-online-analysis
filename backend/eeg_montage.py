"""电极装配：把已确认的名称对应到标准10–20模板，未知位置留空。模板不等于个体实测坐标。

Explicit template geometry only: no rereferencing, interpolation or feature changes."""

from collections import Counter
from functools import lru_cache
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

NOTICE = (
    "坐标来自MNE标准10–20模板（含扩展点位），不是产妇个体实测电极位置。"
    "映射只装配名称、类型和模板几何，不更改采集参考、波形、质控或特征数值，不用于脑源定位。"
    "T1/T2、PG1/PG2等未确认名称不强配；A1/A2仅列为耳部候选，不据此确定采集参考。"
)


@lru_cache(maxsize=1)
def template():
    path = Path(__file__).parent / "data/montage_standard_1020_head.json"
    raw = path.read_bytes()
    value = json.loads(raw)
    if value["coordinate_frame"] != "head" or value["unit"] != "m" or value["individual_measured"]:
        raise ValueError("Montage模板的坐标系或单位声明无效。")
    for position in value["positions"].values():
        if np.asarray(position).shape != (3,) or not np.isfinite(position).all():
            raise ValueError("Montage模板存在无效坐标。")
    value["sha256"] = hashlib.sha256(raw).hexdigest()
    return value


def region(name):
    if name.startswith(("A", "M")) and name[1:].isdigit():
        return "耳部/乳突命名（采集角色待核实）"
    for prefixes, area in [
        (("Fp", "AF"), "额极/前额区"),
        (("FC",), "额中央区"),
        (("FT",), "额颞区"),
        (("CP",), "中央顶区"),
        (("TP",), "颞顶区"),
        (("PO",), "顶枕区"),
        (("F",), "额区"),
        (("C",), "中央区"),
        (("P",), "顶区"),
        (("T",), "颞区"),
        (("O",), "枕区"),
    ]:
        if name.startswith(prefixes):
            return area
    return "待核对"


def montage_mapping(labels=None, units=None, recordinfo=None, included=None, scope="当前通道"):
    from eeg_feature_engine import _normalize_channel_name, _is_eeg_like

    model = template()
    labels = list(labels if labels is not None else model["default_channels"])
    if len(labels) > 1024:
        raise ValueError("Montage映射通道数超过1024。")
    if units is not None and len(units) != len(labels):
        raise ValueError("Montage通道名和单位长度不一致。")
    names = {name.upper(): name for name in model["positions"]}
    configured = {
        _normalize_channel_name(row["通道名称"]): row["配置类型"]
        for row in (recordinfo or {}).get("channels", [])
    }
    retained = set(included) if included is not None else None
    rows = []
    for index, label in enumerate(labels):
        if not isinstance(label, str) or not label.strip() or len(label) > 128:
            raise ValueError("Montage通道名称无效。")
        key = _normalize_channel_name(label)
        source_kind = configured.get(key)
        if source_kind is not None:
            kind = {
                "EEG": "eeg",
                "EOG": "eog",
                "ECG": "ecg",
                "EMG": "emg",
                "STIM": "stim",
                "RESP": "resp",
                "MISC": "misc",
            }.get(source_kind, "misc")
            basis = "RecordInfo配置；用途仍需采集资料确认"
        else:
            kind = next(
                (
                    kind
                    for prefix, kind in [
                        ("STATUS", "stim"),
                        ("TRIGGER", "stim"),
                        ("MARKER", "stim"),
                        ("EOG", "eog"),
                        ("ECG", "ecg"),
                        ("EKG", "ecg"),
                        ("EMG", "emg"),
                    ]
                    if key.startswith(prefix)
                ),
                None,
            )
            if kind is None:
                kind = (
                    "eeg"
                    if key in names or key in model["aliases"] or _is_eeg_like(label, "")
                    else "misc"
                )
            basis = "按通道标签识别的候选类型；未核实采集用途"
        modern = model["aliases"].get(key, key)
        standard = names.get(modern.upper())
        coordinate = model["positions"].get(standard) if standard and kind == "eeg" else None
        ear = bool(standard and standard.startswith(("A", "M")) and standard[1:].isdigit())
        aliases = key in model["aliases"]
        explanation = "标准名称匹配" if coordinate is not None else "标准模板无可确认点位，坐标留空"
        if coordinate is not None and aliases:
            explanation = f"旧名{key}→{standard}；已验证模板点位重合"
        if coordinate is None and kind != "eeg":
            explanation = "非EEG/类型未确认，不装配头皮坐标"
        selected = label in retained if retained is not None else None
        state = (
            ("本次保留" if selected else "未纳入本次汇总")
            if retained is not None
            else "尚未质控；仅核对模板映射"
        )
        note = "不自动重参考；不改变波形或分析通道"
        if key in ("T1", "T2", "PG1", "PG2"):
            note = "需厂家/采集导联定义；不自动套用MNE其他别名规则"
        if ear:
            note = "耳部/乳突命名候选，角色未核实；不自动设为参考或纳入现有头皮特征"
        rows.append(
            {
                "原始通道": label,
                "标准通道": standard if coordinate is not None else "",
                "MNE类型": kind,
                "类型依据": basis,
                "头皮区域": region(standard) if coordinate is not None else "待核对",
                "映射状态": "已映射" if coordinate is not None else "待核对/不适用",
                "匹配说明": explanation,
                "x_m": float(coordinate[0]) if coordinate is not None else None,
                "y_m": float(coordinate[1]) if coordinate is not None else None,
                "z_m": float(coordinate[2]) if coordinate is not None else None,
                "坐标系": "head" if coordinate is not None else "",
                "坐标来源": "MNE模板，非个体实测" if coordinate is not None else "",
                "分析状态": state,
                "可装配": bool(coordinate is not None),
                "说明": note,
            }
        )

    # Two independently named channels cannot silently occupy one standard point.
    def coordinate_key(row):
        return tuple(round(row[key], 12) for key in ("x_m", "y_m", "z_m"))

    positions = Counter(coordinate_key(row) for row in rows if row["可装配"])
    for row in rows:
        if row["可装配"] and positions[coordinate_key(row)] > 1:
            row["可装配"] = False
            row["映射状态"] = "重复点位，需人工核对"
            row["说明"] += "；多个通道指向同一标准点位，禁止自动装配"
    return mapping_summary(rows, scope)


def mapping_summary(rows, scope):
    model = template()
    return {
        "scope": scope,
        "notice": NOTICE,
        "template_requested": model["requested_kind"],
        "template_resolved": model["resolved_kind"],
        "mne_version": model["mne_version"],
        "coordinate_frame": "head",
        "unit": "m",
        "axes": "x向右、y向前、z向上；不等同MNI坐标",
        "source_url": model["source_url"],
        "license": "MNE-Python BSD-3-Clause；完整许可随模板来源说明导出",
        "sha256": model["sha256"],
        "rows": rows,
        "counts": {
            "total": len(rows),
            "mapped": sum(row["可装配"] for row in rows),
            "unresolved": sum(not row["可装配"] for row in rows),
        },
    }


def catalogue_mapping():
    path = Path(__file__).parent / "data/recordinfo_channel_catalog.json"
    catalogue = json.loads(path.read_text(encoding="utf8"))
    return montage_mapping(
        [row["通道名称"] for row in catalogue["channels"]],
        recordinfo=catalogue,
        scope="当前数据RecordInfo中出现过的通道名称/类型目录；不是一次采集或个体实测点位",
    )


def export_montage(info, output, prefix="Montage"):
    from eeg_pair_spike import plotting

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    csv = output / f"{prefix}_通道映射表.csv"
    pd.DataFrame(info["rows"]).to_csv(csv, index=False, encoding="utf-8-sig")
    note = output / f"{prefix}_模板来源与装配说明.txt"
    license_text = (Path(__file__).parent / "data/LICENSE_MNE.txt").read_text(encoding="utf8")
    note.write_text(
        info["notice"]
        + "\n"
        + json.dumps({k: v for k, v in info.items() if k != "rows"}, ensure_ascii=False, indent=2)
        + "\n\nMNE许可：\n"
        + license_text,
        encoding="utf8",
    )
    files = [str(csv), str(note)]
    unique = {}
    for row in info["rows"]:
        if not row["可装配"]:
            continue
        previous = unique.get(row["标准通道"])
        if previous is None or row["分析状态"] == "本次保留":
            unique[row["标准通道"]] = row
    if unique:
        plt = plotting()
        fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
        for ax, (a, b, title, xlabel, ylabel) in zip(
            axes,
            [
                (0, 1, "俯视：前方朝上", "右向 x (m)", "前向 y (m)"),
                (0, 2, "前视：上方朝上", "右向 x (m)", "上向 z (m)"),
                (1, 2, "侧视坐标投影", "前向 y (m)", "上向 z (m)"),
            ],
        ):
            for standard, row in unique.items():
                xyz = [row["x_m"], row["y_m"], row["z_m"]]
                color = (
                    "#c43a3a"
                    if row.get("人工坏道")
                    else (
                        "#248765"
                        if row["分析状态"] == "本次保留"
                        else "#a36a2a" if "尚未" in row["分析状态"] else "#8b95a2"
                    )
                )
                ax.scatter(xyz[a], xyz[b], s=28, color=color)
                ax.annotate(
                    standard,
                    (xyz[a], xyz[b]),
                    fontsize=7,
                    xytext=(3, 3),
                    textcoords="offset points",
                )
            ax.axhline(0, color="#cbd3de", lw=0.6)
            ax.axvline(0, color="#cbd3de", lw=0.6)
            ax.set(title=title, xlabel=xlabel, ylabel=ylabel, aspect="equal")
            ax.grid(alpha=0.15)
        legend = "绿色=本次保留；棕色=尚未质控；灰色=未纳入汇总。未确认通道不绘制。"
        if any(row.get("人工坏道") for row in info["rows"]):
            legend += "红色=人工坏道（保留样本）。"
        fig.suptitle("MNE标准10–20模板点位核对（非个体实测）\n" + legend, fontsize=11)
        image = output / f"{prefix}_模板电极位置示意.png"
        fig.savefig(image, dpi=140, bbox_inches="tight")
        plt.close(fig)
        files.append(str(image))
    return files


def attach_session_montage(result, inputs, recordinfo, output):
    included_numbers = {row["顺序"] for row in result.get("segments", [])}
    rows = []
    for item in inputs:
        retained = result["common_labels"] if item["number"] in included_numbers else []
        mapping = montage_mapping(
            item["labels"], item["units"], recordinfo, retained, f"第{item['number']}段"
        )
        rows.extend({"记录顺序": item["number"], **row} for row in mapping["rows"])
    info = mapping_summary(rows, "多段逐文件映射；仅共同质控保留通道参与本次汇总")
    result["montage_info"] = info
    result["files"].extend(export_montage(info, output, "Montage_多段"))
    return result


def make_mne_montage(labels, recordinfo=None, *, allow_partial=False):
    """Return (DigMontage, mapping). Unknown EEG points require explicit partial mode.

    Keys are ORIGINAL channel names, preserving signal order and FAA bookkeeping.
    This function does not call set_eeg_reference, interpolate_bads or modify Raw.
    """
    import mne

    info = montage_mapping(labels, recordinfo=recordinfo)
    unresolved = [
        row["原始通道"] for row in info["rows"] if row["MNE类型"] == "eeg" and not row["可装配"]
    ]
    if unresolved and not allow_partial:
        raise ValueError("以下EEG通道未确认或点位重复，不能静默装配：" + ", ".join(unresolved))
    positions = {
        row["原始通道"]: [row["x_m"], row["y_m"], row["z_m"]]
        for row in info["rows"]
        if row["可装配"]
    }
    if not positions:
        raise ValueError("没有可装配的已确认模板点位。")
    fid = template()["fiducials"]
    montage = mne.channels.make_dig_montage(
        ch_pos=positions, nasion=fid["nasion"], lpa=fid["lpa"], rpa=fid["rpa"], coord_frame="head"
    )
    return montage, info
