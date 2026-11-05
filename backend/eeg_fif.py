"""FIF读取与已有标记复核：MNE读取数据，保存的坏道/BAD影响后续特征，EOG和眼状态各自解释。

MNE FIF input and original annotation review; no guessed eye labels."""

import html
import re
from pathlib import Path

import numpy as np


def open_fif(path):
    """用MNE打开独立FIF；缺少分卷时拒绝继续，不能把半份数据当成完整记录。"""
    from eeg_mne_runtime import load_mne

    mne = load_mne()
    try:
        raw = mne.io.read_raw_fif(path, preload=False, verbose="ERROR")
    except (FileNotFoundError, OSError) as error:
        raise ValueError(
            "FIF读取失败；分卷FIF需全部配套文件，普通上传入口仅支持独立单文件FIF。"
        ) from error
    if len(raw.filenames) != 1:
        raw.close()
        raise ValueError(
            "当前上传入口仅支持独立单文件FIF，不能只上传分卷的第一卷。请保留全部分卷并先在本机整理为单文件。"
        )
    return raw


def annotation_rows(raw):
    """把MNE注释整理成相对当前文件首样本的时间，保留原始标签与精确时间。"""
    rows = []
    for i, (onset, duration, desc) in enumerate(
        zip(raw.annotations.onset, raw.annotations.duration, raw.annotations.description)
    ):
        extra = dict(raw.annotations.extras[i]) if hasattr(raw.annotations, "extras") else {}
        # Our exports retain original relative seconds; other FIFs follow MNE's first_time convention.
        start = float(extra.get("exact_onset_s", onset)) - raw.first_time
        length = float(extra.get("exact_duration_s", duration))
        if not np.isfinite(start) or not np.isfinite(length) or length < 0:
            raise ValueError("FIF注释时间无效，不能忽略后继续分析。")
        label = str(extra.get("original_description", html.unescape(str(desc))))
        lower = label.lower()
        kind = (
            "BAD"
            if str(desc).lower().startswith("bad") or lower.startswith("bad")
            else (
                "眼状态"
                if lower.startswith("eyes_")
                else "眼动标记" if re.search(r"eog|blink|眼动|眨眼", lower) else "其他"
            )
        )
        channels = list(raw.annotations.ch_names[i]) if hasattr(raw.annotations, "ch_names") else []
        rows.append(
            {
                "start": start,
                "end": start + length,
                "duration": length,
                "label": label,
                "kind": kind,
                "channels": channels,
            }
        )
    return rows


def read_fif(path, max_channels=128):
    """读取FIF的EEG和已有标记，单位统一为µV；已保存坏道不进入特征候选。"""
    from eeg_feature_engine import HeaderInfo, EEGQualityError

    raw = open_fif(path)
    try:
        kinds = raw.get_channel_types()
        bads = list(raw.info["bads"])
        picks = [i for i, kind in enumerate(kinds) if kind == "eeg" and raw.ch_names[i] not in bads]
        if not picks:
            raise EEGQualityError("FIF没有未标为坏道的EEG通道；请核对通道类型与人工标记。")
        if len(picks) > max_channels:
            raise ValueError("FIF有效EEG通道超过配置上限，不能静默截断。")
        fs = float(raw.info["sfreq"])
        rows = annotation_rows(raw)
        date = raw.info.get("meas_date")
        description = raw.info.get("description") or ""
        clock = re.search(
            r"original local clock \(timezone unverified\):\s*(\d{2}\.\d{2}\.\d{2})\s+(\d{2}\.\d{2}\.\d{2})",
            description,
        )
        # Never turn an unverified local clock into UTC. Relative crop offset still applies.
        if clock:
            from eeg_events import recording_start
            from datetime import timedelta

            local = recording_start({"start_date": clock[1], "start_time": clock[2]}) + timedelta(
                seconds=raw.first_time
            )
            day, time = local.strftime("%d.%m.%y"), local.strftime("%H.%M.%S")
        else:
            from datetime import timedelta

            local = date + timedelta(seconds=raw.first_time) if date else None
            day, time = (
                (local.strftime("%d.%m.%y"), local.strftime("%H.%M.%S")) if local else ("", "")
            )
        labels = [raw.ch_names[i] for i in picks]
        header = HeaderInfo(
            file_name=Path(path).name,
            file_path=str(path),
            format="FIF",
            version="MNE",
            patient="",
            recording="",
            start_date=day,
            start_time=time,
            channel_count=len(raw.ch_names),
            signal_count=len(labels),
            data_records=0,
            record_duration=0.0,
            duration_seconds=raw.n_times / fs,
            physical_minimum=0.0,
            physical_maximum=0.0,
            digital_minimum=0,
            digital_maximum=0,
            annotations=len(rows),
            reader="MNE read_raw_fif",
            notes=["FIF EEG按MNE的SI伏特转为µV；已确认坏道不参与特征，BAD注释覆盖的片段剔除。"],
            sampling_rate=fs,
            bad_channels=bads,
            marked_segments=[r for r in rows if r["kind"] == "BAD"],
            annotation_records=rows,
            start_offset_seconds=local.microsecond / 1e6 if local else 0.0,
        )
        return (
            raw.get_data(picks=picks) * 1e6,
            labels,
            ["µV"] * len(labels),
            header,
            [f"{r['start']:g}s / {r['duration']:g}s: {r['label']}" for r in rows],
        )
    finally:
        raw.close()


def preprocess_marked(data, header, settings, source_fs):
    """只在未标记的连续区间内分别滤波/重采样；被BAD覆盖的位置只保留索引，不进入特征。"""
    from eeg_feature_engine import _downsample, _bandpass, _apply_notches, NoUsableEpochsError

    fs = min(source_fs, settings.target_fs)
    duration = data.shape[1] / source_fs
    bad = []
    for row in header.marked_segments:
        a, b = max(0.0, row["start"]), min(duration, row["end"])
        if b > a:
            bad.append((a, b))
        elif 0 <= a < duration:
            bad.append((a, min(duration, a + 1 / source_fs)))
    bad.sort()
    merged = []
    for a, b in bad:
        if merged and a <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(b, merged[-1][1]))
        else:
            merged.append((a, b))
    good = []
    cursor = 0.0
    for a, b in merged:
        if a > cursor:
            good.append((cursor, a))
        cursor = max(cursor, b)
    if cursor < duration:
        good.append((cursor, duration))
    output = np.zeros((data.shape[0], round(duration * fs)))
    valid = np.zeros(output.shape[1], dtype=bool)
    for a, b in good:
        block = data[:, int(np.ceil(a * source_fs)) : int(np.floor(b * source_fs))]
        if block.shape[1] < max(40, int(source_fs * 0.25)):
            continue
        block, actual, _ = _downsample(block, source_fs, settings.target_fs)
        if not np.isclose(actual, fs):
            raise ValueError("FIF重采样率不一致。")
        if block.shape[1] < 40:
            continue
        block = _bandpass(block, fs, 0.5, min(settings.lowpass_hz, 45.0))
        block, _ = _apply_notches(block, fs, settings.notches)
        start = int(np.ceil(a * fs))
        length = min(block.shape[1], int(np.floor(b * fs)) - start, output.shape[1] - start)
        if length > 0:
            output[:, start : start + length] = block[:, :length]
            valid[start : start + length] = True
    if not valid.any():
        raise NoUsableEpochsError("FIF的BAD标记后没有足够长的未标记数据。")
    return (
        output,
        fs,
        valid,
        [
            "FIF按BAD注释分开处理未标记连续段，滤波/重采样不跨BAD；覆盖BAD或过短余段的片段不参与特征。"
        ],
    )


def review_figures(path, output):
    """画FIF原有坏道、坏段、眼状态和EOG的复核图，不自动确认眼动伪迹。"""
    import pandas as pd
    from eeg_pair_spike import plotting

    plt = plotting()
    raw = open_fif(path)
    output = Path(output)
    files = []
    try:
        fs = float(raw.info["sfreq"])
        duration = raw.n_times / fs
        rows = annotation_rows(raw)
        bads = set(raw.info["bads"])

        def save(fig, name):
            target = output / name
            fig.savefig(target, dpi=140)
            plt.close(fig)
            files.append(str(target))

        fig, ax = plt.subplots(
            figsize=(12, max(3, min(16, len(raw.ch_names) * 0.24))), constrained_layout=True
        )
        for i, name in enumerate(raw.ch_names):
            ax.scatter(0, i, color="#c33" if name in bads else "#4685a4", s=35)
            ax.text(
                0.05,
                i,
                name + (" [BAD]" if name in bads else ""),
                color="#c33" if name in bads else "#333",
                va="center",
                fontsize=8,
            )
        ax.invert_yaxis()
        ax.set(
            xlim=(-0.1, 1),
            xticks=[],
            yticks=[],
            title=f"FIF原有坏道标记：{len(bads)} / {len(raw.ch_names)}；红色为已保存BAD",
        )
        save(fig, "FIF复核_坏道标记.png")
        fig, ax = plt.subplots(figsize=(12, 4), constrained_layout=True)
        for i, r in enumerate(rows):
            color = (
                "#c33" if r["kind"] == "BAD" else "#488b73" if r["kind"] == "眼状态" else "#ba8a27"
            )
            ax.broken_barh(
                [(max(0, r["start"]), max(1 / fs, r["duration"]))],
                (i - 0.35, 0.7),
                facecolors=color,
            )
            if r["duration"] == 0:
                ax.scatter([r["start"]], [i], color=color, marker="v", s=35, zorder=3)
            if i < 100:
                ax.text(max(0, r["start"]), i, r["label"][:45], fontsize=7, va="bottom")
        if not rows:
            ax.text(0.5, 0.5, "FIF未保存坏段或其他时间注释", transform=ax.transAxes, ha="center")
        ax.set(
            xlim=(0, max(duration, 1)),
            xlabel="相对本文件首样本时间 (s)",
            ylabel="注释序号",
            title="FIF原始注释时间轴：红=BAD，绿=眼状态，黄=其他/眼动标记",
        )
        save(fig, "FIF复核_坏段与眼状态时间轴.png")
        eog = [i for i, k in enumerate(raw.get_channel_types()) if k == "eog"]
        eye = [r for r in rows if r["kind"] == "眼动标记"]
        focus = max(0, eye[0]["start"] - 2) if eye else 0
        stop = min(duration, focus + 10)
        fig, axes = plt.subplots(
            max(1, len(eog)),
            1,
            figsize=(12, max(3, len(eog) * 2)),
            squeeze=False,
            constrained_layout=True,
        )
        if not eog:
            axes[0, 0].text(
                0.5,
                0.5,
                "没有标为EOG的通道，不能据此识别眼动；EYES_open/closed仅为眼状态",
                ha="center",
                transform=axes[0, 0].transAxes,
            )
        for ax, index in zip(axes[:, 0], eog):
            a = int(focus * fs)
            b = min(raw.n_times, int(stop * fs))
            values = raw.get_data(picks=[index], start=a, stop=b)[0] * 1e6
            step = max(1, len(values) // 6000)
            # extrema preserve transients in long windows
            points = []
            for j in range(0, len(values), step):
                block = values[j : j + step]
                ix = sorted(set([int(np.argmin(block)), int(np.argmax(block))]))
                points.extend((focus + (j + k) / fs, block[k]) for k in ix)
            if points:
                ax.plot(*np.asarray(points).T, lw=0.6)
            for r in eye:
                if r["end"] >= focus and r["start"] <= stop:
                    ax.axvline(r["start"], color="#c33", lw=1)
                    ax.text(
                        r["start"],
                        0.95,
                        r["label"][:25],
                        transform=ax.get_xaxis_transform(),
                        fontsize=7,
                    )
            ax.set(
                xlabel="原始时间 (s)",
                ylabel="µV",
                title=raw.ch_names[index] + " · 原始EOG，未滤波；不自动确认眼动伪迹",
            )
        save(fig, "FIF复核_眼动通道与标记.png")
        path = output / "FIF复核_原始标记.csv"
        pd.DataFrame(
            [
                {
                    "开始_s": r["start"],
                    "结束_s": r["end"],
                    "类别": r["kind"],
                    "原始标签": r["label"],
                    "涉及通道": ",".join(r["channels"]),
                }
                for r in rows
            ],
            columns=["开始_s", "结束_s", "类别", "原始标签", "涉及通道"],
        ).to_csv(path, index=False, encoding="utf-8-sig")
        files.append(str(path))
        return files
    finally:
        raw.close()
