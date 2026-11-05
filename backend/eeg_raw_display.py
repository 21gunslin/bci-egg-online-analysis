"""通道显示顺序：额区在上、枕区在下，辅助信号另组。原始文件里的顺序不因此改变。

Display-only channel ordering and adaptive/fixed-axis waveform snapshots."""

import re


def ordered_channels(names, mapping):
    """Anterior to posterior within EEG; auxiliary signals in a separate group.

    Only the display order changes. BDF/FIF channel/sample order stays intact.
    """
    rows = {r["原始通道"]: r for r in mapping["rows"]}
    regions = [
        ("FP", 0, "额极"),
        ("AF", 1, "前额"),
        ("FC", 3, "额中央"),
        ("FT", 3, "额颞"),
        ("F", 2, "额区"),
        ("CP", 6, "中央顶"),
        ("TP", 6, "颞顶"),
        ("C", 4, "中央"),
        ("T", 5, "颞区"),
        ("PO", 8, "顶枕"),
        ("P", 7, "顶区"),
        ("O", 9, "枕区"),
        ("I", 9, "下枕"),
    ]
    result = []
    for index, name in enumerate(names):
        row = rows.get(name, {})
        label = row.get("标准通道") or name
        label = re.sub(r"^(EEG)[ _-]*", "", label.upper()).split("-")[0]
        eeg = row.get("MNE类型") == "eeg" and not re.fullmatch(r"[AM]\d+", label)
        rank, region = (
            (
                next(
                    ((rank, area) for prefix, rank, area in regions if label.startswith(prefix)),
                    (8.5, "位置待核对"),
                )
                if row.get("标准通道")
                else (8.5, "位置待核对")
            )
            if eeg
            else (20, "辅助/参考")
        )
        # Numbered electrodes then midline, stable within each anterior/posterior band.
        number = re.search(r"\d+", label)
        result.append(
            {
                "name": name,
                "region": region,
                "eeg": eeg,
                "standard": row.get("标准通道", ""),
                "_key": (rank, int(number[0]) if number else 0, label, index),
            }
        )
    result.sort(key=lambda r: r["_key"])
    return [{k: v for k, v in r.items() if k != "_key"} for r in result]


def first_window_plots(source_path, output, mapping, bads=(), bad_segments=()):
    """Auto-fit display snapshots plus a separate 100-uV comparison view."""
    from eeg_raw_bdf import NativeBDF, waveform
    from eeg_viewing import viewing_defaults, export_view

    source = NativeBDF(source_path)
    order = ordered_channels([r["name"] for r in source.signals], mapping)
    defaults = viewing_defaults(source_path, mapping)
    groups = [
        ("EEG", [r["name"] for r in order if r["eeg"]]),
        ("AUX", [r["name"] for r in order if not r["eeg"]]),
    ]
    files = []
    for kind, names in groups:
        for page, start in enumerate(range(0, len(names), 64), 1):
            preview = waveform(
                source_path,
                {"seconds": defaults["seconds"], "channels": names[start : start + 64]},
                mapping=mapping,
            )
            prefix = (
                "原始波形首窗_自适应"
                if kind == "EEG" and page == 1
                else f"原始辅助或后续波形_{kind}_{page}"
            )
            files.extend(
                export_view(
                    preview,
                    output,
                    mode="channel",
                    bads=bads,
                    bad_segments=bad_segments,
                    prefix=prefix,
                )
            )
            if kind == "EEG":
                files.extend(
                    export_view(
                        preview,
                        output,
                        mode="fixed",
                        fixed_step=100,
                        bads=bads,
                        bad_segments=bad_segments,
                        prefix=f"原始波形首窗_统一100uV_{page}",
                    )
                )
    return files
