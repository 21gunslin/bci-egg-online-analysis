"""网站分析主线：先读这里，再沿着导入的模块往下看。

把一次分析想成一条流水线：读取 → 预处理 → 算特征 → 画图和导出。
这里负责安排步骤；具体公式放在各自模块，避免一个大文件包办全部工作。
旧代码仍可从本文件导入原来的函数名，因此网页和测试入口保持兼容。
"""

from __future__ import annotations

import json
import math
import platform
from importlib.metadata import version as package_version
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable
import numpy as np
from eeg_models import (
    SUPPORTED_EXTENSIONS,
    PIPELINE_VERSION,
    PathLike,
    ProgressCallback,
    DEFAULT_BANDS,
    ROI_ALIASES,
    NON_EEG_LABELS,
    FEATURE_DESCRIPTIONS,
    HeaderInfo,
    AnalysisSettings,
    default_progress,
    EEGQualityError,
    NoUsableEpochsError,
)
from eeg_readers import (
    _clean_text,
    _parse_ascii,
    _parse_number,
    _as_float_array,
    _normalize_channel_name,
    _is_annotation_label,
    _is_eeg_like,
    voltage_to_uv,
    _choose_channels,
    _record_duration_to_samples,
    _EDFReader,
    _GDFReader,
    read_eeg_file,
)
from eeg_preprocessing import (
    infer_sampling_rate,
    _estimate_sampling_rate,
    _downsample,
    _bandpass,
    _apply_notches,
    preprocess_eeg,
    channel_qc,
    clean_epochs,
)
from eeg_features import (
    _windowed_features,
    _frequency_features,
    _time_frequency_features,
    plv_matrix,
    graph_metrics,
    _network_features,
    aggregate_features,
    extract_features,
)
from eeg_reporting import (
    _safe_slug,
    save_figures,
    settings_network_bands,
    build_feature_workbook,
)
from eeg_demo import (
    make_example_bdf,
)
from eeg_reporting import make_feature_rows


def analyze_eeg_file(
    input_path: PathLike,
    output_dir: PathLike,
    settings: AnalysisSettings | None = None,
    explicit_fs: float | None = None,
    selected_channels: Iterable[str] | None = None,
    auto_eeg_only: bool = True,
    include_time: bool = True,
    include_frequency: bool = True,
    include_time_frequency: bool = True,
    include_network: bool = True,
    save_preprocessed: bool = False,
    progress: ProgressCallback = default_progress,
) -> dict[str, Any]:
    """一次单份分析的总流程。先读这段函数，把每一步输入、输出连起来，再进入具体模块看公式。"""
    settings = settings or AnalysisSettings()
    drop_prefixes: list[str] = []
    source = Path(input_path).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    slug = _safe_slug(source.stem)

    progress(0.01, "读取脑电文件")
    # 步骤①：把 BDF/EDF/GDF 文件读成数值矩阵（通道×采样点）、通道名、物理单位和文件头。
    data, labels, units, header, annotations = read_eeg_file(
        source,
        selected_channels=selected_channels,
        auto_eeg_only=auto_eeg_only,
        max_channels=settings.max_channels,
    )
    header.channel_labels = labels  # type: ignore[attr-defined]
    header.channel_units = units  # type: ignore[attr-defined]

    # 步骤②：单位换算、重采样到 target_fs、带通/陷波滤波、通道与片段质控（剔除坏段）。
    preprocessed = preprocess_eeg(
        data,
        header,
        settings,
        explicit_fs=explicit_fs,
        selected_channels=None,
        progress=progress,
    )
    provenance = {
        "pipeline_version": PIPELINE_VERSION,
        "python": platform.python_version(),
        "packages": {
            name: package_version(name)
            for name in ("mne", "numpy", "scipy", "pandas", "matplotlib", "openpyxl")
        },
        "settings": asdict(settings),
        "reference": preprocessed["reference"],
        "source_channel_units": preprocessed["source_channel_units"],
        "analysis_voltage_unit": "µV",
        "kept_epoch_indices": preprocessed["kept_epoch_indices"].tolist(),
        "original_epoch_samples": preprocessed["original_epoch_samples"],
        "annotation_policy": "FIF saved bad channels excluded; epochs touching BAD excluded, each unmarked run filtered separately; BDF annotations retained as timed text",
        "boundary_policy": "retained epochs concatenated; exploratory features may contain join effects",
    }
    # 步骤③：计算时域、频域、时频域、网络四类特征，得到通道级明细表和汇总表。
    channel_table, summary_table, metadata = extract_features(preprocessed, settings, progress)

    # 网页上未勾选的特征类别：把对应"前缀"的行从汇总表里删掉（网络类别还会重建 settings 以跳过计算）。
    if not include_time:
        drop_prefixes.append("时域__")
    if not include_frequency:
        drop_prefixes.append("频域__")
    if not include_time_frequency:
        drop_prefixes.append("时频域__")
    if not include_network:
        drop_prefixes.append("网络__")
        settings = AnalysisSettings(
            feature_profile=settings.feature_profile,
            target_fs=settings.target_fs,
            lowpass_hz=settings.lowpass_hz,
            notches=settings.notches,
            epoch_seconds=settings.epoch_seconds,
            amplitude_threshold_uv=settings.amplitude_threshold_uv,
            variance_z_threshold=settings.variance_z_threshold,
            flat_threshold_uv=settings.flat_threshold_uv,
            max_std_uv=settings.max_std_uv,
            duplicate_correlation=settings.duplicate_correlation,
            min_channels=settings.min_channels,
            min_keep_ratio=settings.min_keep_ratio,
            density=settings.density,
            network_bands=(),
            max_network_samples=settings.max_network_samples,
            max_channels=settings.max_channels,
        )
    if drop_prefixes:
        summary_table = summary_table.loc[
            ~summary_table["特征"].astype(str).str.startswith(tuple(drop_prefixes))
        ].reset_index(drop=True)
        metadata["feature_categories"] = {
            category: [
                name
                for name in names
                if not any(name.startswith(prefix) for prefix in drop_prefixes)
            ]
            for category, names in metadata["feature_categories"].items()
        }

    # 步骤④：导出两张 CSV（通道级明细、汇总）。
    feature_csv = output / f"{slug}_特征_通道级.csv"
    summary_csv = output / f"{slug}_特征_汇总.csv"
    channel_table.to_csv(feature_csv, index=False, encoding="utf-8-sig")
    summary_table.to_csv(summary_csv, index=False, encoding="utf-8-sig")

    # 步骤⑤：绘制波形、频谱等图片；步骤⑥：生成汇总 Excel 工作簿。
    profile_files = []
    if settings.feature_profile == "thesis_v1":
        from eeg_thesis_features import plot_profile, export_profile

        metadata["dictionary"] = [
            row for row in metadata["dictionary"] if row["特征"] in set(summary_table["特征"])
        ]
        profile_files = export_profile(summary_table, metadata, output)
        figure_paths = plot_profile([preprocessed], channel_table, metadata, output)
        provenance["boundary_policy"] = (
            "independent continuous runs; pooled valid windows; no spectral/phase computation across joins"
        )
        provenance["feature_definition"] = metadata["definition"]
        preprocessed["notes"] = [note for note in preprocessed["notes"] if "拼接" not in note]
        preprocessed["notes"].extend(
            ["论文方案按独立连续段计算，缺失值不补零；详细口径与覆盖见下载。"]
        )
    else:
        figure_paths = save_figures(preprocessed, channel_table, metadata, output, slug, progress)
    if header.format == "FIF":
        from eeg_fif import review_figures

        figure_paths.extend(Path(p) for p in review_figures(source, output))
    from eeg_montage import montage_mapping, export_montage

    montage_info = montage_mapping(
        preprocessed["raw_channel_names"],
        included=preprocessed["channel_names"],
        scope="单份BDF/EDF实际读取通道；按本次质控结果标记",
    )
    montage_files = export_montage(montage_info, output)
    metadata["montage_info"] = montage_info
    provenance["montage_template"] = {k: v for k, v in montage_info.items() if k != "rows"}
    workbook_path = build_feature_workbook(
        channel_table,
        summary_table,
        header,
        preprocessed,
        metadata,
        output / f"{slug}_脑电特征分析.xlsx",
        settings,
        annotations,
    )

    report_path = output / f"{slug}_分析报告.txt"
    with report_path.open("w", encoding="utf-8") as handle:
        handle.write("脑电特征自动分析报告\n")
        handle.write("=" * 52 + "\n")
        handle.write(
            "方法与软件参数：\n" + json.dumps(provenance, ensure_ascii=False, indent=2) + "\n"
        )
        handle.write(f"输入文件：{source}\n")
        handle.write(f"分析时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        handle.write(f"原始格式/时长：{header.format} / {header.duration_seconds:.2f} s\n")
        handle.write(
            f"原始采样率：{preprocessed['source_fs']:.6g} Hz；分析采样率：{preprocessed['fs']:.6g} Hz\n"
        )
        handle.write(
            f"通道：{len(preprocessed['channel_names'])} / {header.analysis_channel_count or header.signal_count}\n"
        )
        handle.write(
            f"片段保留比例：{preprocessed['keep_ratio']:.2%}（{preprocessed['epoch_count']}/{preprocessed['epoch_total']}）\n"
        )
        handle.write("特征角度：时域、频域、时频域、脑网络、质量控制、汇总\n")
        if preprocessed.get("notes"):
            handle.write("\n预处理提示：\n")
            for note in preprocessed["notes"]:
                handle.write(f"- {note}\n")
        if header.notes:
            handle.write("\n读取提示：\n")
            for note in header.notes:
                handle.write(f"- {note}\n")
        handle.write("\n输出文件：\n")
        for item in [feature_csv, summary_csv, workbook_path, *figure_paths]:
            handle.write(f"- {item.name}\n")

    # 镇痛前后比较(pairForm)会传 save_preprocessed=True：把预处理后的数据缓存成 npz，供 compare_cached() 复用。
    cache_path = None
    if save_preprocessed:
        cache_path = output / "comparison_cache.npz"
        np.savez_compressed(
            cache_path,
            data=preprocessed["data"],
            fs=preprocessed["fs"],
            labels=np.asarray(preprocessed["channel_names"], dtype=str),
            units=np.asarray(preprocessed["channel_units"], dtype=str),
            pipeline_version=PIPELINE_VERSION,
            feature_profile=settings.feature_profile,
            settings_json=json.dumps(asdict(settings)),
            kept_epoch_indices=preprocessed["kept_epoch_indices"],
            original_epoch_samples=preprocessed["original_epoch_samples"],
        )
    progress(1.0, "分析完成")
    # 返回给 _single_analysis()：网页真正用到的是 header（原始采样率/时长）、preprocessed（片段保留数、通道名）和 outputs（下载链接）。
    return {
        "input": str(source),
        "output_dir": str(output),
        "cache_path": str(cache_path) if cache_path else None,
        "pipeline_version": PIPELINE_VERSION,
        "provenance": provenance,
        "header": header.to_dict(),
        "preprocessed": {
            "fs": preprocessed["fs"],
            "source_fs": preprocessed["source_fs"],
            "channels": preprocessed["channel_names"],
            "keep_ratio": preprocessed["keep_ratio"],
            "epoch_count": preprocessed["epoch_count"],
            "epoch_total": preprocessed["epoch_total"],
            "notes": preprocessed["notes"],
        },
        "metadata": {
            "feature_profile": settings.feature_profile,
            "bands": metadata["bands"],
            "feature_counts": {
                key: len(value) for key, value in metadata["feature_categories"].items()
            },
        },
        "outputs": {
            "montage_files": montage_files,
            "profile_files": profile_files,
            "channel_features_csv": str(feature_csv),
            "summary_features_csv": str(summary_csv),
            "workbook": str(workbook_path),
            "report": str(report_path),
            "figures": [str(item) for item in figure_paths],
        },
        "feature_rows": int(len(channel_table)),
        "feature_columns": int(len(channel_table.columns)),
        "summary_rows": int(len(summary_table)),
        "feature_definition": metadata.get("definition"),
        "feature_coverage": metadata.get("coverage"),
        "montage_info": montage_info,
        "research_appendix": metadata.get("research_appendix"),
        "rows": (
            make_feature_rows(summary_table, metadata)
            if settings.feature_profile == "thesis_v1"
            else []
        ),
    }


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser(description="BDF/EDF/GDF 脑电特征提取")
    parser.add_argument("input", help="输入 .bdf/.edf/.gdf 文件")
    parser.add_argument("--out", default="results", help="输出目录")
    parser.add_argument("--fs", type=float, default=None, help="手动采样率（Hz）")
    parser.add_argument("--example", action="store_true", help="生成示例 BDF")
    args = parser.parse_args()

    if args.example:
        target = make_example_bdf(args.input)
        print(target)
    else:
        result = analyze_eeg_file(args.input, args.out, explicit_fs=args.fs)
        print(json.dumps(result, ensure_ascii=False, indent=2))
