"""第5站：把已有计算结果整理成图、Excel和文字说明。

绘图只是展示，不能在这一站偷偷重新滤波、删数据或改变单位。"""

from __future__ import annotations

from eeg_mne_ops import welch_psd

import json
import math
import re
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
from scipy import signal
from eeg_models import (
    AnalysisSettings,
    HeaderInfo,
    DEFAULT_BANDS,
    FEATURE_DESCRIPTIONS,
    ProgressCallback,
    default_progress,
)
from eeg_readers import _clean_text
from eeg_features import plv_matrix


def make_feature_rows(summary_table, metadata):
    """给每个特征数值配上说明，供网页的论文附录展示。

    用普通循环代替多层嵌套推导式：先查字典，再把名称、数值、单位放在一行。
    这一步只整理结果，不重新计算任何特征。
    """
    definitions = {item["特征"]: item for item in metadata.get("dictionary", [])}
    rows = []
    for item in summary_table.to_dict("records"):
        feature_name = item["特征"]
        row = {"特征": feature_name, "数值": item["数值"]}
        for key, value in definitions.get(feature_name, {}).items():
            if key != "特征":
                row[key] = value
        rows.append(row)
    return rows


def _safe_slug(value: str) -> str:
    cleaned = re.sub(r"[^\w\u4e00-\u9fff-]+", "_", _clean_text(value)).strip("_")
    return cleaned or "eeg"


def save_figures(
    preprocessed: dict[str, Any],
    channel_table: pd.DataFrame,
    metadata: dict[str, Any],
    output_dir: PathLike,
    slug: str,
    progress: ProgressCallback = default_progress,
) -> list[Path]:
    """把已得到的波形和特征画成图片；图像用于复核和展示，不能直接当作抑郁风险概率。"""
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import font_manager, rcParams
    import matplotlib.pyplot as plt

    for font_family in ("Microsoft YaHei UI", "Microsoft YaHei", "SimHei", "Noto Sans CJK SC"):
        try:
            font_manager.findfont(font_family, fallback_to_default=False)
        except ValueError:
            continue
        rcParams["font.sans-serif"] = [font_family]
        break
    rcParams["axes.unicode_minus"] = False

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    figure_paths: list[Path] = []

    data = np.asarray(preprocessed["data"])
    fs = float(preprocessed["fs"])
    labels = list(preprocessed["channel_names"])
    bands = metadata["bands"]

    progress(0.96, "绘制波形与通道质控图")
    fig, axes = plt.subplots(2, 1, figsize=(13, 9), constrained_layout=True)
    duration = min(10.0, data.shape[1] / fs)
    samples = int(duration * fs)
    time_axis = np.arange(samples) / fs
    for index, label in enumerate(labels):
        offset = index * (np.std(data[index, :samples]) * 6 + 1e-9)
        axes[0].plot(time_axis, data[index, :samples] - offset, linewidth=0.7)
    axes[0].set_title("预处理后波形（前 10 秒，垂直偏移显示）")
    axes[0].set_xlabel("有效片段拼接时间 (s)，非原始连续时间")
    axes[0].set_ylabel("通道")
    axes[0].grid(alpha=0.25)

    sorted_table = channel_table.sort_values("时域__std")
    axes[1].barh(sorted_table["通道"], sorted_table["时域__std"], color="#4f8cc9")
    axes[1].set_title("通道标准差（自动质控保留通道）")
    axes[1].set_xlabel("标准差")
    axes[1].grid(axis="x", alpha=0.25)
    waveform_path = output / f"{slug}_波形与通道质控.png"
    fig.savefig(waveform_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    figure_paths.append(waveform_path)

    progress(0.97, "绘制功率谱与频段特征")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    nperseg = int(min(1024, 2 ** math.floor(math.log2(max(16, data.shape[1])))))
    frequencies, psd = welch_psd(data, fs=fs, nperseg=nperseg, axis=1)
    mean_psd = np.mean(psd, axis=0)
    axes[0].semilogy(frequencies, mean_psd, color="#3b76c1", linewidth=1.5)
    for low, high in bands.values():
        axes[0].axvspan(low, high, alpha=0.10, color="#e97397")
    axes[0].set_xlim(0, min(fs / 2, 60))
    axes[0].set_title("平均功率谱密度")
    axes[0].set_xlabel("频率 (Hz)")
    axes[0].set_ylabel("PSD (µV²/Hz)")
    axes[0].grid(alpha=0.25)

    band_names = list(bands)
    relative_columns = [f"频域__{name}_relative_power" for name in band_names]
    if all(column in channel_table for column in relative_columns):
        relative = channel_table[relative_columns].to_numpy(dtype=float)
        mean_relative = np.mean(relative, axis=0) * 100
        std_relative = np.std(relative, axis=0) * 100
        axes[1].bar(band_names, mean_relative, yerr=std_relative, color="#77b26c", capsize=4)
        axes[1].set_title("各频段相对功率（全通道均值 ± 标准差）")
        axes[1].set_ylabel("相对功率 (%)")
        axes[1].grid(axis="y", alpha=0.25)
    else:
        axes[1].text(0.5, 0.5, "频段特征不可用", ha="center", va="center")
        axes[1].axis("off")
    spectrum_path = output / f"{slug}_频域特征.png"
    fig.savefig(spectrum_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    figure_paths.append(spectrum_path)

    progress(0.98, "绘制时频图与脑网络图")
    channel_index = int(np.argmax(np.var(data, axis=1)))
    stft_size = int(min(512, 2 ** math.floor(math.log2(max(32, data.shape[1] // 8)))))
    stft_overlap = int(stft_size * 0.75)
    if stft_size >= data.shape[1]:
        stft_size = int(2 ** math.floor(math.log2(max(32, data.shape[1] // 2))))
        stft_overlap = stft_size // 2
    if stft_size > 4:
        stft_frequencies, stft_times, transform = signal.stft(
            data[channel_index],
            fs=fs,
            nperseg=stft_size,
            noverlap=stft_overlap,
            boundary=None,
            padded=False,
        )
        power = np.abs(transform) ** 2
        fig, ax = plt.subplots(figsize=(11, 5), constrained_layout=True)
        mesh = ax.pcolormesh(
            stft_times, stft_frequencies, 10 * np.log10(power + 1e-20), shading="auto", cmap="magma"
        )
        ax.set_ylim(0, min(45, fs / 2))
        ax.set_title(f"代表性通道 {labels[channel_index]} 的 STFT 时频图")
        ax.set_xlabel("有效片段拼接时间 (s)，跨剔除边界需谨慎解释")
        ax.set_ylabel("频率 (Hz)")
        fig.colorbar(mesh, ax=ax, label="功率 (dB)")
        tf_path = output / f"{slug}_时频图.png"
        fig.savefig(tf_path, dpi=160, bbox_inches="tight")
        plt.close(fig)
        figure_paths.append(tf_path)

    network_band = next((name for name in settings_network_bands(metadata) if name in bands), None)
    if network_band:
        low, high = bands[network_band]
        matrix = plv_matrix(data, fs, low, high, 60_000)
        fig, ax = plt.subplots(figsize=(6.5, 5.5), constrained_layout=True)
        image = ax.imshow(matrix, cmap="viridis", vmin=0, vmax=1)
        ax.set_xticks(range(len(labels)), labels, rotation=90, fontsize=8)
        ax.set_yticks(range(len(labels)), labels, fontsize=8)
        ax.set_title(f"{network_band} 频段 PLV 脑网络矩阵")
        fig.colorbar(image, ax=ax, label="PLV")
        network_path = output / f"{slug}_脑网络_{network_band}.png"
        fig.savefig(network_path, dpi=160, bbox_inches="tight")
        plt.close(fig)
        figure_paths.append(network_path)

    return figure_paths


def settings_network_bands(metadata: dict[str, Any]) -> list[str]:
    """从 metadata 中取得网络频段，避免绘图函数额外依赖全局状态。"""
    network_features = metadata.get("network_features", {})
    names: list[str] = []
    for key in network_features:
        match = re.match(r"(.+?)_plv_", key)
        if match and match.group(1) not in names:
            names.append(match.group(1))
    return names


def build_feature_workbook(
    channel_table: pd.DataFrame,
    summary_table: pd.DataFrame,
    header: HeaderInfo,
    preprocessed: dict[str, Any],
    metadata: dict[str, Any],
    output_path: PathLike,
    settings: AnalysisSettings,
    annotations: Sequence[str] | None = None,
) -> Path:
    """生成Excel工作簿：参数、通道、特征和注释分别放在工作表，方便核对来源。"""
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
    from openpyxl.utils import get_column_letter

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    workbook = Workbook()
    header_fill = PatternFill("solid", fgColor="3B76C1")
    section_fill = PatternFill("solid", fgColor="DDEBF7")
    title_font = Font(name="Microsoft YaHei", size=14, bold=True, color="FFFFFF")
    normal_font = Font(name="Microsoft YaHei", size=10)
    thin = Side(style="thin", color="D9E2F3")
    border = Border(left=thin, right=thin, top=thin, bottom=thin)

    info = workbook.active
    info.title = "文件信息"
    info.append(["项目", "值"])
    info_values = {
        "文件名": header.file_name,
        "文件路径": header.file_path,
        "格式": header.format,
        "版本": header.version,
        "患者字段": header.patient,
        "记录字段": header.recording,
        "开始日期": header.start_date,
        "开始时间": header.start_time,
        "原始通道数": header.channel_count,
        "分析通道数": header.analysis_channel_count or len(preprocessed["channel_names"]),
        "数据记录数": header.data_records,
        "记录时长_s": header.record_duration,
        "原始时长_s": header.duration_seconds,
        "分析采样率_Hz": preprocessed["fs"],
        "原始采样率_Hz": preprocessed["source_fs"],
        "使用时长_s": preprocessed["duration_seconds"],
        "片段保留比例": preprocessed["keep_ratio"],
        "干净片段数": preprocessed["epoch_count"],
        "总片段数": preprocessed["epoch_total"],
        "读取器": header.reader,
        "注释数量": header.annotations,
    }
    for key, value in info_values.items():
        info.append([key, value])
    if header.notes:
        info.append(["读取提示", "；".join(header.notes)])
    if preprocessed.get("notes"):
        info.append(["预处理提示", "；".join(preprocessed["notes"])])

    settings_sheet = workbook.create_sheet("分析设置")
    settings_sheet.append(["参数", "值"])
    for key, value in asdict(settings).items():
        settings_sheet.append([key, str(value)])

    summary_sheet = workbook.create_sheet("特征汇总")
    summary_sheet.append(["特征", "数值"])
    for row in summary_table.itertuples(index=False):
        summary_sheet.append([row[0], row[1]])

    channel_sheet = workbook.create_sheet("通道级特征")
    channel_sheet.append(list(channel_table.columns))
    for row in channel_table.itertuples(index=False, name=None):
        channel_sheet.append(list(row))

    qc_sheet = workbook.create_sheet("通道质控")
    qc_columns = ["原始序号", "通道", "单位", "状态", "标准差", "是否饱和或异常"]
    qc_sheet.append(qc_columns)
    for record in preprocessed.get("channel_records", []):
        qc_sheet.append([record.get(column, "") for column in qc_columns])

    network_sheet = workbook.create_sheet("脑网络指标")
    network_sheet.append(["指标", *settings.network_bands])
    network_features = metadata.get("network_features", {})
    network_metrics = (
        "mean_strength",
        "strength_std",
        "mean_degree",
        "degree_std",
        "clustering",
        "global_efficiency",
        "characteristic_path_length",
        "small_worldness",
    )
    if settings.feature_profile == "thesis_v1":
        network_metrics = (
            "clustering",
            "degree_mean",
            "degree_std",
            "global_efficiency",
            "characteristic_path_length",
            "connectivity_strength",
        )
    for metric in network_metrics:
        network_sheet.append(
            [
                metric,
                *[
                    network_features.get(
                        (
                            f"{band}_{metric}"
                            if settings.feature_profile == "thesis_v1"
                            else f"{band}_plv_{metric}"
                        ),
                        math.nan,
                    )
                    for band in settings.network_bands
                ],
            ]
        )

    description_sheet = workbook.create_sheet("字段说明")
    description_sheet.append(["特征类别", "说明"])
    for category, description in FEATURE_DESCRIPTIONS.items():
        description_sheet.append([category, description])
    for category, names in metadata.get("feature_categories", {}).items():
        description_sheet.append([f"{category}（{len(names)} 项）", "；".join(names)])
    if metadata.get("definition"):
        definition_sheet = workbook.create_sheet("论文方案口径")
        definition_sheet.append(["序号", "说明"])
        for index, text in enumerate(metadata["definition"], 1):
            definition_sheet.append([index, text])
        dictionary_sheet = workbook.create_sheet("论文特征字典与缺失")
        dictionary_sheet.append(["特征", "单位", "定义", "缺失原因"])
        for row in metadata["dictionary"]:
            dictionary_sheet.append([row[key] for key in ("特征", "单位", "定义", "缺失原因")])
        if metadata.get("research_appendix"):
            appendix_sheet = workbook.create_sheet("探索性分析附录")
            appendix_rows = metadata["research_appendix"]["rows"]
            appendix_sheet.append(list(appendix_rows[0]))
            for row in appendix_rows:
                appendix_sheet.append(list(row.values()))

    annotation_sheet = workbook.create_sheet("注释")
    if metadata.get("montage_info"):
        montage_sheet = workbook.create_sheet("Montage模板映射")
        montage_rows = metadata["montage_info"]["rows"]
        if montage_rows:
            montage_sheet.append(list(montage_rows[0]))
            for row in montage_rows:
                montage_sheet.append(list(row.values()))
    annotation_sheet.append(["序号", "注释"])
    for index, annotation in enumerate(annotations or [], start=1):
        annotation_sheet.append([index, annotation])
    if not annotations:
        annotation_sheet.append([0, "文件未包含可解析注释。"])

    for sheet in workbook.worksheets:
        sheet.freeze_panes = "A2"
        for cell in sheet[1]:
            cell.fill = header_fill
            cell.font = title_font
            cell.alignment = Alignment(horizontal="center", vertical="center")
            cell.border = border
        for row in sheet.iter_rows(min_row=2):
            for cell in row:
                cell.font = normal_font
                cell.border = border
                if isinstance(cell.value, float):
                    cell.number_format = "0.000000"
        for column_cells in sheet.columns:
            max_length = 0
            letter = get_column_letter(column_cells[0].column)
            for cell in column_cells[:200]:
                if cell.value is not None:
                    max_length = max(max_length, len(str(cell.value)))
            sheet.column_dimensions[letter].width = min(max(max_length + 2, 12), 48)
    info["A1"].fill = header_fill
    workbook.save(path)
    return path
