"""生成代码阅读流程JPG，只画代码调用关系，不读取任何脑电或患者资料。"""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def build(output: Path, font_path: Path):
    """用中文字体画一张可单独查看、打印的入门路线图。"""
    width, height = 2400, 3150
    picture = Image.new("RGB", (width, height), "#F2F6FA")
    draw = ImageDraw.Draw(picture)

    def font(size):
        return ImageFont.truetype(str(font_path), size)

    title_font = font(76)
    heading_font = font(43)
    body_font = font(35)
    small_font = font(30)
    blue = "#245E94"
    teal = "#176F69"
    orange = "#9B6221"
    ink = "#20354B"
    muted = "#52677C"

    def wrapped(text, x, y, max_width, text_font, color=ink, line_height=50):
        line = ""
        for character in text:
            if character == "\n" or (
                line and draw.textlength(line + character, font=text_font) > max_width
            ):
                draw.text((x, y), line, font=text_font, fill=color)
                y += line_height
                line = ""
                if character == "\n":
                    continue
            line += character
        if line:
            draw.text((x, y), line, font=text_font, fill=color)
            y += line_height
        return y

    draw.text((110, 70), "从按钮到脑电结果：代码阅读路线", font=title_font, fill=ink)
    wrapped(
        "先追一份文件如何进、如何出；看懂主线后，再读公式和特殊兼容代码。",
        115,
        180,
        2150,
        body_font,
        muted,
    )
    draw.rounded_rectangle((110, 252, 2290, 282), radius=15, fill="#DCE7EF")
    stages = [
        (
            "先找页面上的按钮",
            "web/eeg_feature_app.html",
            "搜索 singleForm：文件、BDF/FIF单选、参数。",
            blue,
        ),
        (
            "再看按钮触发什么函数",
            "web/eeg_single_analysis.js",
            "handleSingleSubmit → analyzeSingleFile；收集选择并等待返回。",
            blue,
        ),
        (
            "请求到达Python服务",
            "backend/eeg_web_server.py",
            "_single_analysis：检查格式、参数，建立输出目录。",
            teal,
        ),
        (
            "先读后端主流程",
            "backend/eeg_feature_engine.py",
            "analyze_eeg_file：只安排读取、处理、计算和导出的顺序。",
            teal,
        ),
        (
            "读取文件与单位",
            "backend/eeg_readers.py  /  eeg_fif.py",
            "MNE读取 → 数值表、通道名、单位、文件信息和注释。",
            teal,
        ),
        (
            "理解预处理和质控",
            "backend/eeg_preprocessing.py",
            "单位 → 滤波 → 通道质控 → 合格片段；保留原时间索引。",
            teal,
        ),
        (
            "最后逐项学习公式",
            "eeg_features.py  /  eeg_thesis_features.py",
            "时域、频域、时频、网络 → 通道表和汇总表。",
            teal,
        ),
        (
            "沿着结果回到网页",
            "eeg_reporting.py → eeg_ui_common.js",
            "图 / CSV / Excel → JSON与图片链接 → render显示。",
            orange,
        ),
    ]
    x = 110
    card_width = 1420
    card_height = 220
    gap = 45
    start = 330
    for index, (heading, path, description, color) in enumerate(stages, 1):
        y = start + (index - 1) * (card_height + gap)
        draw.rounded_rectangle(
            (x + 5, y + 8, x + card_width + 5, y + card_height + 8), radius=22, fill="#DAE3EB"
        )
        draw.rounded_rectangle(
            (x, y, x + card_width, y + card_height),
            radius=22,
            fill="white",
            outline="#D4DEE7",
            width=2,
        )
        draw.rounded_rectangle((x, y, x + 18, y + card_height), radius=9, fill=color)
        draw.ellipse((x + 34, y + 28, x + 109, y + 103), fill=color)
        number = str(index)
        number_width = draw.textlength(number, font=heading_font)
        draw.text((x + 71 - number_width / 2, y + 35), number, font=heading_font, fill="white")
        draw.text((x + 135, y + 28), heading, font=heading_font, fill=color)
        wrapped(path, x + 135, y + 91, card_width - 165, small_font, ink, line_height=40)
        wrapped(description, x + 135, y + 141, card_width - 165, small_font, muted, line_height=42)
        if index < len(stages):
            center = x + card_width / 2
            top = y + card_height + 9
            bottom = top + gap - 18
            draw.line((center, top, center, bottom), fill="#8399AD", width=5)
            draw.polygon(
                [(center - 11, bottom - 8), (center + 11, bottom - 8), (center, bottom + 5)],
                fill="#8399AD",
            )

    right = 1610
    right_width = 680

    def panel(y, heading, rows, color=blue):
        content_height = sum(
            sum(
                max(1, int(draw.textlength(line, font=small_font) / (right_width - 66)) + 1)
                for line in text.split("\n")
            )
            * 44
            + 20
            for text in rows
        )
        height = content_height + 112
        draw.rounded_rectangle(
            (right, y, right + right_width, y + height),
            radius=22,
            fill="white",
            outline="#D4DEE7",
            width=2,
        )
        draw.text((right + 30, y + 25), heading, font=heading_font, fill=color)
        cursor = y + 100
        for text in rows:
            cursor = (
                wrapped(
                    text, right + 33, cursor, right_width - 66, small_font, muted, line_height=44
                )
                + 20
            )
        return max(y + height, cursor + 15)

    y = panel(
        330,
        "先认识这些词",
        [
            "data：通道 × 采样点的数值表",
            "fs=200：每秒200个点；2秒=400点",
            "header：文件信息卡；settings：分析参数表",
            "JSON：前后端交换的小参数和结果说明",
            "await：等待服务返回，再继续下一步",
            "NaN：缺失或不能计算，不是零",
        ],
    )
    y = panel(
        y + 40,
        "MNE负责什么",
        [
            "read_raw_bdf / read_raw_fif：标准读取",
            "filter_data / notch_filter：既定参数的零相位滤波",
            "psd_array_welch：平均PSD或逐窗功率",
            "Raw / Annotations / Montage：FIF、标记与模板坐标",
        ],
        teal,
    )
    y = panel(
        y + 40,
        "主线看懂后再读",
        [
            "多份：eeg_session_ui.js\n→ eeg_session.py",
            "配对：eeg_pair_analysis.js\n→ eeg_pair_spike.py",
            "黄绿巡检：eeg_artifact_patrol.js\n→ eeg_patrol_math.js",
            "原始保真：eeg_raw_qc_api.py\n→ eeg_raw_qc.py；原生BDF兼容层最后读",
        ],
        teal,
    )
    panel(
        y + 40,
        "黄绿按钮这条支线",
        [
            "黄色：找候选并定位原波形",
            "绿色：只把候选追加到草稿",
            "人工：核对、删除、补充",
            "正式导出：才写入新的FIF",
        ],
        orange,
    )

    draw.rounded_rectangle((110, 2560, 1530, 2780), radius=24, fill="#E5EEF6")
    wrapped("不必先看懂全部数学。先确认：", 145, 2590, 1350, heading_font, blue, line_height=60)
    wrapped(
        "一个函数拿进来什么？做了什么？交出去什么？",
        145,
        2665,
        1320,
        body_font,
        ink,
        line_height=52,
    )
    draw.rounded_rectangle((110, 2870, 2290, 3050), radius=24, fill=ink)
    wrapped(
        "配套阅读：docs/CODE_READING_GUIDE.md",
        150,
        2900,
        2100,
        heading_font,
        "white",
        line_height=60,
    )
    wrapped(
        "按当前 v0.11.0 文件与函数整理 · 图中蓝色为前端，绿色为后端，棕色为输出",
        150,
        2980,
        2100,
        small_font,
        "#DCE7EF",
        line_height=42,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    picture.save(output, format="JPEG", quality=95, subsampling=0, optimize=True, dpi=(220, 220))
    return output


def main():
    parser = argparse.ArgumentParser(description="生成代码阅读流程JPG")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "docs/code_reading_flow.jpg",
    )
    parser.add_argument("--font", type=Path, default=Path("C:/Windows/Fonts/msyh.ttc"))
    args = parser.parse_args()
    if not args.font.is_file():
        raise SystemExit("请用--font提供可显示中文的字体文件。")
    print(build(args.output, args.font))


if __name__ == "__main__":
    main()
