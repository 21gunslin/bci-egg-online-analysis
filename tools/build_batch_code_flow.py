"""用常见的Matplotlib画一张JPG，介绍这套批处理代码的阅读路线。"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


def main():
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, ax = plt.subplots(figsize=(14, 10))
    fig.patch.set_facecolor("#eef3f8")
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis("off")
    ax.text(0.03, 0.96, "从BDF到AUC：代码怎么读", fontsize=26, weight="bold", color="#17364c")
    ax.text(
        0.03,
        0.912,
        "运行按1→6；初学者先读第6步，再追踪第5、4步，最后回看文件处理。",
        fontsize=13,
        color="#405d72",
    )

    def box(x, y, number, title, filename, explanation, color):
        ax.add_patch(
            FancyBboxPatch(
                (x, y),
                0.28,
                0.20,
                boxstyle="round,pad=0.008",
                facecolor="white",
                edgecolor=color,
                linewidth=2,
            )
        )
        ax.text(x + 0.014, y + 0.158, f"{number}  {title}", fontsize=17, weight="bold", color=color)
        ax.text(x + 0.014, y + 0.116, filename, fontsize=11, color="#405568")
        ax.text(
            x + 0.014,
            y + 0.062,
            explanation,
            fontsize=12,
            color="#253c50",
            va="center",
            linespacing=1.7,
        )

    def arrow(start, stop):
        ax.add_patch(
            FancyArrowPatch(
                start, stop, arrowstyle="-|>", mutation_scale=20, linewidth=2, color="#70899c"
            )
        )

    box(
        0.03,
        0.66,
        1,
        "找到每一份data",
        "run_bdf_fif_features.py",
        "按编号和时间整理两次记录\n不把evt、spike当普通脑电",
        "#296b9b",
    )
    box(
        0.36,
        0.66,
        2,
        "保真转换FIF",
        "backend/eeg_batch_fif.py",
        "不滤波、不删样本；读回核对\n原BDF旁保存fdata.fif",
        "#296b9b",
    )
    box(
        0.69,
        0.66,
        3,
        "逐份质控与提特征",
        "run_bdf_fif_features.py",
        "仅在特征副本上滤波与质控\n每份287项，失败原因留痕",
        "#296b9b",
    )
    arrow((0.316, 0.76), (0.349, 0.76))
    arrow((0.646, 0.76), (0.679, 0.76))
    arrow((0.83, 0.646), (0.83, 0.594))

    box(
        0.69,
        0.38,
        4,
        "汇成一次记录",
        "aggregate_subject_features.py",
        "合格文件按电极名称对齐\n汇集窗口，保留文件和坏段断点",
        "#237b70",
    )
    box(
        0.36,
        0.38,
        5,
        "每位产妇一行",
        "aggregate_subject_features.py",
        "前287＋后287＋差值287\n匹配PPD答案，排除已知错误",
        "#237b70",
    )
    box(
        0.03,
        0.38,
        6,
        "随机森林与AUC",
        "train_random_forest.py",
        "从main()→evaluate_model()读\n用未参加训练的预测来评价",
        "#c48022",
    )
    arrow((0.679, 0.48), (0.646, 0.48))
    arrow((0.349, 0.48), (0.316, 0.48))

    ax.add_patch(
        FancyBboxPatch(
            (0.03, 0.065),
            0.94,
            0.235,
            boxstyle="round,pad=0.008",
            facecolor="#fff5df",
            edgecolor="#d7ad56",
            linewidth=1.5,
        )
    )
    ax.text(
        0.05,
        0.254,
        "第6步最关键：训练者学习，测试者考试",
        fontsize=19,
        weight="bold",
        color="#8c5b13",
    )
    ax.text(
        0.05,
        0.208,
        "训练折：填补缺失值 → 删除常数列 → 500棵树学习规则（Pipeline）",
        fontsize=14,
        color="#30485b",
    )
    ax.text(
        0.05,
        0.162,
        "测试折：只提供脑电题目，让模型打分；答案仅用于计算评价指标",
        fontsize=14,
        color="#30485b",
    )
    ax.text(
        0.05,
        0.116,
        "5组轮流测试 × 3次固定划分 → 每人3份OOF分数取平均 → ROC-AUC",
        fontsize=14,
        color="#30485b",
    )
    ax.text(
        0.03,
        0.018,
        "一行＝一个人；同一个人不能同时训练和测试。内部探索性评价，不等于临床诊断或外部验证。",
        fontsize=12,
        color="#4f6678",
    )
    output = Path(__file__).resolve().parents[1] / "docs" / "batch_random_forest_flow.jpg"
    fig.savefig(output, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(output)


if __name__ == "__main__":
    main()
