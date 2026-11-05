"""论文附录：把特征值、公式、单位和解释边界整理在一起，不生成疾病诊断。

Complete, escaped and printable research appendix; no change to EEG estimates."""

from collections import Counter
from html import escape
import math
from pathlib import Path

import pandas as pd

NOTICE = (
    "本网站是论文的探索性脑电特征分析工具。本附录描述当前记录的信号特征及个体内变化，"
    "未将这些数值转换为经过验证的产后抑郁诊断、风险概率或干预结论。"
    "特征是否具有预测价值，还需结合结局标签，在受试者独立的验证中检验；不得把同一人的多个片段当成独立受试者。"
)
TIME = {
    "mean": ("x̄=Σx/N", "描述片段的中心位置；结合波形查看基线。"),
    "std": ("σ=√[Σ(x−x̄)²/(N−1)]", "描述整体幅度波动。"),
    "variance": ("σ²=Σ(x−x̄)²/(N−1)", "描述幅度离散程度；与Hjorth活动性重复。"),
    "skewness": ("E[(x−x̄)³]/σ³", "描述波形分布的不对称程度，可辅助检查不稳定片段。"),
    "kurtosis": ("E[(x−x̄)⁴]/σ⁴−3", "描述分布尖锐程度，对少量极端幅值敏感。"),
    "zero_crossing_rate": (
        "Σ|sgn(x[t+1])−sgn(x[t])|/[2(N−1)]",
        "描述零点穿越频繁程度，零点贡献按公式计算。",
    ),
    "line_length": ("Σ|x[t+1]−x[t]|", "描述2秒波形的总变化量，同时受幅度和频率影响。"),
    "rms": ("√[Σx²/N]", "描述片段的整体幅度/能量水平。"),
    "peak_to_peak": ("max(x)−min(x)", "描述最大幅度范围，对瞬态极值敏感。"),
    "hjorth_activity": ("Activity=var(x)=σ²", "与方差相同，不能计为独立证据。"),
    "hjorth_mobility": (
        "√[var(x′)/var(x)]；x′=fs·diff(x)",
        "描述一阶变化相对整体波动的尺度，单位s⁻¹。",
    ),
    "hjorth_complexity": (
        "Mobility(x′)/Mobility(x)；x″=fs·diff(x′)",
        "描述导数变化相对原信号变化的复杂程度。",
    ),
}
NETWORK = {
    "clustering": ("mean_i[2t_i/(k_i(k_i−1))]；k_i<2取0", "描述阈值后二值图邻居间互连程度。"),
    "degree_mean": ("mean_i(k_i)", "描述每节点保留边数；固定密度下受节点数约束。"),
    "degree_std": ("std_i(k_i)，ddof=0", "描述节点保留边数的不均衡程度。"),
    "global_efficiency": (
        "Σ(i≠j)1/d_ij/[n(n−1)]；断连对贡献0",
        "描述加权图最短路效率；边长=1/PLV。",
    ),
    "characteristic_path_length": (
        "mean(d_ij | i≠j且可达)",
        "描述可达节点间的加权路径，须结合可达比例。",
    ),
    "connectivity_strength": (
        "mean_{i<j}(PLV_ij)，阈值前",
        "描述电极间相位同步强度；不代表解剖连接。",
    ),
}


def specification(feature):
    parts = feature.split("__")
    domain, code = parts[:2]
    scope = parts[2] if len(parts) > 2 else "全网络"
    band = next(
        (b for b in ("delta", "theta", "alpha", "beta", "gamma") if code.startswith(b + "_")), ""
    )
    if code == "theta_beta_ratio":
        band = ""
    base = code[len(band) + 1 :] if band else code
    if domain == "时域":
        formula, purpose = TIME[code]
        limits = "受幅度单位、参考、采样率和伪迹影响；逐2秒片段计算后平均。"
        chart = "时域波形（形态参考，精确数值见附录）"
    elif domain == "网络":
        formula, purpose = NETWORK[base]
        formula = "PLV=|mean exp(j(φi−φj))|；" + formula
        limits = "受参考、公共信号和阈值定义影响；不是脑源定位或已验证抑郁标志。"
        chart = "对应频段PLV矩阵（矩阵不直接显示全部图论量）"
    elif domain == "时频域":
        formula = "B(t)=Σ_{f∈band}PSD(f,t)Δf；"
        formula += "std_t(B)，ddof=1" if base == "TF_std" else "std_t(B)/mean_t(B)，ddof=1"
        purpose = (
            "描述频段总功率随时间的绝对起伏。"
            if base == "TF_std"
            else "描述频段总功率随时间的相对起伏。"
        )
        limits = "至少2个有效STFT窗口；CV均值为0时缺失；颜色不能直接读出精确CV。"
        chart = "时频图（功率分布参考，标准差/CV见附录）"
    else:
        formula, purpose = {
            "absolute_power": ("Σ_{f∈band}PSD(f)Δf", "描述该频段的绝对能量。"),
            "relative_power": ("P_band/P_0.5–45Hz", "描述该频段在总能量中的占比。"),
            "total_power": ("Σ_{0.5≤f≤45}PSD(f)Δf", "描述所分析频率范围的整体能量。"),
            "spectral_entropy": (
                "−Σp_k ln(p_k)；p_k=PSD_kΔf/P_total",
                "描述功率在频点间的分散程度；自然对数、未归一化。",
            ),
            "peak_alpha_frequency": (
                "argmax_{8≤f<13}PSD(f)",
                "描述α频段功率最高的频点，不作峰显著性判定。",
            ),
            "theta_beta_ratio": ("P_theta/P_beta", "描述θ与β功率的相对关系。"),
            "FAA": (
                "ln(P_alpha,右)−ln(P_alpha,左)",
                "描述指定左右电极对α功率差；正值表示右侧α功率较高。",
            ),
        }[base]
        limits = "依赖片段选择、参考和谱估计；无本队列诊断阈值。"
        if base == "FAA":
            limits = "缺少电极对则缺失；C3/C4和T3/T4分别为中央/颞区，不能统称额叶。"
        chart = "频域图（FAA/谱熵/功率比等精确数值见附录）"
    if domain != "网络" and code != "FAA":
        if domain == "时域":
            formula += "；通道内先对有效2秒片段的指标取均值"
        if scope == "全脑标准差":
            formula += "；再对保留通道的指标取std(ddof=0)"
        elif scope == "全脑均值":
            formula += "；再对保留通道的指标取均值"
        else:
            formula += f"；再对{scope}保留电极的指标取均值"
    return {
        "特征域": domain,
        "频段": band or ("alpha" if code == "FAA" else "—"),
        "汇总范围或电极对": scope,
        "公式或定义": formula,
        "探索用途": purpose,
        "解释界限": limits,
        "对应图": chart,
    }


def make_appendix(summary, metadata, context="当前记录"):
    from eeg_feature_engine import PIPELINE_VERSION

    dictionary = {r["特征"]: r for r in metadata["dictionary"]}
    rows = []
    for item in summary.to_dict("records"):
        feature = item["特征"]
        details = dictionary[feature]
        rows.append(
            {
                "特征": feature,
                **specification(feature),
                "中文说明": details["定义"],
                "数值": float(item["数值"]),
                "单位": details["单位"],
                "缺失原因": details["缺失原因"],
            }
        )
    return {
        "title": "论文探索性分析附录",
        "context": context,
        "notice": NOTICE,
        "pipeline_version": PIPELINE_VERSION,
        "profile": "thesis_v1",
        "counts": dict(Counter(r["特征域"] for r in rows)),
        "rows": rows,
        "methods": metadata["definition"],
        "coverage": metadata["coverage"],
    }


def make_paired_appendix(comparison_rows, metadata_a, metadata_b, label_a, label_b):
    frame = pd.DataFrame(
        {
            "特征": [r["特征"] for r in comparison_rows],
            "数值": [r["记录A"] for r in comparison_rows],
        }
    )
    appendix = make_appendix(frame, metadata_a, f"A：{label_a}；B：{label_b}；差值=B−A")
    b_dictionary = {r["特征"]: r for r in metadata_b["dictionary"]}
    for row, comparison in zip(appendix["rows"], comparison_rows):
        del row["数值"], row["缺失原因"]
        row.update(
            {
                "记录A": comparison["记录A"],
                "记录B": comparison["记录B"],
                "变化量_B减A": comparison["变化量_B减A"],
                "A缺失原因": comparison["A缺失原因"],
                "B缺失原因": b_dictionary[row["特征"]]["缺失原因"],
            }
        )
    appendix["coverage"] = {"A": metadata_a["coverage"], "B": metadata_b["coverage"]}
    return appendix


def display(value):
    if isinstance(value, float):
        return format(value, ".8g") if math.isfinite(value) else "缺失"
    return "" if value is None else str(value)


def write_appendix(appendix, output, prefix):
    """Export the exact rows seen in the UI, including values and missing reasons."""
    output = Path(output)
    csv = output / f"{prefix}_探索性分析附录.csv"
    pd.DataFrame(appendix["rows"]).to_csv(csv, index=False, encoding="utf-8-sig")
    html = output / f"{prefix}_探索性分析附录.html"

    def safe(value):
        return escape(display(value), quote=True)

    tables = []
    for domain in ("时域", "频域", "时频域", "网络"):
        rows = [r for r in appendix["rows"] if r["特征域"] == domain]
        if not rows:
            continue
        columns = ["特征", "中文说明", "单位"] + (
            ["数值"] if "数值" in rows[0] else ["记录A", "记录B", "变化量_B减A"]
        )
        columns += ["公式或定义", "探索用途", "解释界限"] + (
            ["缺失原因"] if "数值" in rows[0] else ["A缺失原因", "B缺失原因"]
        )
        tables.append(
            f"<h2>{safe(domain)}（{len(rows)}项）</h2><table><thead><tr>"
            + "".join(f"<th>{safe(k)}</th>" for k in columns)
            + "</tr></thead><tbody>"
            + "".join(
                "<tr>" + "".join(f"<td>{safe(row[k])}</td>" for k in columns) + "</tr>"
                for row in rows
            )
            + "</tbody></table>"
        )
    methods = "<ol>" + "".join(f"<li>{safe(m)}</li>" for m in appendix["methods"]) + "</ol>"
    import json

    coverage = safe(json.dumps(appendix["coverage"], ensure_ascii=False, indent=2))
    html.write_text(
        '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
        "<meta http-equiv=\"Content-Security-Policy\" content=\"default-src 'none'; style-src 'unsafe-inline'\">"
        "<title>论文探索性分析附录</title><style>body{font:14px/1.7 system-ui, sans-serif;margin:32px;color:#23374a}h1{font-size:26px}h2{margin-top:30px}"
        "table{border-collapse:collapse;width:100%;font-size:11px}th,td{border:1px solid #ccd7e4;padding:7px;vertical-align:top;overflow-wrap:anywhere}"
        "th{background:#edf3fa}thead{display:table-header-group}tr{break-inside:avoid}.notice{background:#edf3fa;border-left:4px solid #3769a0;padding:14px}"
        "pre{white-space:pre-wrap}@page{size:A4 landscape;margin:12mm}@media print{body{margin:0;font-size:10px}h2{break-after:avoid}table{font-size:8px}}</style>"
        f'<h1>{safe(appendix["title"])}</h1><p>{safe(appendix["profile"])} · v{safe(appendix["pipeline_version"])} · {safe(appendix["context"])}</p><p class="notice">{safe(appendix["notice"])}</p>'
        f'<p>共{len(appendix["rows"])}项；缺失不补零。浏览器按Ctrl+P可打印或另存为PDF。符号N为单个2秒片段采样点数，E为片段内均值，σ为样本标准差。</p>'
        + "".join(tables)
        + "<h2>计算口径</h2>"
        + methods
        + "<h2>有效数据覆盖</h2><pre>"
        + coverage
        + "</pre></html>",
        encoding="utf8",
    )
    return [str(csv), str(html)]
