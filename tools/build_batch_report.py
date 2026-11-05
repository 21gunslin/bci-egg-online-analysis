"""把批处理与随机森林的实际输出写成可离线阅读的中文HTML报告。"""

import argparse
import ast
from datetime import datetime
import html
import json
import os
from pathlib import Path

import pandas as pd


def read_json(output, name):
    return json.loads((output / name).read_text(encoding="utf8"))


def percent(value):
    return f"{value * 100:.1f}%"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output
    project = Path(__file__).resolve().parents[1]
    try:
        code_url = Path(os.path.relpath(project, output)).as_posix()
    except ValueError:  # Windows中结果与代码不在同一个盘时，使用本机文件链接。
        code_url = project.as_uri()
    cohort = read_json(output, "cohort_summary.json")
    progress = read_json(output, "progress.json")
    metrics = read_json(output, "random_forest_results.json")
    protocol = read_json(output, "random_forest_protocol.json")
    dataset_name = Path(read_json(output, "protocol.json")["input_root"]).name
    statuses = pd.read_csv(output / "file_status_private.csv")
    sessions = pd.read_csv(output / "session_status_private.csv")
    # 仅作复核：电极覆盖差异不进入模型，也不能被包装成生理变化证据。
    subject_ids = pd.read_csv(output / "subject_features_private.csv", usecols=["id"])["id"]
    pair_rows = []
    for sid, group in sessions[sessions["id"].isin(subject_ids)].groupby("id"):
        before = set(ast.literal_eval(group[group["phase"] == "pre"].iloc[0]["channels"]))
        after = set(ast.literal_eval(group[group["phase"] == "post"].iloc[0]["channels"]))
        pair_rows.append(
            {
                "id": int(sid),
                "pre_channel_count": len(before),
                "post_channel_count": len(after),
                "shared_channel_count": len(before & after),
                "same_channel_set": before == after,
            }
        )
    pair_coverage = pd.DataFrame(pair_rows)
    pair_coverage.to_csv(
        output / "pair_channel_coverage_private.csv", index=False, encoding="utf-8-sig"
    )
    pair_text = (
        f"{len(pair_coverage)}对记录中，{int(pair_coverage['same_channel_set'].sum())}对保留电极集合完全相同；"
        f"共同保留电极数中位数{pair_coverage['shared_channel_count'].median():.0f}，"
        f"最少{pair_coverage['shared_channel_count'].min()}个，"
        f"其中{int((pair_coverage['shared_channel_count'] < 3).sum())}对不足3个。"
    )
    audit = (
        read_json(output, "batch_audit.json") if (output / "batch_audit.json").is_file() else None
    )
    names = {
        "pre_only": "仅镇痛前脑电（对照）",
        "pre_post_delta": "镇痛前＋镇痛后＋差值（主要模型）",
    }
    rows = []
    markdown_rows = []
    for m in metrics:
        lo, hi = m["auc_ci95_fixed_oof_bootstrap"]
        repeated = "、".join(f"{value:.3f}" for value in m["repeat_oof_auc"])
        values = [
            names[m["model"]],
            str(m["input_features"]),
            f'{m["auc"]:.3f}',
            f"{lo:.3f}–{hi:.3f}",
            percent(m["accuracy_at_0_5"]),
            percent(m["sensitivity_at_0_5"]),
            percent(m["specificity_at_0_5"]),
            repeated,
        ]
        rows.append(
            "<tr>" + "".join("<td>" + html.escape(value) + "</td>" for value in values) + "</tr>"
        )
        markdown_rows.append("|" + "|".join(values) + "|")
    primary = next(m for m in metrics if m["model"] == "pre_post_delta")
    feature_ranges = []
    for m in metrics:
        fold_metrics = pd.read_csv(output / f"{m['model']}_fold_metrics.csv")
        counts = fold_metrics["features_after_training_only_processing"]
        feature_ranges.append(f"{names[m['model']]}为{counts.min()}–{counts.max()}列")
    lo, hi = primary["auc_ci95_fixed_oof_bootstrap"]
    if lo <= 0.5 <= hi:
        interpretation = "主要模型的区间包含0.5，当前内部评价尚不能支持稳定高于随机排序的区分能力。"
    elif lo > 0.5:
        interpretation = "主要模型在这批样本的内部评价中高于随机排序参考，但仍需独立样本验证，不能据此宣称临床可用。"
    else:
        interpretation = "主要模型的排序表现低于随机参考，当前方案未显示可用的预测效果；不把翻转标签或反向概率当成新的验证结论。"
    exclusions = "".join(
        f"<li>{html.escape(reason)}：{count}人</li>"
        for reason, count in cohort["exclusion_counts"].items()
    )
    failed = statuses[statuses["conversion"] != "ok"]
    failure_reasons = (
        failed["reason"].fillna("").value_counts() if "reason" in failed else pd.Series(dtype=int)
    )
    conversion_reasons = "".join(
        f"<li>{html.escape(str(reason))}：{count}份</li>"
        for reason, count in failure_reasons.items()
    )
    successful = sessions[sessions["status"] == "ok"]
    quality_text = ""
    if not successful.empty:
        quality_text = (
            f"可用记录的有效时长中位数为{successful['clean_seconds'].median():.1f}秒"
            f"（范围{successful['clean_seconds'].min():.1f}–{successful['clean_seconds'].max():.1f}秒），"
            f"共同保留通道数中位数{successful['channel_count'].median():.0f}"
            f"（范围{successful['channel_count'].min():.0f}–{successful['channel_count'].max():.0f}）。"
        )
    audit_text = "尚未生成最终源文件核验记录。"
    if audit:
        audit_text = (
            f"已复核{audit['source_stat_checks']}份源BDF的大小和修改时间，"
            f"并对{audit['source_sha256_rechecks']}份成功转换的源BDF重新核对SHA-256。"
            f"发现问题{audit['issue_count']}项；FIF共{audit['fif_files']}个，"
            f"占{audit['fif_bytes'] / 1024 ** 3:.2f} GiB。"
            f"转换前后最大样本误差{audit['max_fif_roundtrip_error']:.3g}。"
        )
    title = "BDF→FIF→脑电特征→随机森林：探索性结果"
    header = "<tr><th>模型</th><th>输入特征数</th><th>OOF AUC</th><th>95%区间*</th><th>ACC</th><th>敏感度</th><th>特异度</th><th>3次划分各自AUC</th></tr>"
    document = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title}</title><style>
body{{font:16px/1.8 "Microsoft YaHei",sans-serif;color:#172b40;background:#edf2f6;margin:0}}main{{max-width:1140px;margin:auto;padding:28px}}
section{{background:white;padding:24px 30px;margin:18px 0;border-radius:14px;box-shadow:0 2px 10px #ccd5de55}}h1{{font-size:28px}}h2{{font-size:22px;color:#17435f}}p{{margin:12px 0}}
table{{border-collapse:collapse;width:100%;font-size:14px}}th,td{{padding:11px;border:1px solid #d4dde5;text-align:left}}th{{background:#e8f2f9}}img{{max-width:100%;height:auto}}a{{color:#086eaa}}code{{background:#eef3f6;padding:2px 4px}}.big{{font-size:26px;font-weight:bold;color:#12537a}}.note{{background:#fff5d9;padding:16px;border-left:4px solid #d99b1d}}.scroll{{overflow-x:auto}}.flow{{display:flex;gap:10px;flex-wrap:wrap}}.flow span{{padding:12px;background:#e8f2f9;border-radius:8px}}small{{color:#566777}}
</style></head><body><main>
<h1>{title}</h1><p>本次范围：{html.escape(dataset_name)}；输入{cohort['input_subjects']}人。生成时间：{datetime.now():%Y-%m-%d %H:%M}。</p>
<section><h2>先看结果</h2><p class="big">主要模型 AUC＝{primary['auc']:.3f}，95%区间 {lo:.3f}–{hi:.3f}</p>
<p>最终纳入{cohort['included_subjects']}位独立产妇：PPD=1为{cohort['PPD_positive']}人，PPD=0为{cohort['PPD_negative']}人。主要模型使用镇痛前、镇痛后和差值；对照模型仅使用镇痛前脑电。</p>
<p class="note">{interpretation}这是按产妇划分的内部交叉验证结果，PPD标签的具体量表/诊断依据仍待补充。输出分数不是经过临床验证的风险概率。</p>
<div class="scroll"><table>{header}{''.join(rows)}</table></div>
<p><small>*区间来自对固定OOF预测按产妇、按类别自助抽样2000次，没有重训模型，因此没有涵盖全部建模不确定性。ACC、敏感度和特异度使用预先固定的0.5阈值。多数类别全部预测为同一类的ACC基线为{percent(protocol['majority_class_accuracy'])}；随机排序AUC参考为0.5。两模型的小幅AUC差异不能直接解释为已经证实的优势。</small></p></section>
<section><h2>数据经过了什么处理</h2><div class="flow"><span>{progress['total']}份data.bdf</span><span>→ 原始FIF保真转换</span><span>→ 独立文件质控与特征</span><span>→ 每次记录汇集</span><span>→ 每位产妇一行</span><span>→ 独立受试者测试</span></div>
<ul><li>共{cohort['input_subjects']}人、{cohort['input_sessions']}次记录、{progress['total']}份data。evt.bdf和spike.bdf没有混充普通脑电输入。</li>
<li>{progress['converted']}份转换成功；{progress['conversion_failed']}份转换失败；{progress['features_ok']}份生成287项特征，{progress['features_failed']}份转换后未通过特征质控。</li>
<li>{cohort['sessions_ok']}次记录可汇集特征；最终{cohort['included_subjects']}位产妇有完整镇痛前后记录并符合身份、结局条件。</li></ul>
<p>排除前，{cohort['input_subjects']}人基线匹配结局为PPD=1共{cohort['input_PPD_positive_before_exclusions']}人、PPD=0共{cohort['input_PPD_negative_before_exclusions']}人；此计数尚包含随后删除的错误记录。建模人群是能完成配对与质控的子集，可能存在选择偏差。</p>
<p>转换失败原因：</p><ul>{conversion_reasons or '<li>无</li>'}</ul><p>建模排除原因：</p><ul>{exclusions}</ul><p>{quality_text}</p>
<p>每份成功转换的<code>fdata.fif</code>与原<code>data.bdf</code>同目录保存。不同原始采样率分组保留；不滤波、不重采样、不删样本、不重新参考，不自动确认坏道。坐标为标准10–20模板，非实测坐标。</p>
<p>{audit_text}</p></section>
<section><h2>怎样提取特征</h2><p>只在计算特征的副本上沿用网站thesis_v1方案：目标250 Hz，0.5–45 Hz的Butterworth零相位带通，50/100 Hz适用陷波，2秒质控窗口，保留采集参考。按文件头单位换算，没有附加未经确认的÷1000修正。</p>
<p>每次记录287项：时域84、频域103、时频70、网络30。一次记录内采用合格文件共同保留的通道；缺失脑区或电极对留空，不造数。每人形成镇痛前287项、镇痛后287项、后减前287项。时频先汇总频段功率，再计算随时间的标准差和变异系数。</p>
<p>PLV沿用网站规则，在每个独立连续段完成带通和相位计算后，以60,000点为目标上限计算均匀抽样步长，较短记录使用全部点；步长与实际点数保存在各记录的coverage.json中。</p>
<p>文件间及被剔除窗口之间保留断点，滤波、频谱与网络不会把两端当连续脑电。每个时相使用全部合格数据，没有等长裁剪，也没有按未明确含义的evt编码划分临床状态。网络固定密度25%；度均值受节点数约束，方差与Hjorth活动性重复，不能把所有列理解为独立生物标志物。</p></section>
<section><h2>怎样建立与评价随机森林</h2><ol>
<li>先完成所有无结局参与的质控与特征计算，再用唯一编号匹配基线PPD标签。PPD=1且PPD time=0的记录按用户确认删除，身份冲突不自动合并。</li>
<li>每人一行。两个模型使用完全相同的人与划分，编号、时间、姓名、路径、PPD time及临床随访信息都不是预测变量。</li>
<li>参数事先固定：500棵树、叶子至少5人、每次分叉考虑特征数平方根、平衡类别权重。不搜索最优特征组合、种子或测试阈值。</li>
<li>3个固定种子分别做分层5折。每一折只用训练者学习中位数填补、常数列删除和随机森林；测试者从未参与该次训练。随机森林无需标准化。</li>
<li>每人获得3份OOF预测并取平均。AUC使用这些概率，其他分类指标固定阈值0.5；不报告训练集AUC冒充泛化效果。</li>
</ol><p>最后另用全体纳入者拟合研究模型并保存joblib；它不能再次在这些训练者身上证明预测效果。本结果没有独立外部验证。</p>
<p>861项输入中有{cohort['all_missing_feature_columns']}列在本批纳入者中全部缺失。每个训练折填补后删除常数列，实际进入建树的特征数：{'；'.join(feature_ranges)}。未根据测试AUC筛选特征。</p>
<a href="random_forest_roc.png" target="_blank"><img src="random_forest_roc.png" alt="随机森林OOF ROC曲线"></a>
<a href="random_forest_confusion.png" target="_blank"><img src="random_forest_confusion.png" alt="固定阈值混淆矩阵"></a></section>
<section><h2>现阶段如何理解</h2><p>AUC衡量阳性与阴性的排序区分能力，0.5接近随机，不能等同准确率。单看ACC容易受阴性占多数影响，应结合敏感度、特异度和多数类别基线。</p>
<p>当前结果只支持评估这批数据与这套固定流程的初步可行性。自动质控并不保证伪迹已清除；原始幅度标定、参考信息、有效时长、通道覆盖差异、PPD结局标准和样本排除都可能影响结果。后续需要人工复核和独立受试者验证。没有因为AUC不理想而翻转答案或放宽质控。</p>
<p>{pair_text}本次delta是各时相可用电极汇总值相减，尚不是固定电极全集下的严格生理变化量，也可能反映通道覆盖和质控差异。后续应在共同电极与匹配有效时长下进行敏感性分析，再讨论前后变化的生理含义。</p></section>
<section><h2>复核数据与读代码</h2>
<a href="{code_url}/docs/batch_random_forest_flow.jpg" target="_blank"><img src="{code_url}/docs/batch_random_forest_flow.jpg" alt="从BDF到AUC的代码阅读路线"></a><ul>
<li><a href="file_status_private.csv">逐文件转换/质控状态（本机私有数据）</a></li>
<li><a href="session_status_private.csv">逐记录覆盖情况（本机私有数据）</a></li>
<li><a href="subject_features_private.csv">每位产妇861项特征（本机私有数据）</a></li>
<li><a href="subject_record_coverage_private.csv">纳入者前后有效时长与通道覆盖</a>、<a href="subject_feature_missingness.csv">逐特征缺失比例</a></li>
<li><a href="pair_channel_coverage_private.csv">镇痛前后共同电极覆盖核对</a></li>
<li><a href="subject_exclusions_private.csv">排除编号与原因（本机私有数据）</a></li>
<li><a href="pre_post_delta_oof_predictions_private.csv">主要模型OOF预测</a>、<a href="pre_only_oof_predictions_private.csv">镇痛前模型OOF预测</a></li>
<li><a href="random_forest_protocol.json">建模协议与版本</a>、<a href="random_forest_results.json">完整评价数值</a>、<a href="batch_audit.json">转换审计</a></li>
<li><a href="{code_url}/docs/BATCH_RANDOM_FOREST_GUIDE.md">初学者阅读指南</a>、<a href="{code_url}/tools/train_random_forest.py">随机森林代码</a></li></ul>
<p>方法参考：<a href="https://scikit-learn.org/stable/common_pitfalls.html">scikit-learn防止数据泄漏</a>、<a href="https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html">随机森林</a>、<a href="https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html">ROC-AUC</a>。</p>
</section></main></body></html>"""
    (output / "随机森林探索性报告.html").write_text(document, encoding="utf8")
    markdown = (
        f"# {title}\n\n本次{cohort['input_subjects']}人目录，最终{cohort['included_subjects']}人，"
        f"阳性{cohort['PPD_positive']}人，阴性{cohort['PPD_negative']}人。\n\n"
        "|模型|特征数|OOF AUC|95%区间|ACC|敏感度|特异度|每次划分AUC|\n"
        "|---|---:|---:|---|---:|---:|---:|---|\n"
        + "\n".join(markdown_rows)
        + f"\n\n{interpretation}\n\n3次分层5折、按人独立测试。区间条件于固定OOF预测，未重训。"
        + f"\n\n转换成功{progress['converted']}/{progress['total']}份，特征成功{progress['features_ok']}份。{audit_text}\n"
    )
    (output / "结果摘要.md").write_text(markdown, encoding="utf8")
    print(output / "随机森林探索性报告.html")


if __name__ == "__main__":
    main()
