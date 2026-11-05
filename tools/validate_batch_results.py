"""最后复核：按编号重新核对标签、每次测试的人和AUC；不重训、不调参。"""

import argparse
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
from urllib.parse import unquote, urlparse
from urllib.request import url2pathname

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedKFold


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.urls = []

    def handle_starttag(self, tag, attrs):
        for key, value in attrs:
            if (tag == "a" and key == "href") or (tag == "img" and key == "src"):
                self.urls.append(value)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline", type=Path)
    args = parser.parse_args()
    output = args.output
    checks = []

    def require(condition, description):
        if not condition:
            raise ValueError("复核失败：" + description)
        checks.append(description)

    def read_json(name):
        return json.loads((output / name).read_text(encoding="utf8"))

    table = pd.read_csv(output / "subject_features_private.csv")
    cohort = read_json("cohort_summary.json")
    protocol = read_json("random_forest_protocol.json")
    audit = read_json("batch_audit.json")
    inventory = pd.read_csv(output / "inventory_private.csv")
    require(
        audit["issue_count"] == 0 and audit["files_in_inventory"] == len(inventory),
        "全部输入已有无问题的源文件/FIF核验记录",
    )
    require(
        not table["id"].duplicated().any() and len(table) == cohort["included_subjects"],
        "每位产妇恰好一行，纳入人数一致",
    )
    require(int(table["PPD"].sum()) == cohort["PPD_positive"], "阳性人数与汇总一致")
    conflict_ids = set(inventory.loc[inventory["identity_conflict"], "id"])
    require(not set(table["id"]) & conflict_ids, "身份冲突编号没有进入模型")
    if args.baseline:
        require(
            hashlib.sha256(args.baseline.read_bytes()).hexdigest() == cohort["baseline_sha256"],
            "基线表仍是建立模型时的版本",
        )
        labels = pd.read_excel(args.baseline)[["id", "PPD", "PPD time"]].apply(
            pd.to_numeric, errors="coerce"
        )
        labels = labels.dropna(subset=["id"]).set_index("id")
        matched = labels.loc[table["id"], "PPD"].to_numpy()
        require(
            np.array_equal(matched, table["PPD"].to_numpy()),
            "按编号重配基线后，每个人的PPD答案一致，没有行错位",
        )
        invalid = labels.index[(labels["PPD"] == 1) & (labels["PPD time"] == 0)]
        require(not set(table["id"]) & set(invalid), "PPD=1且PPD time=0的错误记录已排除")

    for metrics in read_json("random_forest_results.json"):
        name = metrics["model"]
        predictions = pd.read_csv(output / f"{name}_oof_predictions_private.csv")
        assignments = pd.read_csv(output / f"{name}_fold_assignments_private.csv")
        require(
            predictions["id"].tolist() == table["id"].tolist(), name + "：OOF编号和建模表逐行一致"
        )
        require(np.array_equal(predictions["PPD"], table["PPD"]), name + "：OOF答案与原表一致")
        columns = [f"oof_probability_repeat_{i + 1}" for i in range(len(protocol["seeds"]))]
        average = predictions[columns].mean(axis=1).to_numpy()
        require(
            np.allclose(average, predictions["oof_probability"], atol=1e-14, rtol=0),
            name + "：每人平均分确为3次OOF概率平均",
        )
        y = table["PPD"].to_numpy(dtype=int)
        probability = predictions["oof_probability"].to_numpy()
        require(
            abs(roc_auc_score(y, probability) - metrics["auc"]) < 1e-12, name + "：独立重算AUC一致"
        )
        require(
            abs(accuracy_score(y, probability >= 0.5) - metrics["accuracy_at_0_5"]) < 1e-12,
            name + "：独立重算ACC一致",
        )
        matrix = confusion_matrix(y, probability >= 0.5, labels=[0, 1]).ravel().tolist()
        require(matrix == metrics["confusion_matrix_tn_fp_fn_tp"], name + "：混淆矩阵一致")
        counts = assignments.groupby(["id", "repeat"]).size()
        require(
            len(counts) == len(table) * len(protocol["seeds"]) and (counts == 1).all(),
            name + "：每人每次恰好参加一次测试",
        )
        for repeat, seed in enumerate(protocol["seeds"], start=1):
            splitter = StratifiedKFold(n_splits=protocol["folds"], shuffle=True, random_state=seed)
            for fold, (train, test) in enumerate(splitter.split(table, y), start=1):
                recorded = assignments[
                    (assignments["repeat"] == repeat) & (assignments["fold"] == fold)
                ]
                require(
                    set(recorded["id"]) == set(table.iloc[test]["id"]),
                    f"{name}：重复{repeat}折{fold}测试者与固定划分一致",
                )
                require(
                    not set(table.iloc[train]["id"]) & set(recorded["id"]),
                    f"{name}：重复{repeat}折{fold}训练者与测试者不重叠",
                )
        saved = joblib.load(output / f"{name}_final_research_model.joblib")
        pipeline = saved["pipeline"]
        require(pipeline.classes_.tolist() == [0, 1], name + "：概率第二列对应PPD=1")
        require(
            len(saved["feature_columns"]) == metrics["input_features"],
            name + "：模型保存的输入列数一致",
        )
        require(
            all(k.startswith(("pre__", "post__", "delta__")) for k in saved["feature_columns"]),
            name + "：预测列没有编号或随访答案",
        )
        forest_parameters = pipeline.named_steps["forest"].get_params()
        require(
            all(forest_parameters[k] == v for k, v in protocol["parameters"].items()),
            name + "：最终模型参数与事先协议一致",
        )

    links = Links()
    links.feed((output / "随机森林探索性报告.html").read_text(encoding="utf8"))
    for url in links.urls:
        parsed = urlparse(url)
        if parsed.scheme in ("https", "http") or url.startswith("#"):
            continue
        path = (
            Path(url2pathname(parsed.path))
            if parsed.scheme == "file"
            else output / unquote(parsed.path)
        )
        require(path.is_file(), "报告本地链接可用：" + path.name)
    result = {
        "passed": True,
        "check_count": len(checks),
        "checks": checks,
        "note": "只复核，未重训、未改标签、未按AUC调参",
    }
    (output / "validation_summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf8"
    )
    print(f"结果复核通过，共{len(checks)}项。")


if __name__ == "__main__":
    main()
