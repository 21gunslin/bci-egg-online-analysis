"""第3步：用随机森林预测PPD，并用从未见过该产妇的预测计算AUC。

建议初学者从main()开始读，再读evaluate_model()，最后读make_model()。
输入只有一张表：每行一位产妇，PPD是答案，pre/post/delta开头的列是脑电特征。
参数提前固定，不根据测试结果反复挑模型，也不把训练集AUC当成效果。
"""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import importlib.metadata
import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")  # 批处理只保存图，不弹出几百个窗口。
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline


# 随机种子使别人重跑时能得到相同划分；不是“挑效果最好的种子”。
SEEDS = (20261103, 20261104, 20261105)
FOLDS = 5
MODEL_PARAMETERS = {
    "n_estimators": 500,  # 建500棵树，让许多树共同投票。
    "max_depth": None,  # 不直接限制树高；由叶子最少人数限制复杂度。
    "min_samples_leaf": 5,  # 一个叶子至少5人，减少只记住个别人的情况。
    "max_features": "sqrt",  # 每次分叉只看部分特征，让不同树有所区别。
    "class_weight": "balanced_subsample",  # 每棵树中适当提高少数类别的权重。
    "n_jobs": 4,  # 同时使用4个CPU线程，不改变统计方法。
}


def save_json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf8")


def make_model(seed):
    """把3步处理装进同一个流水线，保证它们只从训练者学习。"""
    return Pipeline(
        [
            # 缺失电极产生的NaN用训练集的中位数填充，不能偷看测试集。
            # 如果训练集某列全缺失，先填0；下一步会删除这个常数列。
            ("fill_missing", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("drop_constant", VarianceThreshold(threshold=0)),
            ("forest", RandomForestClassifier(random_state=seed, **MODEL_PARAMETERS)),
        ]
    )


def check_table(table):
    """宁愿明确报错，也不让重复人或随访信息悄悄进入预测变量。"""
    if table["id"].isna().any() or table["id"].duplicated().any():
        raise ValueError("每位产妇必须恰好一行；id缺失或重复会造成受试者泄漏")
    if not set(table["PPD"].dropna().unique()) <= {0, 1} or table["PPD"].isna().any():
        raise ValueError("PPD必须为无缺失的0/1标签")
    predictors = [name for name in table if name not in ("id", "PPD")]
    if not predictors or any(
        not name.startswith(("pre__", "post__", "delta__")) for name in predictors
    ):
        raise ValueError("预测列只允许脑电pre__/post__/delta__；不允许PPD time等随访信息")
    if table["PPD"].value_counts().min() < FOLDS or table["PPD"].nunique() != 2:
        raise ValueError("阳性和阴性人数均须至少5人，才能进行分层5折评价")
    return predictors


def auc_interval(y, probability, repeats=2000):
    """按人自助抽样：估计这批固定OOF预测的AUC区间，不重新拟合模型。

    每次分别在阳性和阴性人群里有放回抽样，避免抽到只有一个类别。
    这是内部探索性区间，不包含重训的不确定性，不代替外部验证。
    """
    generator = np.random.default_rng(20261103)
    positive = np.flatnonzero(y == 1)
    negative = np.flatnonzero(y == 0)
    values = []
    for _ in range(repeats):
        selected = np.r_[
            generator.choice(positive, len(positive), replace=True),
            generator.choice(negative, len(negative), replace=True),
        ]
        values.append(roc_auc_score(y[selected], probability[selected]))
    return np.quantile(values, [0.025, 0.975]).tolist()


def prediction_metrics(y, probability):
    """AUC用概率计算；ACC等指标使用预先固定的0.5阈值。"""
    predicted = (probability >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()
    return {
        "auc": float(roc_auc_score(y, probability)),
        "auc_ci95_fixed_oof_bootstrap": auc_interval(y, probability),
        "accuracy_at_0_5": float(accuracy_score(y, predicted)),
        "sensitivity_at_0_5": float(tp / (tp + fn)),
        "specificity_at_0_5": float(tn / (tn + fp)),
        "average_precision": float(average_precision_score(y, probability)),
        "brier_score": float(brier_score_loss(y, probability)),
        "confusion_matrix_tn_fp_fn_tp": [int(tn), int(fp), int(fn), int(tp)],
    }


def evaluate_model(table, columns, name, output):
    """最重要的部分：训练、预测、评价。先遮住答案，再让模型做测试题。"""
    # X是题目表（行=产妇，列=脑电特征），y是0/1标准答案；id只用于核对身份。
    X = table[columns].replace([np.inf, -np.inf], np.nan)
    y = table["PPD"].to_numpy(dtype=int)
    people = table["id"].to_numpy(dtype=int)
    probabilities = np.full((len(SEEDS), len(table)), np.nan)
    fold_rows = []
    assignments = []
    repeat_aucs = []

    for repeat, seed in enumerate(SEEDS):
        # 分层使每折阳性比例尽量接近总体；一行就是一位独立产妇。
        splitter = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=seed)
        for fold, (train_indices, test_indices) in enumerate(splitter.split(X, y), start=1):
            if set(people[train_indices]) & set(people[test_indices]):
                raise ValueError("训练和测试出现同一个人，停止评价")
            model = make_model(seed + fold)

            # fit=只看训练集学习。填补中位数、删常数列、建树全部发生在这里。
            model.fit(X.iloc[train_indices], y[train_indices])

            # predict_proba第2列是PPD=1的估计概率。测试答案未参与上面的学习。
            probability = model.predict_proba(X.iloc[test_indices])[:, 1]
            probabilities[repeat, test_indices] = probability
            forest = model.named_steps["forest"]
            fold_rows.append(
                {
                    "model": name,
                    "repeat": repeat + 1,
                    "seed": seed,
                    "fold": fold,
                    "train_people": len(train_indices),
                    "test_people": len(test_indices),
                    "test_positive": int(y[test_indices].sum()),
                    "features_after_training_only_processing": int(forest.n_features_in_),
                    "fold_auc": float(roc_auc_score(y[test_indices], probability)),
                }
            )
            for index, value in zip(test_indices, probability):
                assignments.append(
                    {
                        "id": int(people[index]),
                        "repeat": repeat + 1,
                        "fold": fold,
                        "PPD": int(y[index]),
                        "probability": float(value),
                    }
                )
            print(f"{name}：第{repeat + 1}次划分，第{fold}/{FOLDS}折完成", flush=True)
        if not np.isfinite(probabilities[repeat]).all():
            raise ValueError("有受试者漏掉了测试预测")
        repeat_aucs.append(float(roc_auc_score(y, probabilities[repeat])))

    # 每个人有3次“自己未参加训练”的预测，取平均以减少一次分折的偶然性。
    average_probability = probabilities.mean(axis=0)
    predictions = pd.DataFrame(
        {
            "id": people,
            "PPD": y,
            "oof_probability": average_probability,
            "prediction_threshold_0_5": (average_probability >= 0.5).astype(int),
        }
    )
    for repeat in range(len(SEEDS)):
        predictions[f"oof_probability_repeat_{repeat + 1}"] = probabilities[repeat]
    predictions.to_csv(
        output / f"{name}_oof_predictions_private.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(assignments).to_csv(
        output / f"{name}_fold_assignments_private.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(fold_rows).to_csv(
        output / f"{name}_fold_metrics.csv", index=False, encoding="utf-8-sig"
    )

    metrics = {
        "model": name,
        "people": len(y),
        "positive": int(y.sum()),
        "negative": int((y == 0).sum()),
        "input_features": len(columns),
        "repeat_oof_auc": repeat_aucs,
        **prediction_metrics(y, average_probability),
    }
    save_json(output / f"{name}_metrics.json", metrics)

    # 评价结束后，才用全部人训练一个供以后研究调用的模型。
    # 它见过所有这些人的答案，不能再用它对这些人预测来声称验证效果。
    final_model = make_model(SEEDS[0])
    final_model.fit(X, y)
    joblib.dump(
        {
            "pipeline": final_model,
            "feature_columns": columns,
            "PPD_positive_class": 1,
            "training_subject_ids": people.tolist(),
            "purpose": "research only; internal validation, not clinical diagnosis",
        },
        output / f"{name}_final_research_model.joblib",
    )
    save_json(output / f"{name}_feature_columns.json", columns)
    return metrics, average_probability


def draw_results(y, results, output):
    """ROC横轴是假阳性率，纵轴是真阳性率；越靠左上通常区分能力越好。"""
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
    labels = {"pre_only": "仅镇痛前", "pre_post_delta": "镇痛前＋镇痛后＋差值"}
    for metrics, probability in results:
        fpr, tpr, _ = roc_curve(y, probability)
        ax.plot(fpr, tpr, label=f'{labels[metrics["model"]]}：AUC={metrics["auc"]:.3f}', lw=2)
    ax.plot([0, 1], [0, 1], "--", color="gray", label="随机排序参考：AUC=0.5")
    ax.set(
        xlabel="假阳性率（1−特异度）",
        ylabel="真阳性率（敏感度）",
        xlim=(0, 1),
        ylim=(0, 1.02),
        title="随机森林：按产妇独立测试的重复5折OOF ROC",
    )
    ax.legend(loc="lower right")
    fig.savefig(output / "random_forest_roc.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6), constrained_layout=True)
    for ax, (metrics, probability) in zip(axes, results):
        matrix = confusion_matrix(y, probability >= 0.5, labels=[0, 1])
        ax.imshow(matrix, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(matrix[i, j]), ha="center", va="center", fontsize=18)
        ax.set(
            xticks=[0, 1],
            yticks=[0, 1],
            xticklabels=["预测0", "预测1"],
            yticklabels=["实际0", "实际1"],
            title=labels[metrics["model"]] + "（阈值0.5）",
        )
    fig.savefig(output / "random_forest_confusion.png", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="前两步的结果目录")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    table = pd.read_csv(args.output / "subject_features_private.csv")
    columns = check_table(table)
    if len(columns) != 861:
        raise ValueError("本方案预期287×3=861项脑电特征，请核对第二步")
    y = table["PPD"].to_numpy(dtype=int)
    # 两个模型使用完全相同的人与分折，便于看加入镇痛后记录是否有帮助。
    specifications = {
        "pre_only": [c for c in columns if c.startswith("pre__")],
        "pre_post_delta": columns,
    }
    protocol = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "models": {k: len(v) for k, v in specifications.items()},
        "primary_model": "pre_post_delta",
        "secondary_model": "pre_only",
        "parameters": MODEL_PARAMETERS,
        "seeds": SEEDS,
        "folds": FOLDS,
        "tuning": "none; fixed parameters, no AUC-based feature selection or threshold search",
        "evaluation": "mean of three held-out probabilities per person; repeated stratified 5-fold CV",
        "preprocessing": "training-fold median imputation, then training-fold constant removal, then RF",
        "ci": "2000 stratified participant bootstrap draws of fixed mean OOF probabilities; no model refit",
        "majority_class_accuracy": float(max(np.mean(y == 0), np.mean(y == 1))),
        "chance_roc_auc": 0.5,
        "positive_prevalence": float(y.mean()),
        "versions": {
            package: importlib.metadata.version(package)
            for package in ["mne", "numpy", "scipy", "pandas", "scikit-learn", "joblib"]
        },
    }
    # 记录实际运行代码的文件指纹，方便以后确认是哪一版代码得到的结果。
    project = Path(__file__).resolve().parents[1]
    protocol["code_sha256"] = {}
    for relative in [
        "tools/train_random_forest.py",
        "tools/aggregate_subject_features.py",
        "tools/run_bdf_fif_features.py",
        "backend/eeg_batch_fif.py",
        "backend/eeg_thesis_features.py",
        "backend/eeg_preprocessing.py",
        "backend/eeg_readers.py",
        "backend/eeg_mne_ops.py",
    ]:
        protocol["code_sha256"][relative] = hashlib.sha256(
            (project / relative).read_bytes()
        ).hexdigest()
    # 在训练开始前留下协议，防止看完AUC后悄悄改口径。
    save_json(args.output / "random_forest_protocol.json", protocol)
    results = []
    for name, feature_columns in specifications.items():
        results.append(evaluate_model(table, feature_columns, name, args.output))
    draw_results(y, results, args.output)
    save_json(args.output / "random_forest_results.json", [metrics for metrics, _ in results])
    pd.DataFrame([metrics for metrics, _ in results]).to_csv(
        args.output / "random_forest_metrics.csv", index=False, encoding="utf-8-sig"
    )
    print(json.dumps([metrics for metrics, _ in results], ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
