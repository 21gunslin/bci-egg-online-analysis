"""验证病例排除、每人一行与训练折内填补，避免“好看的AUC”来自泄漏。"""

from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from aggregate_subject_features import build_subject_table, merge_one_record
import train_random_forest as training


class SubjectTableTests(unittest.TestCase):
    def test_record_merge_aligns_names_instead_of_channel_numbers(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rows = []
            for part, names, amplitudes in [
                (1, ["Fp1", "Fp2", "C3", "C4"], [1, 2, 4, 8]),
                (2, ["C4", "Fp2", "Fp1", "C3"], [8, 2, 3, 4]),
            ]:
                row = {
                    "id": 1,
                    "session_key": "s1",
                    "timestamp": "test",
                    "phase": "pre",
                    "identity_conflict": False,
                    "part": part,
                    "file_key": f"p{part}",
                }
                rows.append(row)
                directory = root / "files" / row["file_key"]
                directory.mkdir(parents=True)
                (directory / "result.json").write_text('{"conversion":"ok","features_status":"ok"}')
                wave = np.sin(2 * np.pi * 10 * np.arange(2000) / 250)
                np.savez(
                    directory / "prepared.npz",
                    data=np.asarray(amplitudes)[:, None] * wave,
                    names=names,
                    fs=250,
                    kept=np.arange(4),
                    epoch_samples=500,
                )
            result = merge_one_record(rows, root)
            self.assertEqual(result["status"], "ok")
            self.assertEqual(result["clean_seconds"], 16)
            channels = pd.read_csv(root / "sessions/s1/channel_features.csv").set_index("通道")
            self.assertAlmostEqual(channels.loc["Fp1", "时域__rms"], np.sqrt(2), places=10)

    def test_excludes_record_error_identity_conflict_and_missing_phase(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            inventory, sessions = [], []
            for sid in [1, 94, 212, 3]:
                for phase in ["pre", "post"]:
                    key = f"s{sid}_{phase}"
                    inventory.append(
                        {"id": sid, "session_key": key, "identity_conflict": sid == 94}
                    )
                    sessions.append(
                        {
                            "session_key": key,
                            "phase": phase,
                            "status": "failed" if sid == 3 and phase == "post" else "ok",
                        }
                    )
                    directory = root / "sessions" / key
                    directory.mkdir(parents=True)
                    # 真正建表时需要287项；数值可预测，便于核对差值方向。
                    pd.DataFrame(
                        {
                            "特征": [f"f{i}" for i in range(287)],
                            "数值": np.arange(287) + (2 if phase == "post" else 0),
                        }
                    ).to_csv(directory / "features_287.csv", index=False)
            labels = pd.DataFrame(
                {"id": [1, 94, 212, 3], "PPD": [1, 0, 1, 0], "PPD time": [6, 0, 0, 0]}
            ).set_index("id")
            table, exclusions = build_subject_table(pd.DataFrame(inventory), sessions, labels, root)
            self.assertEqual(table["id"].tolist(), [1])
            self.assertEqual(len(table.columns), 863)
            self.assertTrue((table.filter(like="delta__").iloc[0] == 2).all())
            self.assertEqual({row["id"] for row in exclusions}, {94, 212, 3})


class LeakageTests(unittest.TestCase):
    def test_duplicate_people_and_followup_predictor_are_rejected(self):
        table = pd.DataFrame({"id": list(range(10)), "PPD": [0, 1] * 5, "pre__x": np.arange(10)})
        self.assertEqual(training.check_table(table), ["pre__x"])
        duplicated = table.copy()
        duplicated.loc[1, "id"] = duplicated.loc[0, "id"]
        with self.assertRaisesRegex(ValueError, "一行"):
            training.check_table(duplicated)
        table["PPD time"] = 12
        with self.assertRaisesRegex(ValueError, "随访"):
            training.check_table(table)

    def test_imputation_and_constant_removal_do_not_see_test_data(self):
        model = training.make_model(42)
        model.set_params(forest__n_estimators=8, forest__n_jobs=1)
        train = pd.DataFrame(
            {"varying": [0.0, 2.0, np.nan, 4.0], "empty": [np.nan] * 4, "constant": [7.0] * 4}
        )
        model.fit(train, [0, 1, 0, 1])
        # 测试集的极大值、以及训练集全缺失列的新值，都不能改变已学到的中位数。
        test = pd.DataFrame(
            {"varying": [10000.0, np.nan], "empty": [99.0, 99.0], "constant": [8.0, 8.0]}
        )
        model.predict_proba(test)
        self.assertEqual(model.named_steps["fill_missing"].statistics_[0], 2.0)
        self.assertEqual(model.named_steps["forest"].n_features_in_, 1)

    def test_each_person_gets_exactly_one_held_out_prediction_per_repeat(self):
        table = pd.DataFrame(
            {"id": np.arange(20), "PPD": [0, 1] * 10, "pre__x": np.linspace(0, 1, 20)}
        )
        parameters = {**training.MODEL_PARAMETERS, "n_estimators": 8, "n_jobs": 1}
        with (
            tempfile.TemporaryDirectory() as temporary,
            patch.object(training, "SEEDS", (101, 102)),
            patch.object(training, "FOLDS", 2),
            patch.object(training, "MODEL_PARAMETERS", parameters),
        ):
            output = Path(temporary)
            metrics, probabilities = training.evaluate_model(table, ["pre__x"], "test", output)
            assignments = pd.read_csv(output / "test_fold_assignments_private.csv")
            counts = assignments.groupby(["id", "repeat"]).size()
            self.assertEqual(len(counts), 40)
            self.assertTrue((counts == 1).all())
            self.assertEqual(len(probabilities), 20)
            self.assertEqual(metrics["auc"], training.roc_auc_score(table["PPD"], probabilities))


if __name__ == "__main__":
    unittest.main()
