"""Presentation/export completeness without using patient recordings."""

from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
from eeg_feature_engine import AnalysisSettings
from eeg_thesis_features import extract_collection
from eeg_research_appendix import make_appendix, make_paired_appendix, write_appendix, specification


class ResearchAppendixTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fs = 200
        data = np.random.default_rng(19).normal(size=(3, 1600))
        _, cls.summary, cls.metadata = extract_collection(
            [{"data": data, "fs": fs, "channel_names": ["F3", "F4", "Cz"]}],
            AnalysisSettings(feature_profile="thesis_v1"),
        )

    def test_complete_formula_and_value_coverage(self):
        appendix = make_appendix(self.summary, self.metadata)
        self.assertEqual(len(appendix["rows"]), 287)
        self.assertEqual(sum(appendix["counts"].values()), 287)
        actual = {r["特征"]: r["数值"] for r in appendix["rows"]}
        for row in self.summary.to_dict("records"):
            if np.isnan(row["数值"]):
                self.assertTrue(np.isnan(actual[row["特征"]]))
            else:
                self.assertEqual(actual[row["特征"]], row["数值"])
        self.assertTrue(
            all(
                row["公式或定义"] and row["单位"] and row["探索用途"] and row["对应图"]
                for row in appendix["rows"]
            )
        )
        tf = specification("时频域__alpha_TF_cv__额区")
        self.assertIn("B(t)=Σ", tf["公式或定义"])
        self.assertIn("std_t(B)/mean_t(B)", tf["公式或定义"])
        self.assertIn("C3/C4", specification("频域__FAA__C3_C4")["解释界限"])

    def test_export_preserves_missing_and_escapes_context(self):
        appendix = make_appendix(self.summary, self.metadata, "<script>alert(1)</script>")
        with tempfile.TemporaryDirectory() as folder:
            paths = write_appendix(appendix, folder, "test")
            frame = pd.read_csv(paths[0])
            self.assertEqual(len(frame), 287)
            self.assertTrue(frame.loc[frame["特征"] == "频域__FAA__Fp1_Fp2", "数值"].isna().all())
            html = Path(paths[1]).read_text(encoding="utf8")
            self.assertIn("&lt;script&gt;", html)
            self.assertNotIn("<script>", html)
            self.assertIn("Content-Security-Policy", html)
            self.assertIn("@page{size:A4 landscape", html)
            self.assertIn("时域（84项）", html)
            self.assertIn("网络（30项）", html)

    def test_filtered_domains_are_reported_without_fabricating_rows(self):
        summary = self.summary[self.summary["特征"].str.startswith("时域__")]
        appendix = make_appendix(summary, self.metadata)
        self.assertEqual(len(appendix["rows"]), 84)
        self.assertEqual(appendix["counts"], {"时域": 84})


if __name__ == "__main__":
    unittest.main()
