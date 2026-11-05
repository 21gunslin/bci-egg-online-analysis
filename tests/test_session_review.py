"""Failed segments are visible without entering statistics or hiding structural errors."""

from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest
import warnings

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
from eeg_feature_engine import (
    AnalysisSettings,
    EEGQualityError,
    extract_features,
    make_example_bdf,
    preprocess_eeg,
    read_eeg_file,
)
from eeg_session import analyze_session
from eeg_session_review import preview_waveform
from test_events import event_file


class FailedSegmentReviewTests(unittest.TestCase):
    def test_preview_preserves_extremes_and_raw_data(self):
        x = np.zeros((3, 20000))
        x[0, 12345] = 1000.0
        x[0, 12346] = -700.0
        original = x.copy()
        header = SimpleNamespace(
            sampling_rate=1000, channel_labels=["F3", "F4", "Cz"], channel_units=["uV"] * 3
        )
        preview = preview_waveform(x, header)
        t, y = preview["lines"][0]
        self.assertLessEqual(len(y), 6000)
        self.assertEqual(y.max(), 1000)
        self.assertEqual(y.min(), -700)
        self.assertEqual(t[y.argmax()], 12.345)
        np.testing.assert_array_equal(x, original)

    def test_real_flat_record_is_visible_but_not_in_features(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bad = make_example_bdf(root / "flat.bdf", seconds=8, fs=200)
            raw = bytearray(bad.read_bytes())
            offset = int(raw[184:192])
            raw[offset:] = bytes(len(raw) - offset)
            bad.write_bytes(raw)
            good = make_example_bdf(root / "good.bdf", seconds=8, fs=200)
            raw = bytearray(good.read_bytes())
            raw[176:184] = b"08.30.12"
            good.write_bytes(raw)
            evt = event_file(
                root / "evt.bdf",
                [b"+2\x14KeyIn:253\x14\0", b"+10\x14KeyIn:253\x14\0", b"+13\x14KeyIn:253\x14\0"],
            )
            settings = AnalysisSettings(feature_profile="thesis_v1", target_fs=200)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                result = analyze_session(
                    [{"path": bad}, {"path": good}], evt, root / "mixed", settings, skip_empty=True
                )
            self.assertEqual(result["metrics"]["selected_files"], 2)
            self.assertEqual(result["metrics"]["files"], 1)
            self.assertEqual(result["metrics"]["recorded_seconds"], 16)
            self.assertEqual(result["metrics"]["effective_seconds"], 8)
            self.assertEqual(result["review_segments"][0]["显示状态"], "质控失败，仅供复核")
            self.assertEqual(result["review_segments"][0]["参与特征汇总"], "否")
            self.assertEqual(result["review_segments"][1]["有效拼接起点_s"], 0)
            self.assertEqual(result["event_info"]["rows"][0]["所属分段"], 1)
            self.assertIsNone(result["event_info"]["rows"][0]["有效拼接时间_s"])
            self.assertEqual(result["event_info"]["rows"][1]["状态"], "记录间隙（未采集/未选择）")
            self.assertEqual(result["event_info"]["rows"][2]["有效拼接时间_s"], 1)
            data, _, _, header, _ = read_eeg_file(good)
            pre = preprocess_eeg(data, header, settings)
            _, summary, _ = extract_features(pre, settings)
            expected = summary.set_index("特征")["数值"]
            actual = {row["特征"]: row["数值"] for row in result["rows"]}
            np.testing.assert_allclose(
                [actual[k] for k in expected.index], expected.values, equal_nan=True
            )
            self.assertTrue(any(Path(p).name.startswith("复核_第1段") for p in result["files"]))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                with self.assertRaises(EEGQualityError):
                    data, _, _, header, _ = read_eeg_file(bad)
                    preprocess_eeg(data, header, settings)
            second_bad = root / "flat_later.bdf"
            raw = bytearray(bad.read_bytes())
            raw[176:184] = b"08.30.12"
            second_bad.write_bytes(raw)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                all_failed = analyze_session(
                    [{"path": bad}, {"path": second_bad}],
                    evt,
                    root / "review_only",
                    settings,
                    skip_empty=True,
                )
            self.assertTrue(all_failed["review_only"])
            self.assertEqual(all_failed["metrics"]["files"], 0)
            self.assertEqual(all_failed["rows"], [])
            self.assertEqual(len(all_failed["review_segments"]), 2)
            self.assertEqual(all_failed["event_info"]["rows"][2]["所属分段"], 2)
            self.assertTrue(
                all(r["有效拼接时间_s"] is None for r in all_failed["event_info"]["rows"])
            )


if __name__ == "__main__":
    unittest.main()
