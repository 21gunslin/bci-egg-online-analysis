"""Anatomical display ordering and physically meaningful ROI power averages."""

from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
from eeg_montage import montage_mapping
from eeg_raw_display import ordered_channels
from eeg_occipital_psd import mean_linear_power, build_roi_spectra


class RawDisplayTests(unittest.TestCase):
    def test_front_to_back_display_order_keeps_source_order_unchanged(self):
        source = ["O2", "EKG", "P3", "T3", "Cz", "Fp2", "O1", "F3", "Fp1", "Status"]
        original = source.copy()
        order = ordered_channels(source, montage_mapping(source))
        self.assertEqual(
            [r["name"] for r in order],
            ["Fp1", "Fp2", "F3", "Cz", "T3", "P3", "O1", "O2", "EKG", "Status"],
        )
        self.assertEqual(source, original)
        self.assertTrue(all(not r["eeg"] for r in order[-2:]))

    def test_manual_montage_names_control_display_order(self):
        mapping = {
            "rows": [
                {"原始通道": "EEG1", "标准通道": "O1", "MNE类型": "eeg"},
                {"原始通道": "EEG2", "标准通道": "Fp1", "MNE类型": "eeg"},
            ]
        }
        self.assertEqual(
            [r["name"] for r in ordered_channels(["EEG1", "EEG2"], mapping)], ["EEG2", "EEG1"]
        )

    def test_unmapped_labels_do_not_acquire_a_region_from_their_prefix(self):
        mapping = {
            "rows": [{"原始通道": n, "标准通道": "", "MNE类型": "eeg"} for n in ("PG1", "T1")]
        }
        self.assertTrue(
            all(r["region"] == "位置待核对" for r in ordered_channels(["PG1", "T1"], mapping))
        )

    def test_mean_is_arithmetic_linear_power_not_db_average(self):
        f = np.arange(20.0)
        frequency, power = mean_linear_power([(f, np.ones(20)), (f, np.full(20, 9.0))])
        np.testing.assert_array_equal(power, np.full(20, 5.0))
        self.assertNotAlmostEqual(
            10 * np.log10(power[10]), (10 * np.log10(1) + 10 * np.log10(9)) / 2
        )
        with self.assertRaises(ValueError):
            mean_linear_power([(f, np.ones(20)), (f + 0.1, np.ones(20))])

    def test_time_windows_weighted_within_channels_then_channels_equal(self):
        f = np.arange(20.0)
        full = {"O1": (f, np.ones(20), 256), "O2": (f, np.full(20, 9.0), 256)}
        curves = {
            "O1": [
                ("open", f, np.full(20, 2.0), 1, 1),
                ("open", f, np.full(20, 8.0), 3, 2),
                ("closed", f, np.full(20, 4.0), 1, 3),
            ],
            "O2": [("open", f, np.full(20, 10.0), 2, 1), ("closed", f, np.full(20, 20.0), 8, 3)],
        }
        records, notes = build_roi_spectra(full, curves, [True], [])
        roi = {r["state"]: r for r in records}
        np.testing.assert_allclose(roi["open"]["psd"], 8.25)
        np.testing.assert_allclose(roi["closed"]["psd"], 12.0)
        self.assertEqual(roi["open"]["channels"], roi["closed"]["channels"])
        records, _ = build_roi_spectra(full, curves, [True], ["O2"])
        self.assertTrue(all(r["channels"] == ["O1"] for r in records))

    def test_comparison_requires_common_channels_and_no_eye_labels_are_invented(self):
        f = np.arange(20.0)
        p = np.ones(20)
        full = {"O1": (f, p, 256), "O2": (f, p * 3, 256)}
        curves = {
            "O1": [("open", f, p, 1, 1), ("closed", f, p * 2, 1, 2)],
            "O2": [("open", f, p * 3, 1, 1)],
        }
        rows, _ = build_roi_spectra(full, curves, [True], [])
        self.assertTrue(
            all(r["channels"] == ["O1"] for r in rows if r["state"] in ("open", "closed"))
        )
        rows, _ = build_roi_spectra(full, {}, [], [])
        self.assertEqual([r["state"] for r in rows], ["full_record"])
        rows, notes = build_roi_spectra({"O1": (f, p, 256), "O2": (f, p * 3, 128)}, {}, [], [])
        self.assertEqual(len(rows), 2)
        self.assertTrue(notes)


if __name__ == "__main__":
    unittest.main()
