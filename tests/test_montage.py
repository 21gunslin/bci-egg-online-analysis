"""Template provenance, conservative mapping and MNE geometry verification."""

import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
from eeg_montage import montage_mapping, template, make_mne_montage, export_montage
from eeg_montage import catalogue_mapping
from eeg_metadata import decode_metadata
from test_metadata import record_fixture, SECRET


class MontageTests(unittest.TestCase):
    def test_catalogue_covers_observed_generic_labels_without_identity_data(self):
        info = catalogue_mapping()
        self.assertEqual(info["counts"], {"total": 32, "mapped": 21, "unresolved": 11})
        names = {r["原始通道"] for r in info["rows"]}
        self.assertTrue({"T1", "T2", "PG1", "PG2", "EKG", "X1", "X6"}.issubset(names))

    def test_versioned_template_declares_head_metres_not_individual_positions(self):
        model = template()
        self.assertEqual(model["coordinate_frame"], "head")
        self.assertEqual(model["unit"], "m")
        self.assertFalse(model["individual_measured"])
        self.assertEqual(len(model["sha256"]), 64)
        self.assertEqual(len(model["positions"]), 94)
        self.assertLess(model["positions"]["Fp1"][0], 0)
        self.assertGreater(model["positions"]["Fp2"][0], 0)

    def test_legacy_names_map_to_coincident_modern_template_points(self):
        model = template()
        info = montage_mapping(["EEG Fp1-REF", "T3", "T4", "T5", "T6"])
        self.assertEqual([r["标准通道"] for r in info["rows"]], ["Fp1", "T7", "T8", "P7", "P8"])
        for old, new in model["aliases"].items():
            np.testing.assert_allclose(model["positions"][old], model["positions"][new], atol=1e-12)

    def test_uncertain_and_bipolar_channels_have_no_fabricated_coordinates(self):
        labels = ["T1", "T2", "PG1", "PG2", "F3-A1", "unknown"]
        info = montage_mapping(labels)
        for row in info["rows"]:
            self.assertFalse(row["可装配"])
            self.assertIsNone(row["x_m"])
            self.assertIsNone(row["y_m"])
            self.assertIsNone(row["z_m"])

    def test_declared_auxiliary_types_are_not_mapped_as_scalp_eeg(self):
        safe = {
            "channels": [
                {"通道名称": "Fp1", "配置类型": "EOG"},
                {"通道名称": "F3", "配置类型": "EEG"},
            ]
        }
        info = montage_mapping(["Fp1", "F3", "Status", "ECG"], recordinfo=safe, included=["F3"])
        self.assertEqual(info["rows"][0]["MNE类型"], "eog")
        self.assertIsNone(info["rows"][0]["x_m"])
        self.assertEqual(info["rows"][1]["分析状态"], "本次保留")
        self.assertFalse(info["rows"][2]["可装配"])
        self.assertFalse(info["rows"][3]["可装配"])

    def test_duplicate_physical_points_cannot_be_silently_assembled(self):
        rows = montage_mapping(["T3", "T7", "Fp1"])["rows"]
        self.assertFalse(rows[0]["可装配"])
        self.assertFalse(rows[1]["可装配"])
        self.assertIn("重复", rows[0]["映射状态"])
        self.assertTrue(rows[2]["可装配"])

    def test_recordinfo_preview_adds_mapping_without_exposing_private_fields(self):
        safe = decode_metadata(json.dumps(record_fixture()).encode(), "recordinfo")
        self.assertIn("montage_info", safe)
        text = json.dumps(safe)
        for value in (SECRET, "PRIVATE_USER", "PRIVATE_IP", "DO_NOT_EXPORT"):
            self.assertNotIn(value, text)

    def test_csv_and_template_diagram_export(self):
        info = montage_mapping(["Fp1", "Fp2", "F3", "T1"])
        with tempfile.TemporaryDirectory() as folder:
            files = export_montage(info, folder)
            self.assertEqual(len(files), 3)
            self.assertTrue(all(Path(p).stat().st_size > 100 for p in files))
            text = Path(files[1]).read_text(encoding="utf8")
            self.assertIn("不是产妇个体实测", text)

    def test_optional_mne_montage_assembles_original_labels_in_head_frame(self):
        from importlib.util import find_spec

        if find_spec("mne") is None:
            self.skipTest("Optional MNE verification")
        with (
            tempfile.TemporaryDirectory() as folder,
            patch.dict(os.environ, {"_MNE_FAKE_HOME_DIR": folder}),
        ):
            import mne

            names = ["EEG Fp1-REF", "Fp2", "T3", "T5"]
            montage, info = make_mne_montage(names)
            self.assertEqual(montage.ch_names, names)
            self.assertEqual(montage.get_positions()["coord_frame"], "head")
            raw = mne.io.RawArray(
                np.random.default_rng(0).normal(size=(4, 500)),
                mne.create_info(names, 250, "eeg"),
                verbose=False,
            )
            before = raw.get_data().copy()
            reference_before = raw.info["custom_ref_applied"]
            raw.set_montage(montage, on_missing="raise", verbose=False)
            np.testing.assert_array_equal(raw.get_data(), before)
            self.assertEqual(raw.info["custom_ref_applied"], reference_before)
            self.assertEqual(raw.ch_names, names)
            for channel, row in zip(raw.info["chs"], info["rows"]):
                np.testing.assert_allclose(
                    channel["loc"][:3], [row["x_m"], row["y_m"], row["z_m"]], atol=1e-12
                )
            with self.assertRaises(ValueError):
                make_mne_montage(["Fp1", "T1"])
            partial, info = make_mne_montage(["Fp1", "T1"], allow_partial=True)
            self.assertEqual(partial.ch_names, ["Fp1"])


if __name__ == "__main__":
    unittest.main()
