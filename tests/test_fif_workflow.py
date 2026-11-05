"""Synthetic FIF metadata, rejection boundaries, format isolation and local publication."""

from pathlib import Path
import hashlib
import json
import sys
import tempfile
import unittest
import os
from unittest.mock import patch
import numpy as np

_config = tempfile.TemporaryDirectory()
os.environ["_MNE_FAKE_HOME_DIR"] = _config.name
import mne

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
from eeg_feature_engine import read_eeg_file, preprocess_eeg, AnalysisSettings, analyze_eeg_file
from eeg_fif import review_figures
from eeg_local_fif import register_local, local_source, publish_fif, digest
from eeg_raw_qc import analyze_raw
from eeg_session import _analyze_session
from test_raw_qc import fixture


def make_fif(path, offset=0):
    fs = 200.0
    rng = np.random.default_rng(23 + offset)
    names = ["Fp1", "F3", "F4", "C3", "O1", "O2", "EOG1"]
    values = rng.normal(0, 7e-6, (len(names), 4000))
    t = np.arange(4000) / fs
    values[0] += 2e-3
    values[-1] += 80e-6 * np.exp(-(((t - 9) / 0.15) ** 2))
    raw = mne.io.RawArray(
        values, mne.create_info(names, fs, ["eeg"] * 6 + ["eog"]), first_samp=100, verbose="ERROR"
    )
    raw.info["bads"] = ["Fp1"]
    raw.info["description"] = (
        f"original local clock (timezone unverified): 24.04.24 08.30.{offset:02d}"
    )
    raw.set_annotations(
        mne.Annotations([4, 8, 9], [2, 1, 0], ["BAD_movement", "EYES_closed", "blink"])
    )
    raw.set_montage("standard_1020", on_missing="ignore")
    raw.save(path, fmt="double", overwrite=True, verbose="ERROR")
    raw.close()
    return path


class FifWorkflowTests(unittest.TestCase):
    def test_reader_excludes_saved_bad_channel_and_epochs_touching_bad(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = make_fif(Path(tmp) / "fdata.fif")
            before = digest(path)
            data, labels, units, header, _ = read_eeg_file(path)
            self.assertNotIn("Fp1", labels)
            self.assertNotIn("EOG1", labels)
            self.assertEqual(header.bad_channels, ["Fp1"])
            self.assertAlmostEqual(header.marked_segments[0]["start"], 4)
            self.assertEqual(header.start_time, "08.30.00")
            self.assertEqual(header.start_offset_seconds, 0.5)
            settings = AnalysisSettings(target_fs=200, notches=())
            pre = preprocess_eeg(data, header, settings)
            self.assertNotIn(2, pre["kept_epoch_indices"])
            self.assertEqual(pre["epoch_total"], 10)
            # Marked samples cannot affect the surviving features/filter output.
            changed = data.copy()
            changed[:, 800:1200] = 1e8
            again = preprocess_eeg(changed, header, settings)
            np.testing.assert_allclose(pre["data"], again["data"], atol=0, rtol=0)
            self.assertEqual(before, digest(path))
            files = review_figures(path, tmp)
            self.assertEqual(sum(p.endswith(".png") for p in files), 3)
            self.assertTrue(all(Path(p).is_file() for p in files))

    def test_cropped_export_keeps_annotation_times_relative_to_current_first_sample(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = make_fif(Path(tmp) / "fdata.fif")
            with mne.io.read_raw_fif(path, preload=True, verbose="ERROR") as raw:
                # Explicit export extras describe seconds from the acquisition origin.
                for i, e in enumerate(raw.annotations.extras):
                    e["exact_onset_s"] = float(raw.annotations.onset[i])
                    e["exact_duration_s"] = float(raw.annotations.duration[i])
                raw.crop(tmin=2)
                raw.save(Path(tmp) / "cropped.fif", fmt="double", overwrite=True, verbose="ERROR")
            _, _, _, header, _ = read_eeg_file(Path(tmp) / "cropped.fif")
            self.assertAlmostEqual(header.marked_segments[0]["start"], 2)
            self.assertEqual(header.start_time, "08.30.02")
            self.assertEqual(header.start_offset_seconds, 0.5)

    def test_full_single_fif_pipeline_has_annotation_figures(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = make_fif(root / "input.fif")
            result = analyze_eeg_file(
                path, root / "analysis", settings=AnalysisSettings(target_fs=200, notches=())
            )
            self.assertEqual(result["header"]["format"], "FIF")
            self.assertEqual(
                len(
                    [
                        p
                        for p in result["outputs"]["figures"]
                        if "FIF复核_" in p and p.endswith(".png")
                    ]
                ),
                3,
            )
            self.assertNotIn("Fp1", result["preprocessed"]["channels"])

    def test_mixed_session_formats_rejected_before_any_read(self):
        with self.assertRaisesRegex(ValueError, "禁止混合"):
            _analyze_session([{"path": "a.bdf"}, {"path": "b.fif"}], None, ".", AnalysisSettings())

    def test_local_publish_fdata_and_reversible_backup_with_native_groups(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "data.bdf"
            fixture(source)
            before = digest(source)
            uploads = root / "runs"
            uploads.mkdir()
            choice = register_local(uploads, source)
            self.assertEqual(local_source(uploads, choice["local_token"]), source)
            result = analyze_raw(source, root / "qc", {"bads": ["O2"], "welch_seconds": 2})
            saved = publish_fif(result, source, root / "qc")
            target = root / "fdata.fif"
            self.assertTrue(target.exists())
            self.assertTrue((root / "fdata_group2.fif").exists())
            with mne.io.read_raw_fif(target, preload=False, verbose="ERROR") as raw:
                self.assertIn("O2", raw.info["bads"])
                self.assertEqual(raw.info["sfreq"], 256)
                self.assertIsNotNone(raw.get_montage())
                self.assertEqual(raw.n_times, 3072)
            old = digest(target)
            again = publish_fif(result, source, root / "qc")
            self.assertEqual(digest(Path(again["local_fif"]["backup"]) / "fdata.fif"), old)
            self.assertEqual(digest(source), before)
            self.assertFalse(list(root.glob(".fdata_staging_*")))


if __name__ == "__main__":
    unittest.main()
