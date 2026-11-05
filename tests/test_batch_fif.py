"""批量转换必须保留所有信号、单位、样本和注释，且不能覆盖既有未知FIF。"""

from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
from eeg_batch_fif import convert_bdf_to_fif, file_sha256
from eeg_mne_runtime import load_mne
from eeg_raw_bdf import NativeBDF
from test_raw_qc import fixture


class BatchFifTests(unittest.TestCase):
    def test_native_groups_roundtrip_and_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "data.bdf"
            fixture(path)
            before = file_sha256(path)
            result = convert_bdf_to_fif(path)
            self.assertEqual(len(result["groups"]), 2)
            mne = load_mne()
            source = NativeBDF(path)
            lookup = {r["name"]: r for r in source.signals}
            found = []
            for group in result["groups"]:
                with mne.io.read_raw_fif(
                    Path(tmp) / group["file"], preload=False, verbose="ERROR"
                ) as raw:
                    found.extend(raw.ch_names)
                    for i, name in enumerate(raw.ch_names):
                        np.testing.assert_allclose(
                            raw.get_data(picks=[i])[0],
                            source.read(lookup[name]),
                            rtol=2e-15,
                            atol=0,
                        )
                    self.assertEqual(raw.info["bads"], [])
                    self.assertIn("BAD_source_acquisition_skip", raw.annotations.description)
            self.assertEqual(set(found), set(lookup))
            self.assertEqual(before, file_sha256(path))
            self.assertTrue(convert_bdf_to_fif(path)["reused"])
            self.assertFalse(list(Path(tmp).glob(".fdata_batch_tmp_*")))

    def test_unknown_existing_fif_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "data.bdf"
            fixture(path)
            target = Path(tmp) / "fdata.fif"
            target.write_bytes(b"prior manual file")
            with self.assertRaisesRegex(ValueError, "不覆盖"):
                convert_bdf_to_fif(path)
            self.assertEqual(target.read_bytes(), b"prior manual file")


if __name__ == "__main__":
    unittest.main()
