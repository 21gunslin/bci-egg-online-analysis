"""PSD view matches unfiltered native samples; source and existing FIF stay intact."""

import hashlib
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from scipy.signal import welch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
from eeg_view_spectrum import inspect_spectrum, export_spectrum
from test_raw_qc import fixture
from test_artifact_patrol import replace_signal


class SpectrumViewTests(unittest.TestCase):
    def test_full_native_psd_matches_scipy_and_preserves_mixed_rates_units(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "source.bdf"
            expected = fixture(path)
            before = hashlib.sha256(path.read_bytes()).hexdigest()
            result = inspect_spectrum(
                path,
                {
                    "channels": ["O1", "PG1"],
                    "scope": "full",
                    "welch_seconds": 2,
                    "fmin": 0,
                    "fmax": 128,
                },
            )
            voltage, count = result["curves"]
            f, p = welch(
                np.asarray(expected["O1"], dtype=float),
                fs=256,
                window="hann",
                nperseg=512,
                noverlap=256,
                nfft=512,
                detrend=False,
                scaling="density",
            )
            np.testing.assert_allclose(voltage["frequency"], f)
            np.testing.assert_allclose(voltage["power"], p, rtol=1e-10, atol=1e-10)
            self.assertEqual(voltage["unit"], "µV")
            self.assertEqual(count["unit"], "pcs")
            self.assertEqual(count["nyquist"], 4)
            self.assertEqual(result["summary"][0]["原始样本数"], 3072)
            self.assertEqual(before, hashlib.sha256(path.read_bytes()).hexdigest())

    def test_window_range_flat_input_and_csv_are_explicit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "source.bdf"
            fixture(path)
            replace_signal(path, "O2", np.zeros(3072, dtype=int))
            result = inspect_spectrum(
                path,
                {
                    "channels": ["O2"],
                    "start": 1,
                    "seconds": 2,
                    "welch_seconds": 4,
                    "fmin": 0.5,
                    "fmax": 45,
                },
            )
            c = result["curves"][0]
            self.assertTrue(c["flat"])
            self.assertEqual(c["nperseg"], 512)
            self.assertEqual(c["df"], 0.5)
            self.assertEqual((result["start"], result["stop"]), (1, 3))
            self.assertTrue(all(0.5 <= f <= 45 for f in c["frequency"]))
            self.assertTrue(Path(export_spectrum(result, root)).exists())
            self.assertEqual(list(root.glob("*.fif")), [])
            with self.assertRaises(ValueError):
                inspect_spectrum(path, {"channels": ["O1"], "fmin": 100, "fmax": 10})


if __name__ == "__main__":
    unittest.main()
