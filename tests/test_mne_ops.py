"""用明确的数值定义检验MNE替换，避免仅仅验证函数“能运行”。"""

from pathlib import Path
import sys
import unittest
import tempfile

import numpy as np
from scipy import signal

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
from eeg_mne_ops import bandpass_zero_phase, notch_zero_phase, welch_psd, power_spectrum_windows
from eeg_readers import read_eeg_file
from eeg_raw_bdf import NativeBDF
from test_raw_qc import fixture


class MneOperationsTests(unittest.TestCase):
    def test_zero_phase_filters_preserve_the_defined_scipy_design(self):
        values = np.random.default_rng(2026).normal(size=(3, 4000))
        fs = 200
        expected = signal.sosfiltfilt(
            signal.butter(4, [0.5, 45], btype="bandpass", fs=fs, output="sos"), values
        )
        np.testing.assert_allclose(
            bandpass_zero_phase(values, fs, 0.5, 45), expected, atol=1e-13, rtol=1e-13
        )
        numerator, denominator = signal.iirnotch(50, Q=30, fs=fs)
        np.testing.assert_allclose(
            notch_zero_phase(values, fs, 50),
            signal.filtfilt(numerator, denominator, values),
            atol=1e-13,
            rtol=1e-13,
        )

    def test_welch_preserves_hann_overlap_units_and_dc_policy(self):
        values = np.random.default_rng(8).normal(size=(3, 4000)) + 50
        for detrend in (False, "constant"):
            f, power = welch_psd(values, fs=200, nperseg=400, noverlap=200, detrend=detrend, axis=1)
            expected_f, expected = signal.welch(
                values, fs=200, nperseg=400, noverlap=200, window="hann", detrend=detrend, axis=1
            )
            np.testing.assert_array_equal(f, expected_f)
            np.testing.assert_allclose(power, expected, atol=1e-10, rtol=1e-11)

    def test_per_window_mne_power_matches_unpadded_stft_density(self):
        values = np.random.default_rng(9).normal(size=(2, 1800))
        f, t, power = power_spectrum_windows(values, 200, 400, 200)
        expected_f, expected_t, complex_values = signal.stft(
            values,
            fs=200,
            window="hann",
            nperseg=400,
            noverlap=200,
            nfft=400,
            detrend="constant",
            boundary=None,
            padded=False,
            scaling="psd",
        )
        expected = np.abs(complex_values) ** 2
        expected[:, 1:-1, :] *= 2
        np.testing.assert_array_equal(f, expected_f)
        np.testing.assert_array_equal(t, expected_t)
        np.testing.assert_allclose(power, expected, atol=1e-14, rtol=1e-11)

    def test_bdf_reader_uses_mne_without_upsampling_slow_auxiliary_channels(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "data.bdf"
            fixture(path)
            data, names, units, header, _ = read_eeg_file(path)
            source = NativeBDF(path)
            lookup = {r["name"]: r for r in source.signals}
            self.assertIn("MNE", header.reader)
            self.assertEqual(header.sampling_rate, 256)
            self.assertNotIn("PG1", names)
            self.assertEqual(data.shape[1], 3072)
            for i, name in enumerate(names):
                np.testing.assert_allclose(
                    data[i], source.read(lookup[name]) * 1e6, atol=1e-10, rtol=1e-12
                )
            self.assertEqual(units, ["µV"] * len(names))


if __name__ == "__main__":
    unittest.main()
