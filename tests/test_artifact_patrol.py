"""Known synthetic defects, exact coverage ledger, and source/FIF immutability."""

import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
from eeg_raw_bdf import NativeBDF
from eeg_montage import montage_mapping
from eeg_artifact_patrol import scan_record, review_state
from test_raw_qc import fixture


def replace_signal(path, name, values):
    source = NativeBDF(path)
    row = next(r for r in source.signals if r["name"] == name)
    raw = bytearray(path.read_bytes())
    values = np.asarray(values, dtype="i4")
    assert len(values) == source.records * row["spr"]
    for record in range(source.records):
        u = values[record * row["spr"] : (record + 1) * row["spr"]] & 0xFFFFFF
        payload = (
            np.stack([u & 255, (u >> 8) & 255, (u >> 16) & 255], axis=1).astype("u1").tobytes()
        )
        start = source.header_bytes + record * source.record_bytes + row["offset"]
        raw[start : start + len(payload)] = payload
    path.write_bytes(raw)


class PatrolTests(unittest.TestCase):
    def test_known_flat_rail_jump_and_line_are_candidates_not_source_edits(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "data.bdf"
            expected = fixture(path)
            fs = 256
            t = np.arange(12 * fs) / fs
            replace_signal(
                path,
                "O1",
                np.rint(100 * np.sin(2 * np.pi * 50 * t) + 2 * np.sin(2 * np.pi * 10 * t)),
            )
            replace_signal(path, "O2", np.zeros(len(t), dtype=int))
            x = np.array(expected["T1"])
            x[3 * fs + 20] = 4000
            x[6 * fs : 6 * fs + 30] = 8388607
            replace_signal(path, "T1", x)
            mapping = montage_mapping(["O1", "O2", "T1", "Status", "PG1"])
            before = hashlib.sha256(path.read_bytes()).hexdigest()
            result = scan_record(path, mapping)
            reasons = {(c["channel"], c["reason"]) for c in result["candidates"]}
            self.assertIn(("O2", "平线"), reasons)
            self.assertIn(("T1", "数字边界/饱和候选"), reasons)
            self.assertIn(("T1", "瞬时跳变候选"), reasons)
            self.assertIn(("O1", "50Hz强工频候选"), reasons)
            flat = [
                c for c in result["candidates"] if c["channel"] == "O2" and c["reason"] == "平线"
            ]
            self.assertEqual(len(flat), 1)
            self.assertTrue(flat[0]["persistent"])
            self.assertEqual(flat[0]["end"], 12)
            self.assertEqual(before, hashlib.sha256(path.read_bytes()).hexdigest())
            self.assertFalse(any(p.suffix == ".fif" for p in Path(tmp).iterdir()))

    def test_review_is_explicit_persistent_separate_and_undoable(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "data.bdf"
            fixture(path)
            source = NativeBDF(path)
            self.assertEqual(review_state(tmp, source, {})["entries"], [])
            state = review_state(
                tmp,
                source,
                {
                    "action": "append",
                    "channels": ["O1"],
                    "start": 1,
                    "end": 1.5,
                    "status": "uncertain",
                },
            )
            self.assertEqual(state["entries"][0]["end"], 1.5)
            self.assertEqual(review_state(tmp, source, {})["entries"], state["entries"])
            self.assertEqual(review_state(tmp, source, {"action": "undo"})["entries"], [])
            with self.assertRaises(ValueError):
                review_state(
                    tmp, source, {"action": "append", "channels": ["missing"], "start": 0, "end": 1}
                )

    def test_invalid_limits_are_rejected_without_mutation(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "data.bdf"
            fixture(path)
            mapping = montage_mapping(["O1", "O2", "T1", "Status", "PG1"])
            with self.assertRaises(ValueError):
                scan_record(path, mapping, seconds=float("nan"))
            with self.assertRaises(ValueError):
                scan_record(path, mapping, sensitivity=0)


if __name__ == "__main__":
    unittest.main()
