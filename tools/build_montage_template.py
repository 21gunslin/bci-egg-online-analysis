"""Generate a frozen MNE 10–20 template in HEAD coordinates (metres).

Run with requirements-mne.txt installed. No EEG recordings are loaded.
"""

from importlib.metadata import distribution
import json
from pathlib import Path
import shutil


def build(output):
    import mne
    import numpy as np

    requested = "standard_1020"
    available = mne.channels.get_builtin_montages()
    # MNE 1.13 renamed this same coordinate set; avoid the deprecated alias.
    resolved = "colin27_1020" if "colin27_1020" in available else requested
    native = mne.channels.make_standard_montage(resolved, head_size="auto")
    transform = mne.channels.compute_native_head_t(native)
    head = native.copy()
    head.apply_trans(transform)
    positions = head.get_positions()
    assert positions["coord_frame"] == "head"
    aliases = {"T3": "T7", "T4": "T8", "T5": "P7", "T6": "P8"}
    for old, new in aliases.items():
        # Verify legacy/modern points coincide in this exact template.
        np.testing.assert_allclose(positions["ch_pos"][old], positions["ch_pos"][new], atol=1e-12)
    info = mne.create_info(head.ch_names, 250, "eeg", verbose=False)
    info.set_montage(native, match_case=True, match_alias=False, on_missing="raise", verbose=False)
    for channel in info["chs"]:
        np.testing.assert_allclose(
            channel["loc"][:3], positions["ch_pos"][channel["ch_name"]], atol=1e-12
        )
    result = {
        "schema_version": 1,
        "requested_kind": requested,
        "resolved_kind": resolved,
        "mne_version": mne.__version__,
        "head_size": "auto",
        "source_frame": native.get_positions()["coord_frame"],
        "coordinate_frame": "head",
        "unit": "m",
        "individual_measured": False,
        "source_url": "https://mne.tools/stable/generated/mne.channels.make_standard_montage.html",
        "generator": "tools/build_montage_template.py",
        "aliases": aliases,
        "transform_native_to_head": transform["trans"].tolist(),
        "fiducials": {k: positions[k].tolist() for k in ("nasion", "lpa", "rpa")},
        "positions": {k: v.tolist() for k, v in positions["ch_pos"].items()},
        "default_channels": [
            "Fp1",
            "Fp2",
            "F7",
            "F3",
            "Fz",
            "F4",
            "F8",
            "T3",
            "C3",
            "Cz",
            "C4",
            "T4",
            "T5",
            "P3",
            "Pz",
            "P4",
            "T6",
            "O1",
            "O2",
            "T1",
            "T2",
            "PG1",
            "PG2",
            "A1",
            "A2",
        ],
    }
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf8")
    for file in distribution("mne").files:
        if str(file).endswith("licenses/LICENSE.txt"):
            shutil.copyfile(
                distribution("mne").locate_file(file), output.parent / "LICENSE_MNE.txt"
            )
            break
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default=str(
            Path(__file__).resolve().parents[1] / "backend/data/montage_standard_1020_head.json"
        ),
    )
    args = parser.parse_args()
    result = build(args.output)
    print(
        json.dumps(
            {
                "template": result["resolved_kind"],
                "mne": result["mne_version"],
                "electrodes": len(result["positions"]),
                "frame": result["coordinate_frame"],
            }
        )
    )
