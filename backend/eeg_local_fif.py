"""本机文件选择和同目录保存：先核对源BDF，再保存fdata.fif；旧版先备份。

Explicit local BDF selection and reversible same-directory FIF publication."""

import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import uuid
from datetime import datetime


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""):
            h.update(block)
    return h.hexdigest()


def register_local(root, path):
    from eeg_raw_bdf import NativeBDF

    source = Path(str(path).strip().strip('"')).expanduser().resolve()
    if source.suffix.lower() != ".bdf" or not source.is_file():
        raise ValueError("请选择实际存在的本机BDF完整路径。")
    NativeBDF(source)
    folder = root / "_uploads"
    folder.mkdir(exist_ok=True)
    token = uuid.uuid4().hex
    (folder / (token + ".local.json")).write_text(
        json.dumps({"path": str(source), "sha256": digest(source)}, ensure_ascii=False),
        encoding="utf8",
    )
    return {
        "local_token": token,
        "source_path": str(source),
        "target_path": str(source.parent / "fdata.fif"),
    }


def local_source(root, token):
    """根据本机选择令牌找回源BDF，并重新比对SHA-256，防止选错或分析期间文件被替换。"""
    if not isinstance(token, str) or not re.fullmatch(r"[a-f0-9]{32}", token):
        raise ValueError("本机源文件编号无效，请重新选择。")
    path = root / "_uploads" / (token + ".local.json")
    if not path.is_file():
        raise ValueError("本机源文件选择已失效，请重新选择。")
    info = json.loads(path.read_text(encoding="utf8"))
    source = Path(info["path"])
    if not source.is_file() or digest(source) != info["sha256"]:
        raise ValueError("选择后的源BDF已变化，请重新选择核对。")
    return source


def pick_bdf():
    # Separate GUI process avoids Tk thread affinity in the threaded HTTP server.
    code = "import tkinter as t;from tkinter import filedialog;r=t.Tk();r.withdraw();r.attributes('-topmost',True);p=filedialog.askopenfilename(title='选择原始BDF，同目录导出fdata.fif',filetypes=[('BDF','*.bdf')]);print(p);r.destroy()"
    import sys

    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        encoding="utf8",
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        timeout=300,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    if result.returncode:
        raise ValueError("本机文件选择窗口不可用，请在页面粘贴源BDF完整路径。")
    return result.stdout.strip()


def publish_fif(result, source, run):
    """把核验后的主EEG组保存为源BDF旁的fdata.fif，旧版先备份；其他采样率组另存。"""
    from eeg_mne_runtime import load_mne

    mne = load_mne()
    source = Path(source).resolve()
    parent = source.parent
    run = Path(run)
    fif = [Path(p) for p in result["files"] if Path(p).suffix.lower() == ".fif"]
    # Each primary group owns its split files; only open group primaries.
    primaries = [
        p for p in fif if re.fullmatch(r"group\d+_.+Hz_raw\.fif|fdata(?:_group\d+)?\.fif", p.name)
    ]
    if not primaries:
        raise ValueError("没有可保存的FIF采样率组。")
    stage = parent / (".fdata_staging_" + uuid.uuid4().hex)
    stage.mkdir()
    committed = []
    backup = None
    try:
        groups = []

        # Prefer the group containing the most EEG channels as fdata.fif.
        def eeg_count(path):
            raw = mne.io.read_raw_fif(path, preload=False, verbose="ERROR")
            try:
                return raw.get_channel_types().count("eeg")
            finally:
                raw.close()

        primaries.sort(key=eeg_count, reverse=True)
        for i, path in enumerate(primaries, 1):
            raw = mne.io.read_raw_fif(path, preload=False, verbose="ERROR")
            try:
                name = "fdata.fif" if i == 1 else f"fdata_group{i}.fif"
                raw.save(
                    stage / name,
                    fmt="double",
                    overwrite=False,
                    proj=False,
                    drop_small_buffer=False,
                    verbose="ERROR",
                )
                check = mne.io.read_raw_fif(stage / name, preload=False, verbose="ERROR")
                try:
                    if (
                        check.n_times != raw.n_times
                        or check.ch_names != raw.ch_names
                        or check.info["bads"] != raw.info["bads"]
                    ):
                        raise ValueError("同目录FIF回读核验失败。")
                    import numpy as np

                    if check.get_channel_types() != raw.get_channel_types() or list(
                        check.annotations.description
                    ) != list(raw.annotations.description):
                        raise ValueError("同目录FIF类型或注释回读不一致。")
                    if [dict(e) for e in check.annotations.extras] != [
                        dict(e) for e in raw.annotations.extras
                    ]:
                        raise ValueError("同目录FIF精确注释回读不一致。")
                    for a, b in zip(raw.info["chs"], check.info["chs"]):
                        if a["unit"] != b["unit"] or not np.allclose(
                            a["loc"], b["loc"], atol=1e-8, equal_nan=True
                        ):
                            raise ValueError("同目录FIF坐标或单位回读不一致。")
                    for start in range(0, raw.n_times, 100000):
                        if not np.array_equal(
                            raw.get_data(start=start, stop=start + 100000),
                            check.get_data(start=start, stop=start + 100000),
                        ):
                            raise ValueError("同目录FIF样本回读不一致。")
                    groups.append(
                        {
                            "file": name,
                            "sampling_rate": check.info["sfreq"],
                            "channels": check.ch_names,
                            "bads": check.info["bads"],
                        }
                    )
                finally:
                    check.close()
            finally:
                raw.close()
        info = {
            "source_sha256": result["source_sha256"],
            "groups": groups,
            "note": "fdata.fif为主EEG组；其他原采样率组另存，不上采样。",
        }
        (stage / "fdata_export.json").write_text(
            json.dumps(info, ensure_ascii=False, indent=2), encoding="utf8"
        )
        staged_fifs = [p for p in stage.iterdir() if p.suffix.lower() == ".fif"]
        for path in staged_fifs:
            shutil.copyfile(path, run / path.name)
        existing = [
            p
            for p in parent.iterdir()
            if p.is_file()
            and (
                re.fullmatch(r"fdata(?:_group\d+)?(?:-\d+)?\.fif", p.name)
                or p.name == "fdata_export.json"
            )
        ]
        if existing:
            backup = (
                parent
                / ".fdata_versions"
                / (datetime.now().strftime("%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:6])
            )
            backup.mkdir(parents=True)
            for path in existing:
                shutil.copy2(path, backup / path.name)
        # Verify the source was not replaced during analysis before publishing.
        if digest(source) != result["source_sha256"]:
            raise ValueError("分析期间源BDF发生变化，同目录保存已取消。")
        staged = list(stage.iterdir())
        new_names = {p.name for p in staged}
        try:
            for path in staged:
                target = parent / path.name
                os.replace(path, target)
                committed.append(target)
            for path in existing:
                if path.name not in new_names:
                    path.unlink()
        except Exception:
            for path in committed:
                path.unlink(missing_ok=True)
            if backup:
                for path in backup.iterdir():
                    shutil.copy2(path, parent / path.name)
            raise
        result["local_fif"] = {
            "path": str(parent / "fdata.fif"),
            "files": [str(parent / g["file"]) for g in groups],
            "backup": str(backup) if backup else None,
            "groups": groups,
        }
        result["files"] = [p for p in result["files"] if Path(p).suffix.lower() != ".fif"] + [
            str(run / p.name) for p in staged_fifs
        ]
        result["warnings"].append(
            "FIF已保存到源BDF同目录：" + str(parent / "fdata.fif") + "；旧版同名导出已保留备份。"
        )
        if len(groups) > 1:
            result["warnings"].append(
                "存在不同原采样率：主EEG组为fdata.fif，其他组为fdata_groupN.fif，不自动混合或上采样。"
            )
        (run / "raw_qc_result.json").write_text(
            json.dumps(
                result,
                ensure_ascii=False,
                default=lambda v: v.item() if hasattr(v, "item") else str(v),
            ),
            encoding="utf8",
        )
        return result
    finally:
        # Only remove our checked, UUID-named staging directory.
        if stage.parent == parent and stage.name.startswith(".fdata_staging_"):
            shutil.rmtree(stage)
