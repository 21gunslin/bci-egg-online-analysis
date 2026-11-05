"""原始检视的接口分发：只读载入、波形、PSD、恢复、导出是不同动作，不要把查看当作修改。

Local raw-QC HTTP integration; persistent source is private, never downloadable."""

import importlib.util
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys


def raw_run(root, run_id):
    if not isinstance(run_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]{8,80}", run_id):
        raise ValueError("原始质控编号无效。")
    run = root / run_id
    if (
        not (run / "raw_qc_private.json").is_file()
        or not (run / "source.bdf").is_file()
        or not (run / "raw_qc_result.json").is_file()
    ):
        raise ValueError("原始质控缓存不存在，请重新上传。")
    return run


def mne_python():
    if importlib.util.find_spec("mne") is not None:
        return sys.executable
    root = Path(__file__).resolve().parents[1]
    for base in (root, root / "github/bci-egg-online-analysis"):
        for name in (".venv", ".venv-mne-review"):
            path = base / name / "Scripts/python.exe"
            if path.is_file():
                return str(path)
    raise ValueError("该功能需要MNE-Python。请使用已安装MNE的项目Python环境启动服务。")


def handle_raw_qc(handler, root, route):
    """按URL分派原始数据请求：查看、扫描和正式FIF导出是不同的入口。"""
    if route == "/api/raw-qc/local-source":
        from eeg_web_extra import read_json
        from eeg_local_fif import pick_bdf, register_local

        opts = read_json(handler)
        path = opts.get("path") or pick_bdf()
        handler.send_json(200, register_local(root, path) if path else {"cancelled": True})
        return True
    if route in ("/api/raw-qc/patrol", "/api/raw-qc/review", "/api/raw-qc/review-export"):
        from eeg_patrol_api import handle_patrol

        return handle_patrol(handler, root, route)
    if route not in (
        "/api/raw-qc",
        "/api/raw-qc/waveform",
        "/api/raw-qc/restore",
        "/api/raw-qc/view",
        "/api/raw-qc/open",
        "/api/raw-qc/psd",
    ):
        return False
    from eeg_web_extra import read_json, public_files, new_run, upload_path

    opts = read_json(handler)
    if route == "/api/raw-qc/psd":
        from eeg_view_spectrum import inspect_spectrum, export_spectrum

        run = raw_run(root, opts.get("run_id"))
        result = inspect_spectrum(run / "source.bdf", opts)
        if opts.get("export", False):
            run_id, output = new_run(root)
            result["files"] = public_files([export_spectrum(result, output)], run_id)
        handler.send_json(200, result)
        return True
    if route == "/api/raw-qc/open":
        from eeg_raw_bdf import NativeBDF
        from eeg_raw_display import ordered_channels
        from eeg_raw_qc import assemble_mapping
        from eeg_viewing import viewing_defaults
        from eeg_local_fif import local_source

        local_path = local_source(root, opts["local_token"]) if opts.get("local_token") else None
        original = local_path or upload_path(root, opts.get("upload_id"))
        source = NativeBDF(original)
        events = source.annotations()
        metadata = None
        if opts.get("recordinfo_upload"):
            metadata = json.loads(
                upload_path(root, opts["recordinfo_upload"], ".json").read_text(encoding="utf8")
            )
            if metadata.get("kind") != "recordinfo":
                raise ValueError("需要RecordInfo类型的配置。")
        mapping = assemble_mapping(source, metadata)
        order = ordered_channels([r["name"] for r in source.signals], mapping)
        lookup = {r["name"]: r for r in source.signals}
        defaults = viewing_defaults(original, mapping)
        run_id, run = new_run(root)
        shutil.copyfile(original, run / "source.bdf")
        result = {
            "kind": "raw_qc",
            "view_only_import": True,
            "duration": source.duration,
            "files": [],
            "montage_info": mapping,
            "channels": [
                {**r, "fs": lookup[r["name"]]["fs"], "unit": lookup[r["name"]]["unit"]}
                for r in order
            ],
            "summary": [
                {
                    "模式": "仅载入查看",
                    "原始信号通道": len(source.signals),
                    "时长_s": source.duration,
                    "滤波/删道删段": "无",
                    "FIF": "未生成",
                }
            ],
            "quality": [],
            "checks": [],
            "segments": [],
            "segment_total": 0,
            "alpha": [],
            "annotations": [
                {
                    "起点_s": e["onset"],
                    "持续_s": e["duration"],
                    "标签": e["description"],
                    "来源": "BDF",
                }
                for e in events[:1000]
            ],
            "options": {
                "bads": [],
                "bad_segments": [],
                "eyes": [],
                "overrides": [],
                "welch_seconds": 4,
                "segment_seconds": 30,
            },
            "view_defaults": defaults,
            "warnings": [
                "只读载入：仅估计查看参数，未运行完整PSD质控、特征提取或α分析，也未导出FIF。横纵轴调整不改动源数据。"
            ],
        }
        (run / "raw_qc_private.json").write_text(
            json.dumps(
                {
                    "source": str(run / "source.bdf"),
                    "original_source": str(local_path) if local_path else None,
                    "output": str(run),
                    "recordinfo": metadata,
                    "options": result["options"],
                },
                ensure_ascii=False,
            ),
            encoding="utf8",
        )
        (run / "raw_qc_result.json").write_text(
            json.dumps(result, ensure_ascii=False), encoding="utf8"
        )
        if not local_path:
            original.unlink(missing_ok=True)
        result["run_id"] = run_id
        result["original_source"] = str(local_path) if local_path else None
        handler.send_json(200, result)
        return True
    if route == "/api/raw-qc/restore":
        run_id = opts.get("run_id")
        if not run_id:
            candidates = sorted(
                root.glob("*/raw_qc_result.json"), key=lambda p: p.stat().st_mtime, reverse=True
            )
            for candidate in candidates:
                result = json.loads(candidate.read_text(encoding="utf8"))
                if result.get("kind") == "raw_qc" and (candidate.parent / "source.bdf").is_file():
                    run_id = candidate.parent.name
                    break
        if not run_id:
            raise ValueError("本机没有可恢复的原始质控结果，请先上传一份BDF。")
        run = raw_run(root, run_id)
        result = json.loads((run / "raw_qc_result.json").read_text(encoding="utf8"))
        from eeg_viewing import viewing_defaults
        from eeg_raw_display import ordered_channels

        native = {c["name"]: c for c in result["channels"]}
        result["channels"] = [
            {**native[r["name"]], **r}
            for r in ordered_channels(list(native), result["montage_info"])
        ]
        result["view_defaults"] = viewing_defaults(run / "source.bdf", result["montage_info"])
        result["original_source"] = json.loads(
            (run / "raw_qc_private.json").read_text(encoding="utf8")
        ).get("original_source")
        result["run_id"] = run_id
        result["files"] = public_files(result["files"], run_id)
        result["view_only_restored"] = True
        handler.send_json(200, result)
        return True
    if route == "/api/raw-qc/view":
        from eeg_raw_bdf import waveform
        from eeg_viewing import export_view

        run = raw_run(root, opts.get("run_id"))
        result = json.loads((run / "raw_qc_result.json").read_text(encoding="utf8"))
        view = waveform(run / "source.bdf", opts, mapping=result["montage_info"])
        run_id, output = new_run(root)
        bad_segments = [
            {"start": r["起点_s"], "end": r["起点_s"] + r["持续_s"]}
            for r in result.get("annotations", [])
            if r["标签"].lower().startswith("bad")
        ]
        paths = export_view(
            view,
            output,
            mode=opts.get("mode", "channel"),
            gain=float(opts.get("gain", 1)),
            fixed_step=float(opts.get("fixed_step", 100)),
            bads=result.get("options", {}).get("bads", []),
            bad_segments=bad_segments,
        )
        handler.send_json(
            200,
            {
                "kind": "raw_view",
                "run_id": run_id,
                "source_run": opts["run_id"],
                "files": public_files(paths, run_id),
                "notice": "仅导出当前坐标视图及参数；未重新分析或写入BDF/FIF。",
            },
        )
        return True
    if route == "/api/raw-qc/waveform":
        from eeg_raw_bdf import waveform

        run = raw_run(root, opts.get("run_id"))
        result = json.loads((run / "raw_qc_result.json").read_text(encoding="utf8"))
        handler.send_json(
            200, waveform(run / "source.bdf", opts, mapping=result.get("montage_info"))
        )
        return True
    recordinfo = None
    if opts.get("source_run"):
        previous = raw_run(root, opts["source_run"])
        original = previous / "source.bdf"
        private = json.loads((previous / "raw_qc_private.json").read_text(encoding="utf8"))
        recordinfo = private.get("recordinfo")
        original_source = private.get("original_source")
        uploaded = False
    else:
        from eeg_local_fif import local_source

        local_path = local_source(root, opts["local_token"]) if opts.get("local_token") else None
        original = local_path or upload_path(root, opts.get("upload_id"))
        uploaded = not bool(local_path)
        original_source = str(local_path) if local_path else None
    if opts.get("source_path"):
        from eeg_local_fif import register_local, local_source

        original_source = str(
            local_source(root, register_local(root, opts["source_path"])["local_token"])
        )
    if not original_source:
        raise ValueError(
            "要保存到BDF同目录，请用“从本机选择BDF”或填写源BDF完整路径；普通浏览器上传不提供原始目录。"
        )
    from eeg_local_fif import digest

    if digest(original) != digest(original_source):
        raise ValueError("填写的本机BDF与当前记录不一致，不能保存到该目录。")
    if opts.get("recordinfo_upload"):
        metadata = upload_path(root, opts["recordinfo_upload"], ".json")
        recordinfo = json.loads(metadata.read_text(encoding="utf8"))
        if recordinfo.get("kind") != "recordinfo":
            raise ValueError("需要RecordInfo类型的配置。")
    options = {
        k: opts[k]
        for k in ("bads", "bad_segments", "eyes", "overrides", "welch_seconds", "segment_seconds")
        if k in opts
    }
    interpreter = mne_python()
    run_id, run = new_run(root)
    shutil.copyfile(original, run / "source.bdf")
    config = {
        "source": str(run / "source.bdf"),
        "original_source": original_source,
        "output": str(run),
        "options": options,
        "recordinfo": recordinfo,
    }
    config_path = run / "raw_qc_private.json"
    config_path.write_text(json.dumps(config, ensure_ascii=False), encoding="utf8")
    result_path = run / "raw_qc_result.json"
    with (run / "raw_qc_worker.log").open("wb") as log:
        process = subprocess.run(
            [
                interpreter,
                str(Path(__file__).with_name("eeg_raw_qc.py")),
                str(config_path),
                str(result_path),
            ],
            stdout=log,
            stderr=log,
            timeout=3600,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    if not result_path.is_file():
        raise ValueError(
            "原始质控未完成，请查看本机raw_qc_worker.log；原上传数据保留，可重新尝试。"
        )
    result = json.loads(result_path.read_text(encoding="utf8"))
    if result.get("error") or process.returncode:
        raise ValueError(result.get("error", "原始质控进程未正常结束。"))
    from eeg_local_fif import publish_fif

    result = publish_fif(result, original_source, run)
    result["original_source"] = original_source
    # Uploaded source is retained until successful export; private run copy supports revisions.
    if uploaded:
        original.unlink(missing_ok=True)
    result["run_id"] = run_id
    result["files"] = public_files(result["files"], run_id)
    handler.send_json(200, result)
    return True
