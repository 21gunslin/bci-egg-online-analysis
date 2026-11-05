"""巡检接口：接收网页扫描/复核进度请求。原始文件和人工进度分开保存。

Endpoints for triage candidates and explicit manual-review coverage."""

import json
from pathlib import Path

import pandas as pd


def handle_patrol(handler, root, route):
    """把扫描、人工复核进度和进度CSV导出分派给巡检模块。"""
    if route not in ("/api/raw-qc/patrol", "/api/raw-qc/review", "/api/raw-qc/review-export"):
        return False
    from eeg_web_extra import read_json, public_files, new_run
    from eeg_raw_qc_api import raw_run
    from eeg_raw_bdf import NativeBDF
    from eeg_artifact_patrol import scan_record, export_scan, review_state, VERSION

    opts = read_json(handler)
    run = raw_run(root, opts.get("run_id"))
    source = NativeBDF(run / "source.bdf")
    if route == "/api/raw-qc/review":
        handler.send_json(200, review_state(run, source, opts))
        return True
    if route == "/api/raw-qc/review-export":
        state = review_state(run, source, {})
        run_id, output = new_run(root)
        csv = output / "人工复核进度_非BAD标记.csv"
        pd.DataFrame(
            [
                {
                    "通道": name,
                    "起点_s": r["start"],
                    "终点_s": r["end"],
                    "人工状态": "已复核" if r["status"] == "reviewed" else "需再看",
                    "说明": "仅人工浏览记录，不是正常判定，不自动转换为BAD",
                }
                for r in state["entries"]
                for name in r["channels"]
            ]
            or [{"说明": "尚无明确人工复核记录"}]
        ).to_csv(csv, index=False, encoding="utf-8-sig")
        handler.send_json(200, {"files": public_files([csv], run_id)})
        return True
    result = json.loads((run / "raw_qc_result.json").read_text(encoding="utf8"))
    settings = {
        "seconds": float(opts.get("seconds", 2)),
        "sensitivity": float(opts.get("sensitivity", 6)),
        "scope": opts.get("scope", "eeg"),
    }
    signature = {
        "version": VERSION,
        "settings": settings,
        "source_size": source.path.stat().st_size,
        "source_mtime_ns": source.path.stat().st_mtime_ns,
    }
    cache = run / "artifact_patrol_cache.json"
    previous = json.loads(cache.read_text(encoding="utf8")) if cache.is_file() else {}
    if previous.get("signature") == signature:
        scan = previous["scan"]
    else:
        scan = scan_record(source.path, result["montage_info"], **settings)
        temporary = cache.with_suffix(".tmp")
        temporary.write_text(
            json.dumps({"signature": signature, "scan": scan}, ensure_ascii=False, allow_nan=False),
            encoding="utf8",
        )
        temporary.replace(cache)
    csv = export_scan(scan, run)
    handler.send_json(
        200,
        {
            **scan,
            "files": public_files([csv], opts["run_id"]),
            "review": review_state(run, source, {}),
        },
    )
    return True
