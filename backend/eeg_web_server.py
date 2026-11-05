# -*- coding: utf-8 -*-
"""HTTP服务入口：像接线员一样接收网页请求，调用计算函数，再把JSON和图片链接送回网页。

只监听本机的 EEG 特征提取网页服务。"""
from __future__ import annotations

import json
import mimetypes
import os
import re
import sys
import tempfile
import threading
import math
from dataclasses import asdict
import uuid
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, quote, unquote, urlparse

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
WEB_FILE = ROOT / "web" / "eeg_feature_app.html"
OUTPUT_ROOT = ROOT / "results" / "eeg_web_output"
MAX_BYTES = 2 * 1024 * 1024 * 1024
sys.path.insert(0, str(SCRIPT_DIR))
from eeg_feature_engine import (
    AnalysisSettings,
    analyze_eeg_file,
    PIPELINE_VERSION,
    EEGQualityError,
)  # noqa: E402
from eeg_web_extra import handle_extra, public_files, PUBLIC_SUFFIXES, upload_path, pack_event_info
from eeg_events import analyze_events

ANALYSIS_LOCK = threading.Lock()


def json_bytes(obj):
    def clean(value):
        if isinstance(value, dict):
            return {str(k): clean(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [clean(v) for v in value]
        if isinstance(value, float) and not math.isfinite(value):
            return None
        return value

    return json.dumps(clean(obj), ensure_ascii=False, allow_nan=False).encode("utf-8")


class Handler(BaseHTTPRequestHandler):
    server_version = "LocalEEG/1.0"

    def log_message(self, fmt, *args):
        # Do not log URLs or uploaded filenames, which may contain patient identifiers.
        return

    def send_json(self, status, obj):
        body = json_bytes(obj)
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        parsed = urlparse(self.path)
        assets = {
            "/assets/eeg_app.css": ("eeg_app.css", "text/css; charset=utf-8"),
            "/assets/eeg_ui_common.js": (
                "eeg_ui_common.js",
                "application/javascript; charset=utf-8",
            ),
            "/assets/eeg_events_ui.js": (
                "eeg_events_ui.js",
                "application/javascript; charset=utf-8",
            ),
            "/assets/eeg_single_analysis.js": (
                "eeg_single_analysis.js",
                "application/javascript; charset=utf-8",
            ),
            "/assets/eeg_pair_analysis.js": (
                "eeg_pair_analysis.js",
                "application/javascript; charset=utf-8",
            ),
            "/assets/eeg_spike_ui.js": ("eeg_spike_ui.js", "application/javascript; charset=utf-8"),
            "/assets/eeg_session_ui.js": (
                "eeg_session_ui.js",
                "application/javascript; charset=utf-8",
            ),
            "/assets/eeg_result_explainer.js": (
                "eeg_result_explainer.js",
                "application/javascript; charset=utf-8",
            ),
            "/assets/eeg_raw_qc.js": ("eeg_raw_qc.js", "application/javascript; charset=utf-8"),
            "/assets/eeg_view_axes.js": (
                "eeg_view_axes.js",
                "application/javascript; charset=utf-8",
            ),
            "/assets/eeg_artifact_patrol.js": (
                "eeg_artifact_patrol.js",
                "application/javascript; charset=utf-8",
            ),
            "/assets/eeg_patrol_math.js": (
                "eeg_patrol_math.js",
                "application/javascript; charset=utf-8",
            ),
            "/assets/eeg_view_spectrum.js": (
                "eeg_view_spectrum.js",
                "application/javascript; charset=utf-8",
            ),
            "/assets/eeg_folder_import.js": (
                "eeg_folder_import.js",
                "application/javascript; charset=utf-8",
            ),
            "/assets/eeg_app.js": ("eeg_app.js", "application/javascript; charset=utf-8"),
        }
        if parsed.path in assets:
            name, mime = assets[parsed.path]
            body = (WEB_FILE.parent / name).read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)
            return
        if parsed.path in ("/", "/index.html"):
            try:
                body = WEB_FILE.read_bytes()
            except OSError:
                return self.send_json(500, {"error": "找不到网页文件"})
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)
            return
        if parsed.path == "/health":
            return self.send_json(
                200, {"ok": True, "service": "local-eeg", "version": PIPELINE_VERSION}
            )
        if parsed.path == "/api/montage":
            if not ANALYSIS_LOCK.acquire(blocking=False):
                return self.send_json(409, {"error": "当前正在分析，请完成后再生成Montage模板图。"})
            try:
                from eeg_montage import catalogue_mapping, export_montage

                info = catalogue_mapping()
                run_id = "montage_standard_1020_v1"
                files = export_montage(info, OUTPUT_ROOT / run_id)
                return self.send_json(
                    200,
                    {"kind": "montage", "montage_info": info, "files": public_files(files, run_id)},
                )
            finally:
                ANALYSIS_LOCK.release()
        match = re.fullmatch(r"/(download|preview)/([A-Za-z0-9_-]+)/([^/]+)", parsed.path)
        if match:
            mode, run_id, raw_name = match.groups()
            name = unquote(raw_name)
            if Path(name).name != name or name in (".", ".."):
                return self.send_json(400, {"error": "无效文件名"})
            run_dir = (OUTPUT_ROOT / run_id).resolve()
            target = (run_dir / name).resolve()
            if (
                not target.is_relative_to(run_dir)
                or not target.is_relative_to(OUTPUT_ROOT.resolve())
                or not target.is_file()
                or target.suffix.lower() not in PUBLIC_SUFFIXES
            ):
                return self.send_json(404, {"error": "结果文件不存在"})
            size = target.stat().st_size
            self.send_response(200)
            self.send_header(
                "Content-Type", mimetypes.guess_type(target.name)[0] or "application/octet-stream"
            )
            self.send_header("Content-Length", str(size))
            disposition = (
                "inline" if mode == "preview" and target.suffix.lower() == ".png" else "attachment"
            )
            self.send_header(
                "Content-Disposition", disposition + "; filename*=UTF-8''" + quote(target.name)
            )
            self.end_headers()
            with target.open("rb") as f:
                while chunk := f.read(1024 * 1024):
                    self.wfile.write(chunk)
            return
        self.send_json(404, {"error": "页面不存在"})

    def do_POST(self):
        origin = self.headers.get("Origin")
        if origin and urlparse(origin).netloc not in ("127.0.0.1:8765", "localhost:8765"):
            self.close_connection = True
            return self.send_json(403, {"error": "请从本机网页地址访问分析接口。"})
        if not ANALYSIS_LOCK.acquire(blocking=False):
            self.close_connection = True
            return self.send_json(
                409, {"error": "本机正在执行另一项上传或分析，请等待完成后重试。"}
            )
        try:
            if handle_extra(self, OUTPUT_ROOT):
                return
            return self._single_analysis()
        except (ValueError, KeyError, TypeError) as exc:
            self.close_connection = True
            return self.send_json(400, {"error": str(exc)[:400]})
        except Exception as exc:
            self.close_connection = True
            return self.send_json(
                500, {"error": f"分析失败：{type(exc).__name__}: {str(exc)[:300]}"}
            )
        finally:
            ANALYSIS_LOCK.release()

    # 主线第③站：接住网页 single() 发来的 POST /api/analyze。
    # 输入 = 请求体里的整个脑电文件 + 查询串 filename + 请求头 X-EEG-Settings 里的分析参数。
    # 输出 = send_json 返回的 JSON（run_id、metrics、files 下载链接等），交给网页 render()。
    # 本函数只管“上传、参数、接口、下载”，真正的脑电计算全部交给 analyze_eeg_file()。
    def _single_analysis(self):
        if urlparse(self.path).path != "/api/analyze":
            return self.send_json(404, {"error": "接口不存在"})
        try:
            # ① 校验文件：大小（≤2 GB）与扩展名（仅 .bdf/.edf/.gdf）。
            length = int(self.headers.get("Content-Length", "-1"))
            if length <= 0 or length > MAX_BYTES:
                return self.send_json(413, {"error": "文件为空或超过 2 GB 限制"})
            qs = parse_qs(urlparse(self.path).query)
            filename = Path(unquote(qs.get("filename", ["recording.bdf"])[0])).name
            ext = Path(filename).suffix.lower()
            if ext not in (".bdf", ".fif"):
                return self.send_json(400, {"error": "EEG入口只支持BDF或FIF，请先勾选一种格式"})
            # ② 读取并校验网页传来的参数（采样率、片段长度、陷波、特征类别）。
            try:
                opts = json.loads(self.headers.get("X-EEG-Settings", "{}"))
            except (ValueError, TypeError):
                return self.send_json(400, {"error": "分析设置格式无效"})
            input_format = opts.get("input_format", "bdf")
            if input_format not in ("bdf", "fif") or ext != "." + input_format:
                return self.send_json(
                    400, {"error": "文件扩展名与勾选格式不一致；每次只能使用一种格式。"}
                )
            target_fs = int(opts.get("target_fs", 250))
            epoch_seconds = float(opts.get("epoch_seconds", 2))
            if not 100 <= target_fs <= 2000 or not 1 <= epoch_seconds <= 60:
                return self.send_json(
                    400,
                    {
                        "error": "为实现 45 Hz 上限的带通滤波，目标采样率需为 100–2000 Hz；分段长度需为 1–60 秒"
                    },
                )
            features = set(opts.get("features", ["time", "frequency", "time_frequency", "network"]))
            allowed = {"time", "frequency", "time_frequency", "network"}
            if not features or not features <= allowed:
                return self.send_json(400, {"error": "特征类别设置无效"})
            # 把网页参数装进引擎配置 AnalysisSettings（notch=none 则不陷波，否则陷 50 Hz 及其谐波 100 Hz）。
            notch = opts.get("notch", "50")
            notches = () if notch == "none" else (float(notch), float(notch) * 2)
            settings = AnalysisSettings(
                target_fs=target_fs,
                epoch_seconds=epoch_seconds,
                notches=notches,
                feature_profile=opts.get("feature_profile", "standard"),
            )
            # ③ 为本次分析建立输出目录（输出根目录/时间戳_runid），并把上传流分块写入临时文件。
            run_id = datetime.now().strftime("%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:8]
            run_dir = OUTPUT_ROOT / run_id
            run_dir.mkdir(parents=True, exist_ok=False)
            fd, tmp_name = tempfile.mkstemp(prefix="eeg_", suffix=ext, dir=str(OUTPUT_ROOT))
            evt_path = None
            try:
                with os.fdopen(fd, "wb") as out:
                    remaining = length
                    while remaining:
                        chunk = self.rfile.read(min(1024 * 1024, remaining))
                        if not chunk:
                            raise ValueError("上传中断，未收到完整文件")
                        out.write(chunk)
                        remaining -= len(chunk)
                # ④ 核心：调用分析总入口，按需启用四类特征；结果落在 run_dir。
                try:
                    result = analyze_eeg_file(
                        tmp_name,
                        run_dir,
                        settings=settings,
                        auto_eeg_only=bool(opts.get("auto_eeg_only", True)),
                        include_time="time" in features,
                        include_frequency="frequency" in features,
                        include_time_frequency="time_frequency" in features,
                        include_network="network" in features,
                        save_preprocessed=bool(opts.get("cache_for_comparison", False)),
                    )
                except EEGQualityError as error:
                    if ext != ".fif" or opts.get("cache_for_comparison"):
                        raise
                    from eeg_fif import review_figures, open_fif

                    paths = review_figures(tmp_name, run_dir)
                    raw = open_fif(tmp_name)
                    try:
                        metrics = {
                            "format": "FIF",
                            "sampling_rate": float(raw.info["sfreq"]),
                            "analysis_channels": sum(
                                k == "eeg" and n not in raw.info["bads"]
                                for k, n in zip(raw.get_channel_types(), raw.ch_names)
                            ),
                            "duration_seconds": raw.n_times / raw.info["sfreq"],
                            "total_epochs": int(raw.n_times / raw.info["sfreq"] / epoch_seconds),
                            "kept_epochs": 0,
                            "keep_ratio": 0.0,
                        }
                    finally:
                        raw.close()
                    return self.send_json(
                        200,
                        {
                            "kind": "single",
                            "review_only": True,
                            "run_id": run_id,
                            "metrics": metrics,
                            "files": public_files(paths, run_id),
                            "warnings": [
                                "FIF质控未通过，仅展示原始标记与EOG供复核，未生成特征："
                                + str(error)
                            ],
                        },
                    )
                # ⑤ 若网页勾选了事件标记，再把 evt.bdf 与分析结果按时间对齐。
                if opts.get("evt_upload"):
                    evt_path = upload_path(OUTPUT_ROOT, opts["evt_upload"])
                    result["event_info"] = analyze_events(
                        evt_path,
                        result["header"],
                        run_dir,
                        result["provenance"],
                        result["preprocessed"]["fs"],
                        opts.get("evt_offset", 0),
                    )
                    result["outputs"]["event_files"] = result["event_info"]["files"]
            finally:
                if evt_path:
                    evt_path.unlink(missing_ok=True)
                try:
                    os.unlink(tmp_name)
                except OSError:
                    pass
            # ⑥ 从引擎结果里挑出网页要显示的指标（采样率、通道数、时长、片段保留率），
            #    并把输出文件打包成可下载的公开链接，最后 send_json 返回给 render()。
            header = result["header"]
            pre = result["preprocessed"]
            metrics = {
                "format": header.get("format", ""),
                "sampling_rate": round(float(header.get("sampling_rate", 0)), 2),
                "analysis_channels": int(
                    len(pre.get("channels", []))
                    or header.get("analysis_channel_count", 0)
                    or len(header.get("channel_labels", []))
                ),
                "duration_seconds": round(float(header.get("duration_seconds", 0)), 2),
                "total_epochs": int(pre.get("epoch_total", 0)),
                "kept_epochs": int(pre.get("epoch_count", 0)),
                "keep_ratio": float(pre.get("keep_ratio", 0)),
            }
            paths = []
            for path in result["outputs"].values():
                items = path if isinstance(path, (list, tuple)) else [path]
                for p in items:
                    p = Path(p)
                    if p.is_file():
                        paths.append(str(p))
            result["settings"] = {
                **asdict(settings),
                "auto_eeg_only": bool(opts.get("auto_eeg_only", True)),
            }
            (run_dir / "analysis_manifest.json").write_text(
                json.dumps(result, ensure_ascii=False), encoding="utf-8"
            )
            return self.send_json(
                200,
                {
                    "kind": "single",
                    "run_id": run_id,
                    "metrics": metrics,
                    "files": public_files(paths, run_id),
                    "feature_definition": result.get("feature_definition"),
                    "feature_coverage": result.get("feature_coverage"),
                    "rows": result.get("rows", []),
                    "research_appendix": result.get("research_appendix"),
                    "montage_info": result.get("montage_info"),
                    "warnings": pre.get("notes", []),
                    "channels": pre.get("channels", []),
                    "event_info": pack_event_info(result.get("event_info"), run_id),
                },
            )
        except Exception as exc:
            # Avoid returning raw paths or patient metadata in an error message.
            return self.send_json(
                500, {"error": f"分析出错：{type(exc).__name__}: {str(exc)[:300]}"}
            )


def main():
    """命令行入口：解析启动参数后调用主流程，不在这里编写分析公式。"""
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    server = ThreadingHTTPServer(("127.0.0.1", 8765), Handler)
    server.daemon_threads = True
    print("脑电特征网页已启动：http://127.0.0.1:8765/", flush=True)
    print("此服务只接受本机连接。关闭窗口或按 Ctrl+C 可停止。", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("正在停止本地服务…", flush=True)
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
