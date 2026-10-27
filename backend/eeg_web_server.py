# -*- coding: utf-8 -*-
"""只监听本机的 EEG 特征提取网页服务。"""
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
from eeg_feature_engine import AnalysisSettings, analyze_eeg_file  # noqa: E402
from eeg_web_extra import handle_extra, public_files, PUBLIC_SUFFIXES

ANALYSIS_LOCK = threading.Lock()


def json_bytes(obj):
    def clean(value):
        if isinstance(value, dict): return {str(k): clean(v) for k,v in value.items()}
        if isinstance(value, (list,tuple)): return [clean(v) for v in value]
        if isinstance(value, float) and not math.isfinite(value): return None
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
        assets = {'/assets/eeg_app.css': ('eeg_app.css','text/css; charset=utf-8'),
                  '/assets/eeg_app.js': ('eeg_app.js','application/javascript; charset=utf-8')}
        if parsed.path in assets:
            name, mime = assets[parsed.path]
            body = (WEB_FILE.parent/name).read_bytes()
            self.send_response(200); self.send_header('Content-Type',mime)
            self.send_header('Content-Length',str(len(body))); self.send_header('Cache-Control','no-store')
            self.end_headers(); self.wfile.write(body); return
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
            return self.send_json(200, {"ok": True, "service": "local-eeg", "version": "pair-spike-1"})
        match = re.fullmatch(r"/(download|preview)/([A-Za-z0-9_-]+)/([^/]+)", parsed.path)
        if match:
            mode, run_id, raw_name = match.groups()
            name = unquote(raw_name)
            if Path(name).name != name or name in (".", ".."):
                return self.send_json(400, {"error": "无效文件名"})
            run_dir = (OUTPUT_ROOT / run_id).resolve()
            target = (run_dir / name).resolve()
            if not target.is_relative_to(run_dir) or not target.is_relative_to(OUTPUT_ROOT.resolve()) or not target.is_file() or target.suffix.lower() not in PUBLIC_SUFFIXES:
                return self.send_json(404, {"error": "结果文件不存在"})
            size = target.stat().st_size
            self.send_response(200)
            self.send_header("Content-Type", mimetypes.guess_type(target.name)[0] or "application/octet-stream")
            self.send_header("Content-Length", str(size))
            disposition = 'inline' if mode == 'preview' and target.suffix.lower() == '.png' else 'attachment'
            self.send_header("Content-Disposition", disposition + "; filename*=UTF-8''" + quote(target.name))
            self.end_headers()
            with target.open("rb") as f:
                while chunk := f.read(1024 * 1024):
                    self.wfile.write(chunk)
            return
        self.send_json(404, {"error": "页面不存在"})

    def do_POST(self):
        origin = self.headers.get('Origin')
        if origin and urlparse(origin).netloc not in ('127.0.0.1:8765','localhost:8765'):
            self.close_connection = True
            return self.send_json(403, {'error':'请从本机网页地址访问分析接口。'})
        if not ANALYSIS_LOCK.acquire(blocking=False):
            self.close_connection = True
            return self.send_json(409, {'error':'本机正在执行另一项上传或分析，请等待完成后重试。'})
        try:
            if handle_extra(self, OUTPUT_ROOT): return
            return self._single_analysis()
        except (ValueError, KeyError, TypeError) as exc:
            self.close_connection = True
            return self.send_json(400, {'error':str(exc)[:400]})
        except Exception as exc:
            self.close_connection = True
            return self.send_json(500, {'error':f'分析失败：{type(exc).__name__}: {str(exc)[:300]}'})
        finally:
            ANALYSIS_LOCK.release()

    def _single_analysis(self):
        if urlparse(self.path).path != "/api/analyze":
            return self.send_json(404, {"error": "接口不存在"})
        try:
            length = int(self.headers.get("Content-Length", "-1"))
            if length <= 0 or length > MAX_BYTES:
                return self.send_json(413, {"error": "文件为空或超过 2 GB 限制"})
            qs = parse_qs(urlparse(self.path).query)
            filename = Path(unquote(qs.get("filename", ["recording.bdf"])[0])).name
            ext = Path(filename).suffix.lower()
            if ext not in (".bdf", ".edf", ".gdf"):
                return self.send_json(400, {"error": "仅支持 BDF、EDF、GDF 文件"})
            try:
                opts = json.loads(self.headers.get("X-EEG-Settings", "{}"))
            except (ValueError, TypeError):
                return self.send_json(400, {"error": "分析设置格式无效"})
            target_fs = int(opts.get("target_fs", 250))
            epoch_seconds = float(opts.get("epoch_seconds", 2))
            if not 100 <= target_fs <= 2000 or not 1 <= epoch_seconds <= 60:
                return self.send_json(400, {"error": "为实现 45 Hz 上限的带通滤波，目标采样率需为 100–2000 Hz；分段长度需为 1–60 秒"})
            features = set(opts.get("features", ["time", "frequency", "time_frequency", "network"]))
            allowed = {"time", "frequency", "time_frequency", "network"}
            if not features or not features <= allowed:
                return self.send_json(400, {"error": "特征类别设置无效"})
            notch = opts.get("notch", "50")
            notches = () if notch == "none" else (float(notch), float(notch) * 2)
            settings = AnalysisSettings(target_fs=target_fs, epoch_seconds=epoch_seconds, notches=notches)
            run_id = datetime.now().strftime("%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:8]
            run_dir = OUTPUT_ROOT / run_id
            run_dir.mkdir(parents=True, exist_ok=False)
            fd, tmp_name = tempfile.mkstemp(prefix="eeg_", suffix=ext, dir=str(OUTPUT_ROOT))
            try:
                with os.fdopen(fd, "wb") as out:
                    remaining = length
                    while remaining:
                        chunk = self.rfile.read(min(1024 * 1024, remaining))
                        if not chunk:
                            raise ValueError("上传中断，未收到完整文件")
                        out.write(chunk)
                        remaining -= len(chunk)
                result = analyze_eeg_file(
                    tmp_name, run_dir, settings=settings,
                    auto_eeg_only=bool(opts.get("auto_eeg_only", True)),
                    include_time="time" in features,
                    include_frequency="frequency" in features,
                    include_time_frequency="time_frequency" in features,
                    include_network="network" in features,
                    save_preprocessed=bool(opts.get('cache_for_comparison', False)),
                )
            finally:
                try:
                    os.unlink(tmp_name)
                except OSError:
                    pass
            header = result["header"]
            pre = result["preprocessed"]
            metrics = {
                "format": header.get("format", ""),
                "sampling_rate": round(float(header.get("sampling_rate", 0)), 2),
                "analysis_channels": int(len(pre.get("channels", [])) or header.get("analysis_channel_count", 0) or len(header.get("channel_labels", []))),
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
            result['settings'] = {**asdict(settings), 'auto_eeg_only': bool(opts.get('auto_eeg_only',True))}
            (run_dir/'analysis_manifest.json').write_text(json.dumps(result,ensure_ascii=False),encoding='utf-8')
            return self.send_json(200, {"kind":"single", "run_id":run_id, "metrics": metrics, "files": public_files(paths,run_id),
                'warnings':pre.get('notes',[]), 'channels':pre.get('channels',[])})
        except Exception as exc:
            # Avoid returning raw paths or patient metadata in an error message.
            return self.send_json(500, {"error": f"分析出错：{type(exc).__name__}: {str(exc)[:300]}"})


def main():
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
