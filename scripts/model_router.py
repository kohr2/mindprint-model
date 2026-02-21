#!/usr/bin/env python3
"""
Single-URL OpenAI-compatible router for MLX models.

Routes /v1/chat/completions calls by requested model name to an underlying
mlx_lm server, loading/switching models on demand so callers can use one URL.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import threading
import time
from typing import Dict, Optional, Tuple

import requests
from fastapi import FastAPI, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse, Response, StreamingResponse


ROUTER_HOST = os.getenv("MODEL_ROUTER_HOST", "0.0.0.0")
ROUTER_PORT = int(os.getenv("MODEL_ROUTER_PORT", "8000"))
UPSTREAM_HOST = os.getenv("MODEL_ROUTER_UPSTREAM_HOST", "127.0.0.1")
UPSTREAM_PORT = int(os.getenv("MODEL_ROUTER_UPSTREAM_PORT", "8090"))
UPSTREAM_BASE = f"http://{UPSTREAM_HOST}:{UPSTREAM_PORT}"
MODEL_LOAD_TIMEOUT_S = float(os.getenv("MODEL_ROUTER_LOAD_TIMEOUT_S", "240"))
UPSTREAM_CONNECT_TIMEOUT_S = float(os.getenv("MODEL_ROUTER_CONNECT_TIMEOUT_S", "10"))
UPSTREAM_READ_TIMEOUT_S = float(os.getenv("MODEL_ROUTER_READ_TIMEOUT_S", "240"))
UPSTREAM_HEALTHCHECK_TIMEOUT_S = float(
    os.getenv("MODEL_ROUTER_HEALTHCHECK_TIMEOUT_S", "8")
)

DEFAULT_MODEL = os.getenv("MODEL_ROUTER_DEFAULT_MODEL", "qwen_transcripts_on_curriculum")

# Model aliases for single-URL switching.
MODEL_MAP: Dict[str, str] = {
    "qwen": "Qwen/Qwen2.5-7B-Instruct",
    "gemma": "google/gemma-3-12b-it",
    "qwen_transcripts": "/Users/memetica-studio/mindprint-model/output/merged/transcripts_20260217_165256",
    "qwen_curriculum": "/Users/memetica-studio/mindprint-model/output/merged/curriculum_20260217_165256",
    "qwen_transcripts_on_curriculum": "/Users/memetica-studio/mindprint-model/output/merged/transcripts_on_curriculum_20260217_165256",
    # Backward compatibility alias (prefer qwen_transcripts_on_curriculum)
    "transcripts_on_curriculum": "/Users/memetica-studio/mindprint-model/output/merged/transcripts_on_curriculum_20260217_165256",
}

# Optional extra aliases via env:
# MODEL_ROUTER_ALIASES_JSON='{"curriculum":".../output/merged/curriculum_xxx"}'
_extra_aliases = os.getenv("MODEL_ROUTER_ALIASES_JSON", "").strip()
if _extra_aliases:
    try:
        parsed = json.loads(_extra_aliases)
        if isinstance(parsed, dict):
            for k, v in parsed.items():
                if isinstance(k, str) and isinstance(v, str) and k.strip() and v.strip():
                    MODEL_MAP[k.strip()] = v.strip()
    except Exception as exc:
        print(f"[model-router] invalid MODEL_ROUTER_ALIASES_JSON: {exc}", flush=True)

# Also allow full model identifiers directly.
for _v in list(MODEL_MAP.values()):
    MODEL_MAP[_v] = _v

_lock = threading.Lock()
_current_model_key: Optional[str] = None
_upstream_proc: Optional[subprocess.Popen] = None

app = FastAPI(title="Mindprint Model Router")


def _log(msg: str) -> None:
    print(f"[model-router] {msg}", flush=True)


def _stop_upstream() -> None:
    global _upstream_proc, _current_model_key
    if _upstream_proc is None:
        _current_model_key = None
        return
    if _upstream_proc.poll() is None:
        _log(f"stopping upstream pid={_upstream_proc.pid}")
        _upstream_proc.terminate()
        try:
            _upstream_proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            _upstream_proc.kill()
            _upstream_proc.wait(timeout=5)
    _upstream_proc = None
    _current_model_key = None


def _healthcheck(timeout_s: float = 2.0) -> bool:
    try:
        r = requests.get(f"{UPSTREAM_BASE}/v1/models", timeout=timeout_s)
        return r.status_code == 200
    except Exception:
        return False


def _start_upstream(model_path: str) -> subprocess.Popen:
    cmd = [
        "python3",
        "-m",
        "mlx_lm",
        "server",
        "--model",
        model_path,
        "--host",
        UPSTREAM_HOST,
        "--port",
        str(UPSTREAM_PORT),
    ]
    _log(f"starting upstream: {' '.join(cmd)}")
    return subprocess.Popen(cmd)


def _ensure_model(model_key: str) -> Tuple[bool, str]:
    global _upstream_proc, _current_model_key
    model_path = MODEL_MAP.get(model_key)
    if not model_path:
        return False, f"Unknown model alias: {model_key}"

    with _lock:
        if (
            _upstream_proc is not None
            and _upstream_proc.poll() is None
            and _current_model_key == model_key
            and _healthcheck(timeout_s=UPSTREAM_HEALTHCHECK_TIMEOUT_S)
        ):
            return True, model_path

        _stop_upstream()
        _upstream_proc = _start_upstream(model_path)
        _current_model_key = model_key

        deadline = time.time() + MODEL_LOAD_TIMEOUT_S
        while time.time() < deadline:
            if _upstream_proc.poll() is not None:
                return False, f"upstream exited with code {_upstream_proc.returncode}"
            if _healthcheck(timeout_s=UPSTREAM_HEALTHCHECK_TIMEOUT_S):
                _log(f"upstream ready for model={model_key}")
                return True, model_path
            time.sleep(1.0)

        _stop_upstream()
        return False, f"timed out loading model={model_key}"


def _pick_model_key(payload: Dict) -> str:
    model = payload.get("model")
    if isinstance(model, str) and model.strip():
        m = model.strip()
        return m if m in MODEL_MAP else m
    return DEFAULT_MODEL


def _request_timeout() -> Tuple[float, float]:
    return (UPSTREAM_CONNECT_TIMEOUT_S, UPSTREAM_READ_TIMEOUT_S)


def _post_upstream(payload: Dict, stream: bool = False) -> requests.Response:
    return requests.post(
        f"{UPSTREAM_BASE}/v1/chat/completions",
        json=payload,
        stream=stream,
        timeout=_request_timeout(),
    )


@app.get("/health")
def health() -> JSONResponse:
    return JSONResponse(
        {
            "ok": True,
            "current_model": _current_model_key,
            "upstream_alive": bool(_upstream_proc and _upstream_proc.poll() is None),
        }
    )


@app.get("/v1/models")
def models() -> JSONResponse:
    data = [
        {"id": k, "object": "model", "owned_by": "mindprint"}
        for k in sorted({k for k in MODEL_MAP.keys() if "/" not in k})
    ]
    return JSONResponse({"object": "list", "data": data})


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    try:
        payload = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON body"}, status_code=400)

    model_key = _pick_model_key(payload)
    ok, detail = await run_in_threadpool(_ensure_model, model_key)
    if not ok:
        return JSONResponse({"error": detail}, status_code=500)

    payload = dict(payload)
    payload["model"] = detail

    stream = bool(payload.get("stream"))
    if stream:
        try:
            upstream = await run_in_threadpool(_post_upstream, payload, True)
        except requests.RequestException as exc:
            return JSONResponse(
                {"error": f"Upstream stream request failed: {exc}"},
                status_code=504,
            )
        if upstream.status_code >= 400:
            try:
                err = upstream.json()
            except Exception:
                err = {"error": upstream.text}
            return JSONResponse(err, status_code=upstream.status_code)

        def iter_bytes():
            try:
                for chunk in upstream.iter_content(chunk_size=None):
                    if chunk:
                        yield chunk
            except requests.RequestException as exc:
                _log(f"stream relay error: {exc}")
            finally:
                upstream.close()

        return StreamingResponse(
            iter_bytes(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
        )

    try:
        upstream = await run_in_threadpool(_post_upstream, payload, False)
    except requests.RequestException as exc:
        return JSONResponse(
            {"error": f"Upstream request failed: {exc}"},
            status_code=504,
        )
    return Response(
        content=upstream.content,
        status_code=upstream.status_code,
        media_type="application/json",
    )


def _shutdown(*_args):
    _stop_upstream()
    raise SystemExit(0)


signal.signal(signal.SIGTERM, _shutdown)
signal.signal(signal.SIGINT, _shutdown)


if __name__ == "__main__":
    import uvicorn

    _log(f"router starting on {ROUTER_HOST}:{ROUTER_PORT}, upstream={UPSTREAM_BASE}")
    uvicorn.run(app, host=ROUTER_HOST, port=ROUTER_PORT, log_level="info")
