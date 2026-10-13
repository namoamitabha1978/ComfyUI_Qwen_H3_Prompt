"""ComfyUI HTTP 路由：Qwen H3 本地 LLM 运行时的状态检测与安装。

提供三个 UI 按钮后端：
  GET  /qwen_h3/runtime/status          -> 检测当前 runtime 有效路径/后端/版本信息
  POST /qwen_h3/runtime/install         -> 后台安装指定后端 runtime (cuda / vulkan)
  GET  /qwen_h3/runtime/install/status  -> 查询安装进度与结果
"""
from __future__ import annotations

import json
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

# ---- 路径（与 node.py 保持一致） ----
NODE_ROOT = Path(__file__).resolve().parent
RUNTIME_CONFIG = NODE_ROOT / "runtime_config.json"
INSTALL_SCRIPT = NODE_ROOT / "install_runtime.py"

_REGISTERED = False

# ---- 安装状态（进程内单例） ----
_INSTALL_LOCK = threading.Lock()
_INSTALL_STATE: dict[str, Any] = {
    "state": "idle",  # idle | running | done | error
    "backend": None,
    "message": "",
    "started_at": None,
    "finished_at": None,
    "log_path": None,
    "log_tail": "",
}


def _install_log_path() -> Path:
    return NODE_ROOT / "runtime_install_ui.log"


def _tail_log(log_path: Path, max_chars: int = 2500) -> str:
    try:
        data = log_path.read_bytes()[-max_chars:].decode("utf-8", errors="replace")
        return data[-max_chars:]
    except OSError:
        return ""


def _run_install(backend: str) -> None:
    """后台线程：调用 install_runtime.py 安装指定后端。"""
    log_path = _install_log_path()
    started = time.time()
    with _INSTALL_LOCK:
        _INSTALL_STATE.update(
            state="running", backend=backend,
            message=f"正在安装 {backend} runtime，请稍候（需联网下载）...",
            started_at=started, finished_at=None, log_path=str(log_path), log_tail="",
        )
    try:
        proc = subprocess.run(
            [sys.executable, str(INSTALL_SCRIPT), "--backend", backend],
            cwd=str(NODE_ROOT),
            stdout=open(log_path, "w", encoding="utf-8"),
            stderr=subprocess.STDOUT,
            timeout=1200,
        )
        ok = proc.returncode == 0
    except subprocess.TimeoutExpired:
        ok = False
        msg = "安装超时（超过 20 分钟）。"
    except Exception as error:  # noqa: BLE001
        ok = False
        msg = f"安装失败：{type(error).__name__}: {error}"
    else:
        msg = f"{backend} runtime 安装完成（exit={proc.returncode}）。可回到节点重跑工作流。" if ok \
            else f"{backend} runtime 安装失败（exit={proc.returncode}）。详情见日志。"
    with _INSTALL_LOCK:
        _INSTALL_STATE.update(
            state="done" if ok else "error", message=msg, finished_at=time.time(),
            log_tail=_tail_log(log_path),
        )


def _file_info(path: Path) -> dict | None:
    """单个文件的路径 + 大小（MB）；非文件/不可读返回 None。"""
    try:
        if not path.is_file():
            return None
        size_mb = round(path.stat().st_size / 1048576, 2)
    except OSError:
        return None
    return {"path": str(path), "size_mb": size_mb}


def _dir_listing(root: Path) -> list[dict]:
    """递归列出目录下所有文件（相对路径 + 大小）。"""
    out: list[dict] = []
    if not root.is_dir():
        return out
    try:
        for p in sorted(root.rglob("*")):
            if p.is_file():
                try:
                    out.append({"path": str(p.relative_to(root)), "size_mb": round(p.stat().st_size / 1048576, 2)})
                except OSError:
                    pass
    except OSError:
        pass
    return out


def _build_status() -> dict:
    """汇总当前 LLM 运行时检测信息（路径/文件/后端/版本/状态）。"""
    from .runtime import load_runtime_spec  # noqa: PLC0415
    from .runtime_detect import detect_python_runtime, runtime_status  # noqa: PLC0415
    from . import node as _node  # noqa: PLC0415

    summary = runtime_status(root=NODE_ROOT)

    standalone_detail: dict | None = None
    try:
        spec = load_runtime_spec()
        lib_files: list[dict] = []
        for lib_dir in spec.library_dirs:
            lib_files.extend(_dir_listing(Path(lib_dir)))
        standalone_detail = {
            "backend": spec.backend,
            "executable": str(spec.executable),
            "executable_file": _file_info(spec.executable),
            "platform": spec.platform_name,
            "library_dirs": [str(p) for p in spec.library_dirs],
            "library_files": lib_files,
            "n_gpu_layers": spec.n_gpu_layers,
            "fit": spec.fit,
            "fit_target_mib": spec.fit_target_mib,
            "flash_attention": spec.flash_attention,
        }
    except Exception:  # noqa: BLE001
        standalone_detail = None

    py = detect_python_runtime()
    python_detail = None if py is None else {
        "backend": py.backend,
        "version": py.version,
        "gpu_offload": py.gpu_offload,
        "native_lib": py.native_lib,
        "native_lib_file": _file_info(Path(py.native_lib)) if py.native_lib else None,
        "source": py.source,
        "platform": py.platform_name,
    }

    model_dir = _node.MODEL_DIR
    gguf_files = sorted(model_dir.rglob("*.gguf")) if model_dir.is_dir() else []
    models = [
        {"path": str(p), "name": p.name, "size_mb": round(p.stat().st_size / 1048576, 2)}
        for p in gguf_files
    ]

    runtime_root = NODE_ROOT / "runtime"
    config_payload: dict | None = None
    try:
        config_payload = json.loads(RUNTIME_CONFIG.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        config_payload = None

    return {
        **summary,
        "standalone": standalone_detail,
        "llama_cpp_python": python_detail,
        "paths": {
            "node_root": str(NODE_ROOT),
            "runtime_config": str(RUNTIME_CONFIG) if RUNTIME_CONFIG.exists() else None,
            "install_script": str(INSTALL_SCRIPT) if INSTALL_SCRIPT.exists() else None,
            "runtime_dir": str(runtime_root) if runtime_root.exists() else None,
            "model_dir": str(model_dir) if model_dir.is_dir() else None,
        },
        "runtime_config": config_payload,
        "runtime_files": _dir_listing(runtime_root),
        "models": {"count": len(models), "gguf": models},
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
    }


def _register() -> None:
    global _REGISTERED
    if _REGISTERED:
        return
    server_module = sys.modules.get("server")
    if server_module is None:
        return
    try:
        from aiohttp import web
    except ImportError:
        return
    PromptServer = getattr(server_module, "PromptServer", None)
    if PromptServer is None:
        return
    prompt_server = getattr(PromptServer, "instance", None)
    if prompt_server is None:
        return

    async def status_route(_request: Any) -> Any:
        return web.json_response(_build_status(), headers={"Cache-Control": "no-store"})

    async def install_route(request: Any) -> Any:
        try:
            payload = await request.json()
        except Exception:  # noqa: BLE001
            payload = {}
        backend = str(payload.get("backend", "")).strip().lower()
        if backend not in ("cuda", "vulkan"):
            return web.json_response({"started": False, "error": f"unsupported backend: {backend!r}"}, status=400)
        with _INSTALL_LOCK:
            already = _INSTALL_STATE["state"] == "running"
            if already:
                return web.json_response({"started": False, "error": "already running", "backend": _INSTALL_STATE["backend"]}, status=409)
        threading.Thread(target=_run_install, args=(backend,), daemon=True).start()
        return web.json_response({"started": True, "backend": backend})

    async def install_status_route(_request: Any) -> Any:
        with _INSTALL_LOCK:
            state = dict(_INSTALL_STATE)
        return web.json_response(state, headers={"Cache-Control": "no-store"})

    for method, path, handler in (
        ("get", "/qwen_h3/runtime/status", status_route),
        ("post", "/qwen_h3/runtime/install", install_route),
        ("get", "/qwen_h3/runtime/install/status", install_status_route),
    ):
        try:
            getattr(prompt_server.routes, method)(path)(handler)
        except RuntimeError as error:
            if "already registered" not in str(error).lower():
                raise
    _REGISTERED = True


def register_runtime_api_routes() -> None:
    """幂等注册路由（server 未就绪时静默跳过）。"""
    try:
        _register()
    except Exception:  # noqa: BLE001
        pass


# 供诊断/测试直接查看安装状态
def current_install_state() -> dict:
    with _INSTALL_LOCK:
        return dict(_INSTALL_STATE)
