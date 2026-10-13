"""大语言 runtime 检测：探测本机可用的 llama.cpp 推理运行时。

两种受支持的运行时后端：

* ``standalone``  —— 插件目录 ``runtime_config.json`` 指向的独立 ``llama-server``
  可执行文件（由 ``install_runtime.py`` 安装）。
* ``llama-cpp-python`` —— ComfyUI Python 环境内已安装的 ``llama_cpp`` 库
  （含 CUDA 时使用 ``ggml-cuda.dll``）。

``runtime_status()`` 返回结构化检测结果，供插件在启动/执行时记录日志或在界面上
暴露当前可用的推理运行时。
"""
from __future__ import annotations

import importlib.util
import json
import platform
from dataclasses import dataclass
from pathlib import Path


NODE_ROOT = Path(__file__).resolve().parent
RUNTIME_CONFIG_PATH = NODE_ROOT / "runtime_config.json"
LLAMA_CPP_PYTHON_WHEELS_URL = "https://github.com/JamePeng/llama-cpp-python/releases"


def normalize_runtime_platform(system_name: str | None = None) -> str:
    value = (system_name or platform.system()).strip().casefold()
    names = {
        "windows": "windows",
        "linux": "linux",
        "darwin": "macos",
        "macos": "macos",
    }
    if value not in names:
        raise RuntimeError(f"Unsupported operating system: {system_name or platform.system()}")
    return names[value]


@dataclass(frozen=True)
class PythonRuntimeSpec:
    """已检测到的 llama-cpp-python 运行时信息。"""

    backend: str = "llama-cpp-python"
    version: str = "unknown"
    source: str = "ComfyUI Python environment"
    gpu_offload: bool = True
    native_lib: str = ""
    platform_name: str = "unknown"


def _safe_json_config(config_path: Path) -> dict | None:
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        return None
    return payload


def detect_standalone_runtime(
    root: Path = NODE_ROOT,
    config_path: Path | None = None,
) -> dict | None:
    """独立 llama-server 运行时：配置存在且可执行文件在位则返回其信息。"""
    config_path = config_path or (root / RUNTIME_CONFIG_PATH.name)
    config = _safe_json_config(config_path)
    if config is None:
        return None
    executable_value = config.get("executable")
    if not isinstance(executable_value, str) or not executable_value.strip():
        return None
    try:
        executable = (root / Path(executable_value)).resolve()
        executable.relative_to(root.resolve())
    except (ValueError, OSError):
        return None
    if not executable.is_file():
        return None
    return {
        "kind": "standalone",
        "backend": str(config.get("backend", "unknown")),
        "executable": str(executable),
        "platform": str(config.get("platform", "unknown")),
        "source": str(config.get("source", "runtime_config.json")),
    }


def _python_runtime_native_lib() -> str:
    try:
        from llama_cpp import llama_cpp as _lc
    except (ImportError, OSError):
        return ""
    return str(getattr(_lc, "__file__", "") or "")


def _python_gpu_offload() -> bool:
    """是否具备 GPU 卸载能力（CUDA 后端可用）。

    ``llama_supports_gpu_offload()`` 在部分 0.4.x 构建上误报 False，
    因此优先以 ``ggml-cuda.dll`` 是否存在为准（实测该构建确实走 GPU）。
    """
    try:
        import llama_cpp
        lib_dir = Path(llama_cpp.__file__).resolve().parent / "lib"
        if any(p.name.casefold() == "ggml-cuda.dll" for p in lib_dir.glob("*.dll")):
            return True
    except (ImportError, OSError, AttributeError):
        pass
    try:
        from llama_cpp import llama_cpp as _lc
        if hasattr(_lc, "llama_supports_gpu_offload"):
            try:
                return bool(_lc.llama_supports_gpu_offload())
            except Exception:  # noqa: BLE001
                pass
    except (ImportError, OSError):
        pass
    return False


def detect_python_runtime() -> PythonRuntimeSpec | None:
    """llama-cpp-python 运行时：库可导入则返回其信息。"""
    if importlib.util.find_spec("llama_cpp") is None:
        return None
    try:
        import llama_cpp
    except (ImportError, OSError):
        return None
    return PythonRuntimeSpec(
        version=str(getattr(llama_cpp, "__version__", "unknown")),
        gpu_offload=_python_gpu_offload(),
        native_lib=_python_runtime_native_lib(),
        platform_name=normalize_runtime_platform(),
    )


def detect_available_runtimes(
    root: Path = NODE_ROOT,
) -> tuple[list[dict], list[str]]:
    """返回 (可用运行时列表, 警告列表)。"""
    runtimes: list[dict] = []
    warnings: list[str] = []

    standalone = detect_standalone_runtime(root=root)
    if standalone is not None:
        runtimes.append(standalone)
    else:
        warnings.append(
            "独立 llama-server 运行时未就绪（无有效的 runtime_config.json / llama-server.exe）。"
        )

    python_spec = detect_python_runtime()
    if python_spec is not None:
        runtimes.append(
            {
                "kind": "llama-cpp-python",
                "backend": python_spec.backend,
                "version": python_spec.version,
                "gpu_offload": python_spec.gpu_offload,
                "native_lib": python_spec.native_lib,
                "source": python_spec.source,
            }
        )
    else:
        warnings.append(
            "llama-cpp-python 未安装或无法加载。可选预编译 wheel："
            f"{LLAMA_CPP_PYTHON_WHEELS_URL}"
        )

    return runtimes, warnings


def select_runtime(
    root: Path = NODE_ROOT,
) -> tuple[str, dict, list[str]]:
    """选择推理运行时：优先 standalone，其次 llama-cpp-python。

    返回 ``(kind, runtime_info, warnings)``；均不可用时 kind 为 ``"none"``。
    """
    runtimes, warnings = detect_available_runtimes(root=root)
    ordered = {"standalone": 0, "llama-cpp-python": 1}
    runtimes.sort(key=lambda item: ordered.get(item.get("kind"), 99))
    if not runtimes:
        return "none", {}, warnings
    selected = runtimes[0]
    return selected.get("kind", "none"), selected, warnings


def runtime_status(root: Path = NODE_ROOT) -> dict:
    """汇总当前 LLM 推理运行时检测状态（界面/日志可展示）。"""
    runtimes, warnings = detect_available_runtimes(root=root)
    kind, selected, _ = select_runtime(root=root)
    return {
        "runtimes_installed": [item for item in runtimes if item.get("kind") != "none"],
        "runtime_count": len(runtimes),
        "selected_runtime": kind if kind != "none" else None,
        "selected_backend": selected.get("backend") if selected else None,
        "gpu_offload_available": bool(
            any(item.get("gpu_offload") for item in runtimes)
        ),
        "warnings": warnings,
        "ready": kind != "none",
    }
