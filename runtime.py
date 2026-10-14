from __future__ import annotations

import json
import logging
import os
import platform
import socket
import subprocess
import tempfile
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator

from .runtime_detect import (
    PythonRuntimeSpec,
    detect_python_runtime,
    detect_standalone_runtime,
    runtime_status,
)
from .runtime_python import PythonLlamaServer, PythonRuntimeError


LOGGER = logging.getLogger("ComfyUI.QwenH3Prompt.runtime")
NODE_ROOT = Path(__file__).resolve().parent
RUNTIME_CONFIG_PATH = NODE_ROOT / "runtime_config.json"
LLAMA_SEED_MODULUS = 0xFFFFFFFF


def normalize_llama_seed(seed: int) -> int:
    """Map ComfyUI's uint64 seed to llama.cpp's deterministic uint32 range."""
    return int(seed) % LLAMA_SEED_MODULUS


@dataclass(frozen=True)
class RuntimeSpec:
    executable: Path
    library_dirs: tuple[Path, ...]
    platform_name: str
    backend: str
    n_gpu_layers: str
    fit: bool
    fit_target_mib: int
    flash_attention: str


def _configured_path(root: Path, value: object, label: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise RuntimeError(f"Invalid {label} in {RUNTIME_CONFIG_PATH.name}")
    path = (root / Path(value)).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as error:
        raise RuntimeError(
            f"{label} must remain inside the custom node directory: {path}"
        ) from error
    return path


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


def _installer_name(_platform_name: str) -> str:
    return "install_runtime.py"


def load_runtime_spec(
    root: Path = NODE_ROOT,
    system_name: str | None = None,
) -> RuntimeSpec:
    active_platform = normalize_runtime_platform(system_name)
    installer = _installer_name(active_platform)
    config_path = root / RUNTIME_CONFIG_PATH.name
    if config_path.is_file():
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise RuntimeError(
                f"Cannot read {config_path}. Run {installer} --force."
            ) from error
        configured_platform = config.get("platform")
        if config.get("schema_version") != 1:
            raise RuntimeError(
                f"Unsupported runtime configuration in {config_path}. "
                f"Run {installer} --force."
            )
        if configured_platform != active_platform:
            raise RuntimeError(
                f"Runtime configuration is for {configured_platform}, but ComfyUI is "
                f"running on {active_platform}. Run {installer} --force."
            )
        executable = _configured_path(root, config.get("executable"), "executable")
        raw_library_dirs = config.get("library_dirs", [])
        if not isinstance(raw_library_dirs, list):
            raise RuntimeError(f"Invalid library_dirs in {config_path}")
        library_dirs = tuple(
            _configured_path(root, value, "library directory")
            for value in raw_library_dirs
        )
        options = config.get("runtime_options", {})
        if not isinstance(options, dict):
            raise RuntimeError(f"Invalid runtime_options in {config_path}")
        return RuntimeSpec(
            executable=executable,
            library_dirs=library_dirs or (executable.parent,),
            platform_name=active_platform,
            backend=str(config.get("backend", "unknown")),
            n_gpu_layers=str(options.get("n_gpu_layers", "auto")),
            fit=bool(options.get("fit", True)),
            fit_target_mib=int(options.get("fit_target_mib", 1536)),
            flash_attention=str(options.get("flash_attention", "auto")),
        )

    raise RuntimeError(
        f"llama.cpp runtime is not installed for {active_platform}. "
        f"Run {installer} from the custom node directory before starting ComfyUI."
    )


def build_runtime_environment(
    runtime_spec: RuntimeSpec,
    base_environment: dict[str, str] | None = None,
) -> dict[str, str]:
    environment = dict(os.environ if base_environment is None else base_environment)
    library_path = os.pathsep.join(str(path) for path in runtime_spec.library_dirs)
    environment["PATH"] = library_path + os.pathsep + environment.get("PATH", "")
    if runtime_spec.platform_name == "linux":
        environment["LD_LIBRARY_PATH"] = (
            library_path + os.pathsep + environment.get("LD_LIBRARY_PATH", "")
        )
    elif runtime_spec.platform_name == "macos":
        environment["DYLD_LIBRARY_PATH"] = (
            library_path + os.pathsep + environment.get("DYLD_LIBRARY_PATH", "")
        )
    return environment


class LlamaServer:
    def __init__(
        self,
        model: Path,
        mmproj: Path,
        context_size: int = 32768,
        runtime_spec: RuntimeSpec | None = None,
        n_gpu_layers: str | None = None,
        flash_attention: str | None = None,
        fit: bool | None = None,
        fit_target_mib: int | None = None,
    ):
        self.model = model
        self.mmproj = mmproj
        self.context_size = context_size
        self.runtime_spec = runtime_spec or load_runtime_spec()
        # UI 可覆盖 runtime_config.json 里的默认参数；未指定时回退到 spec 值。
        self.n_gpu_layers = (
            n_gpu_layers if n_gpu_layers is not None else self.runtime_spec.n_gpu_layers
        )
        self.flash_attention = (
            flash_attention
            if flash_attention is not None
            else self.runtime_spec.flash_attention
        )
        self.fit = fit if fit is not None else self.runtime_spec.fit
        self.fit_target_mib = (
            fit_target_mib
            if fit_target_mib is not None
            else self.runtime_spec.fit_target_mib
        )
        self.executable = self.runtime_spec.executable
        # Keep the child working directory outside the runtime payload. Some Windows
        # installs inherit restrictive ACLs on the extracted runtime directory even
        # when the executable itself is readable.
        self.root = NODE_ROOT
        self.backend = self.runtime_spec.backend
        self.port = self._free_port()
        self.process: subprocess.Popen | None = None
        self.log = None

    @staticmethod
    def _free_port() -> int:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", 0))
            return sock.getsockname()[1]

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    @property
    def is_running(self) -> bool:
        return self.process is not None and self.process.poll() is None

    def __enter__(self):
        try:
            self.start()
        except Exception:
            self.stop()
            raise
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.stop()

    def _log_tail(self) -> str:
        if self.log is None:
            return ""
        self.log.flush()
        self.log.seek(0)
        text = self.log.read().decode("utf-8", errors="replace")
        return text[-5000:]

    def start(self, timeout: float = 120.0) -> None:
        if not self.executable.is_file():
            installer = _installer_name(self.runtime_spec.platform_name)
            raise FileNotFoundError(
                f"llama.cpp runtime is missing: {self.executable}. "
                f"Run {installer} from the custom node directory."
            )
        self.log = tempfile.TemporaryFile(mode="w+b")  # noqa: SIM115
        arguments = [
            str(self.executable),
            "--model",
            str(self.model),
            "--mmproj",
            str(self.mmproj),
            "--alias",
            "qwen3.8-27b",
            "--api-key",
            "comfyui-local",
            "--host",
            "127.0.0.1",
            "--port",
            str(self.port),
            "--ctx-size",
            str(self.context_size),
            "--parallel",
            "1",
            "--n-gpu-layers",
            self.n_gpu_layers,
            "--flash-attn",
            self.flash_attention,
            "--image-min-tokens",
            "1024",
            "--image-max-tokens",
            "1024",
            "--jinja",
            "--no-webui",
        ]
        # 量化 KV 缓存（q8_0）是支撑 8GB 卡大 context 的关键；但它要求 flash_attn 必须开启，
        # 因此当用户选择 flash_attention=off 时，必须同时去掉量化缓存参数，否则模型加载失败。
        if self.flash_attention != "off":
            arguments.extend(["--cache-type-k", "q8_0", "--cache-type-v", "q8_0"])
        if self.fit:
            arguments.extend(
                [
                    "--fit",
                    "on",
                    "--fit-target",
                    str(self.fit_target_mib),
                ]
            )
        env = build_runtime_environment(self.runtime_spec)
        process_options = {}
        if self.runtime_spec.platform_name == "windows":
            process_options["creationflags"] = getattr(
                subprocess, "CREATE_NO_WINDOW", 0
            )
        self.process = subprocess.Popen(
            arguments,
            cwd=self.root,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=self.log,
            stderr=subprocess.STDOUT,
            **process_options,
        )
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError(f"llama.cpp exited while loading the model.\n{self._log_tail()}")
            try:
                self._request("GET", "/health", timeout=2.0)
                return
            except (OSError, RuntimeError):
                time.sleep(0.25)
        raise TimeoutError(
            f"llama.cpp did not become ready in {timeout:.0f} seconds.\n{self._log_tail()}"
        )

    def stop(self) -> None:
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)
        self.process = None
        if self.log is not None:
            self.log.close()
            self.log = None

    def _request(self, method: str, path: str, payload=None, timeout: float = 900.0):
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            self.base_url + path,
            data=data,
            method=method,
            headers={
                "Authorization": "Bearer comfyui-local",
                "Content-Type": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"llama.cpp HTTP {error.code}: {detail}") from error

    def _request_stream(
        self,
        method: str,
        path: str,
        payload: dict[str, Any],
        timeout: float = 900.0,
    ) -> Iterator[dict[str, Any]]:
        """POST 一个带 ``stream: True`` 的请求，逐行解析 SSE ``data:`` 事件并产出 JSON 块。"""
        data = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            self.base_url + path,
            data=data,
            method=method,
            headers={
                "Authorization": "Bearer comfyui-local",
                "Content-Type": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                for raw_line in response:
                    line = raw_line.decode("utf-8", errors="replace").strip()
                    if not line.startswith("data:"):
                        continue
                    text = line[len("data:"):].strip()
                    if not text or text == "[DONE]":
                        continue
                    try:
                        yield json.loads(text)
                    except json.JSONDecodeError:
                        continue
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"llama.cpp HTTP {error.code}: {detail}") from error

    def chat(
        self,
        messages: list[dict[str, object]],
        *,
        seed: int,
        max_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        min_p: float,
        presence_penalty: float,
        repetition_penalty: float,
        think_mode: bool,
        reasoning_effort: str,
        progress_cb: Callable[[int, int], None] | None = None,
    ) -> tuple[str, dict[str, int]]:
        payload = {
            "model": "qwen3.8-27b",
            "messages": messages,
            "seed": normalize_llama_seed(seed),
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "min_p": min_p,
            "presence_penalty": presence_penalty,
            "repeat_penalty": repetition_penalty,
            "chat_template_kwargs": {
                "enable_thinking": think_mode,
                "preserve_thinking": False,
            },
        }
        if think_mode:
            payload["reasoning_effort"] = reasoning_effort

        if progress_cb is None:
            response = self._request("POST", "/v1/chat/completions", payload)
            content = response["choices"][0]["message"].get("content") or ""
            usage = response.get("usage", {})
        else:
            payload["stream"] = True
            content = ""
            generated = 0
            usage: dict[str, int] = {}
            for chunk in self._request_stream("POST", "/v1/chat/completions", payload):
                if chunk.get("usage"):
                    usage = chunk["usage"]
                choices = chunk.get("choices") or []
                message = (choices[0] or {}).get("message")
                if message and message.get("content"):
                    # 终帧携带完整 message.content，作为权威结果
                    content = message["content"]
                else:
                    delta = (choices[0] or {}).get("delta") or {}
                    text = delta.get("content") or ""
                    if text:
                        content += text
                        generated += 1
                        progress_cb(generated, max_tokens)

        if not content.strip():
            raise RuntimeError(
                "Qwen returned no final answer. Disable think mode or increase max_tokens so thinking does not consume the output budget."
            )
        return content.strip(), dict(usage)


class LlamaServerManager:
    """Owns the one llama.cpp inference backend used by this custom node.

    运行时选择：优先独立 llama-server（standalone，官方 --fit 路径，能支撑大
    context——低显存下把部分层溢出到系统内存），其次 llama-cpp-python（CUDA 加速，
    但全量上 GPU 时 context 受显存限制，作为兜底）。启动失败时自动降级到下一个
    可用运行时。选择结果在 acquire 时记录，可用 ``runtime_status()`` 查看。
    """

    def __init__(self):
        self._lock = threading.RLock()
        self._server: LlamaServer | PythonLlamaServer | None = None
        self._key: tuple[object, ...] | None = None

    def _select_specs(
        self,
    ) -> tuple[RuntimeSpec | None, PythonRuntimeSpec | None]:
        standalone_spec: RuntimeSpec | None = None
        try:
            standalone_spec = load_runtime_spec()
        except RuntimeError:
            standalone_spec = None
        python_spec = detect_python_runtime()
        return standalone_spec, python_spec

    def acquire(
        self,
        model: Path,
        mmproj: Path,
        context_size: int = 32768,
        backend: str = "auto",
        fit: bool | None = None,
        fit_target_mib: int | None = None,
        n_gpu_layers: str | None = None,
        flash_attention: str | None = None,
    ) -> tuple[LlamaServer | PythonLlamaServer, bool]:
        standalone_spec, python_spec = self._select_specs()
        detection = runtime_status()
        LOGGER.info(
            "[Qwen H3] Runtime detection | standalone=%s(%s) | llama-cpp-python=%s(%s) | gpu_offload=%s",
            standalone_spec is not None,
            standalone_spec.backend if standalone_spec is not None else "n/a",
            python_spec is not None,
            python_spec.version if python_spec is not None else "n/a",
            detection.get("gpu_offload_available"),
        )
        if standalone_spec is None and python_spec is None:
            warnings = " ".join(detection.get("warnings", [])) or "no runtime detected"
            raise RuntimeError(
                "No usable llama.cpp runtime was found. Install a standalone runtime by "
                "running install_runtime.py from the custom node directory, or install "
                f"llama-cpp-python in the ComfyUI Python environment. {warnings}"
            )

        # 依据 UI 的「推理后端」选项决定尝试顺序与范围。
        backend_choice = (backend or "auto").strip().lower()
        attempts: list[tuple[str, RuntimeSpec | PythonRuntimeSpec]] = []
        if backend_choice in ("auto", "vulkan", "cuda"):
            if standalone_spec is not None:
                spec_backend = standalone_spec.backend
                if backend_choice == "auto" or spec_backend == backend_choice:
                    attempts.append(("standalone", standalone_spec))
                else:
                    LOGGER.warning(
                        "[Qwen H3] UI requested standalone backend '%s' but installed "
                        "standalone runtime is '%s'; falling back to llama-cpp-python.",
                        backend_choice,
                        spec_backend,
                    )
        if backend_choice in ("auto", "llama-cpp-python") or not attempts:
            if python_spec is not None:
                attempts.append(("llama-cpp-python", python_spec))

        if not attempts:
            raise RuntimeError(
                f"Backend '{backend_choice}' is not available on this machine. "
                "Available: standalone=" + (
                    standalone_spec.backend if standalone_spec is not None else "none"
                ) + ", llama-cpp-python=" + (
                    "yes" if python_spec is not None else "no"
                )
            )

        with self._lock:
            if (
                self._server is not None
                and self._key == (
                    model.resolve(),
                    mmproj.resolve(),
                    context_size,
                    backend_choice,
                    attempts[0][0],
                    getattr(attempts[0][1], "backend", ""),
                    n_gpu_layers,
                    flash_attention,
                    fit,
                    fit_target_mib,
                )
                and self._server.is_running
            ):
                return self._server, False

            self.release()
            last_error: Exception | None = None
            for kind, spec in attempts:
                try:
                    if kind == "standalone":
                        server: LlamaServer | PythonLlamaServer = LlamaServer(
                            model,
                            mmproj,
                            context_size,
                            spec,  # type: ignore[arg-type]
                            n_gpu_layers=n_gpu_layers,
                            flash_attention=flash_attention,
                            fit=fit,
                            fit_target_mib=fit_target_mib,
                        )
                    else:
                        server = PythonLlamaServer(
                            model,
                            mmproj,
                            context_size,
                            spec,  # type: ignore[arg-type]
                            n_gpu_layers=n_gpu_layers,
                        )
                    server.start()
                    self._server = server
                    self._key = (
                        model.resolve(),
                        mmproj.resolve(),
                        context_size,
                        kind,
                        server.backend,
                    )
                    LOGGER.info(
                        "[Qwen H3] Selected llama runtime: %s (backend=%s)",
                        kind,
                        server.backend,
                    )
                    return server, True
                except Exception as error:  # noqa: BLE001
                    last_error = error
                    LOGGER.warning(
                        "[Qwen H3] Runtime '%s' failed to start (%s: %s), trying next.",
                        kind,
                        type(error).__name__,
                        error,
                    )
                    try:
                        server.stop()  # noqa: F821
                    except Exception:  # noqa: BLE001
                        pass
            message = (
                "All llama.cpp runtimes failed to start. "
                f"Tried: {', '.join(k for k, _ in attempts)}."
            )
            if last_error is not None:
                message += f" Last error: {type(last_error).__name__}: {last_error}"
            raise RuntimeError(message)

    def release(self) -> None:
        with self._lock:
            server = self._server
            self._server = None
            self._key = None
            if server is not None:
                try:
                    server.stop()
                except Exception:  # noqa: BLE001
                    LOGGER.debug("Error while stopping llama runtime", exc_info=True)
