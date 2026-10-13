"""llama-cpp-python 推理后端（借自 comfyui-minimax-h3-prompt-enhancer-T8 的本地 Qwen 方案）。

通过 ``llama_cpp.Llama`` 在进程内加载 GGUF 模型与 mmproj 视觉投影器，
提供与 ``runtime.LlamaServer`` 相同的接口（``start`` / ``stop`` /
``is_running`` / ``chat``），使插件在缺少独立 llama-server 时可回退到
ComfyUI Python 环境内已安装的 llama-cpp-python（含 CUDA 时自动使用
``ggml-cuda.dll`` 做 GPU 卸载）。

注意：H3 节点会先用 ``think_mode=False`` 做模式/技能路由，再用
``think_mode=True`` 做最终推理。llama-cpp-python 的 chat handler 在构造时
固定 ``enable_thinking``，因此在 think_mode 变化时本后端会重建模型实例以
保证正确性。
"""
from __future__ import annotations

import importlib.util
import inspect
import logging
import struct
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .runtime_detect import PythonRuntimeSpec, detect_python_runtime


LOGGER = logging.getLogger("ComfyUI.QwenH3Prompt.runtime_python")


class PythonRuntimeError(RuntimeError):
    pass


LLAMA_SEED_MODULUS = 0xFFFFFFFF


def normalize_llama_seed(seed: int) -> int:
    return int(seed) % LLAMA_SEED_MODULUS


def read_gguf_model_info(
    path: Path, read_bytes: int = 64 * 1024 * 1024
) -> tuple[str, int, int, int, int]:
    """从 GGUF 头部元数据读取 KV 键。

    返回 ``(general.architecture, block_count, embedding_length, head_count,
    head_count_kv)``，缺失项以 0 表示。
    """
    scalar = {
        0: ("<B", 1), 1: ("<b", 1), 2: ("<H", 2), 3: ("<h", 2),
        4: ("<I", 4), 5: ("<i", 4), 6: ("<f", 4), 7: ("<?", 1),
        10: ("<Q", 8), 11: ("<q", 8), 12: ("<d", 8),
    }

    def u32(b, o): return struct.unpack_from("<I", b, o)[0]
    def u64(b, o): return struct.unpack_from("<Q", b, o)[0]

    try:
        with open(path, "rb") as fh:
            head = fh.read(read_bytes)
    except OSError:
        return "", 0, 0, 0, 0
    if head[:4] != b"GGUF":
        return "", 0, 0, 0, 0
    architecture = ""
    block_count = 0
    embedding_length = 0
    head_count = 0
    head_count_kv = 0
    try:
        kv_count = u64(head, 16)
        offset = 24
        arch_prefix = ""
        for _ in range(min(kv_count, 4000)):
            key_len = u64(head, offset); offset += 8
            key = head[offset:offset + key_len].decode("utf-8", "replace"); offset += key_len
            value_type = u32(head, offset); offset += 4
            if value_type == 8:  # string
                string_len = u64(head, offset); offset += 8
                value = head[offset:offset + string_len].decode("utf-8", "replace"); offset += string_len
                if key == "general.architecture":
                    architecture = value
                    arch_prefix = value
            elif value_type in scalar:
                fmt, size = scalar[value_type]
                value = struct.unpack_from(fmt, head, offset)[0]; offset += size
                if key == f"{arch_prefix}.block_count" and block_count == 0:
                    block_count = int(value)
                elif key == f"{arch_prefix}.embedding_length" and embedding_length == 0:
                    embedding_length = int(value)
                elif key == f"{arch_prefix}.attention.head_count" and head_count == 0:
                    head_count = int(value)
                elif key == f"{arch_prefix}.attention.head_count_kv" and head_count_kv == 0:
                    head_count_kv = int(value)
            elif value_type == 9:  # array
                array_type = u32(head, offset); offset += 4
                array_len = u64(head, offset); offset += 8
                for _ in range(array_len):
                    if array_type == 8:
                        string_len = u64(head, offset); offset += 8
                        offset += string_len
                    elif array_type in scalar:
                        offset += scalar[array_type][1]
                    else:
                        return architecture, block_count, embedding_length, head_count, head_count_kv
            else:
                return architecture, block_count, embedding_length, head_count, head_count_kv
    except (struct.error, IndexError, OSError):
        pass
    return architecture, block_count, embedding_length, head_count, head_count_kv


def read_gguf_architecture(path: Path) -> str:
    return read_gguf_model_info(path)[0]


def _handler_class(architecture: str, think_mode: bool) -> tuple[type[Any], dict[str, Any]]:
    try:
        from llama_cpp import llama_chat_format
    except (ImportError, OSError) as error:
        raise PythonRuntimeError(
            "llama-cpp-python chat handlers could not be loaded."
        ) from error
    normalized = architecture.casefold().replace("-", "").replace("_", "")
    candidates: list[tuple[str, dict[str, Any]]] = []
    if normalized == "qwen35":
        candidates.append(("Qwen35ChatHandler", {"enable_thinking": think_mode}))
    elif "qwen3" in normalized:
        candidates.append(("Qwen3VLChatHandler", {"force_reasoning": think_mode}))
    elif "qwen2" in normalized:
        candidates.append(("Qwen25VLChatHandler", {}))
    candidates.append(("MTMDChatHandler", {}))
    for name, options in candidates:
        handler = getattr(llama_chat_format, name, None)
        if handler is not None:
            return handler, options
    raise PythonRuntimeError(
        "The installed llama-cpp-python build has no compatible multimodal chat handler. "
        "Install a newer matching wheel."
    )


def _supports_keyword_argument(callback: Any, name: str) -> bool:
    try:
        parameters = inspect.signature(callback).parameters
    except (TypeError, ValueError):
        return True
    parameter = parameters.get(name)
    if parameter is not None:
        return parameter.kind is not inspect.Parameter.POSITIONAL_ONLY
    return any(
        item.kind is inspect.Parameter.VAR_KEYWORD
        for item in parameters.values()
    )


def _available_vram_mb() -> tuple[int, int] | None:
    """返回 ``(total_mb, free_mb)``；探测失败时返回 None。"""
    try:
        import pynvml

        pynvml.nvmlInit()
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            return int(info.total / 1048576), int(info.free / 1048576)
        finally:
            pynvml.nvmlShutdown()
    except (ImportError, OSError, RuntimeError, AttributeError):
        pass
    try:
        import torch

        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info(0)
            return int(total / 1048576), int(free / 1048576)
    except (ImportError, OSError, RuntimeError, AttributeError):
        pass
    return None


def _kv_bytes_per_token(model: Path) -> int:
    """估算 KV 缓存每 token 字节数（按 q8_0，即实测的默认 KV 类型）；失败回退 128KB。"""
    _, block_count, embedding_length, head_count, head_count_kv = read_gguf_model_info(model)
    if block_count > 0 and head_count > 0 and head_count_kv > 0 and embedding_length > 0:
        head_dim = embedding_length // head_count
        # K 与 V 各一份，q8_0 每元素 1 字节
        return 2 * block_count * head_count_kv * head_dim
    return 128 * 1024


class PythonLlamaServer:
    """llama-cpp-python 推理后端，接口与 ``runtime.LlamaServer`` 兼容。"""

    def __init__(
        self,
        model: Path,
        mmproj: Path,
        context_size: int = 32768,
        runtime_spec: PythonRuntimeSpec | None = None,
    ):
        self.model = model
        self.mmproj = mmproj
        self.context_size = int(context_size)
        self.runtime_spec = runtime_spec or detect_python_runtime() or PythonRuntimeSpec()
        self.backend = "llama-cpp-python"
        self.port = 0  # 进程内推理，无 HTTP 端口
        self.llm: Any = None
        self.chat_handler: Any = None
        self._loaded_think_mode: bool | None = None
        self._lock = threading.RLock()

    @property
    def is_running(self) -> bool:
        return self.llm is not None

    def _stop_inner(self) -> None:
        llm = self.llm
        handler = self.chat_handler
        self.llm = None
        self.chat_handler = None
        self._loaded_think_mode = None
        try:
            if llm is not None:
                llm.close()
        except (AttributeError, RuntimeError, OSError):
            pass
        try:
            exit_stack = getattr(handler, "_exit_stack", None)
            if exit_stack is not None:
                exit_stack.close()
        except (AttributeError, RuntimeError, OSError):
            pass

    def stop(self) -> None:
        with self._lock:
            self._stop_inner()

    def start(self, think_mode: bool = False, timeout: float = 240.0) -> None:
        del timeout
        with self._lock:
            self._stop_inner()
            try:
                from llama_cpp import Llama
            except (ImportError, OSError) as error:
                raise PythonRuntimeError(
                    "llama-cpp-python could not be imported from the active ComfyUI "
                    f"Python environment: {type(error).__name__}: {error}"
                ) from error

            handler = None
            handler_options: dict[str, Any] = {}
            if self.mmproj is not None and Path(self.mmproj).is_file():
                architecture = read_gguf_architecture(self.model)
                handler_class, handler_options = _handler_class(architecture, think_mode)
                try:
                    handler = handler_class(
                        clip_model_path=str(self.mmproj),
                        verbose=False,
                        use_gpu=True,
                        image_min_tokens=1024,
                        image_max_tokens=1024,
                        **handler_options,
                    )
                except (TypeError, ValueError, RuntimeError, OSError) as error:
                    raise PythonRuntimeError(
                        "llama-cpp-python could not initialize the selected model/mmproj "
                        f"pair. {type(error).__name__}: {error}"
                    ) from error

            # 本后端现作为兜底（主路径为 standalone --fit）。因此优先保证 context
            # 足够 H3 工作流（含图片时需 8K+），避免 M-RoPE 禁 context-shift 导致硬错误。
            # 显存足够时全量上 GPU（快）；显存不足时允许 llama.cpp 回退到 CPU（慢但
            # 可用），而不是把 context 压到会报错的小值。
            requested_ctx = int(self.context_size)
            effective_ctx = requested_ctx
            vram = _available_vram_mb()
            if vram is not None:
                free_mb = vram[1]
                base_mb = 0.0
                if Path(self.model).is_file():
                    base_mb += Path(self.model).stat().st_size / 1048576
                if self.mmproj is not None and Path(self.mmproj).is_file():
                    base_mb += Path(self.mmproj).stat().st_size / 1048576
                kv_bytes = _kv_bytes_per_token(self.model)
                reserve_mb = 300.0
                kv_budget_mb = free_mb - base_mb - reserve_mb
                gpu_max_ctx = int(kv_budget_mb * 1048576 / kv_bytes) if kv_bytes > 0 else requested_ctx
                # 只要显存能支撑 H3 所需的 16384 context 就留在 GPU；否则回退 CPU 也保住 context
                effective_ctx = min(requested_ctx, max(16384, gpu_max_ctx))
            if effective_ctx != requested_ctx:
                LOGGER.warning(
                    "[Qwen H3] python backend uses context %d (requested %d, GPU-fit ~%d; "
                    "may fall back to CPU to preserve context) | free VRAM ~%.0f MB",
                    effective_ctx,
                    requested_ctx,
                    (gpu_max_ctx if vram is not None else requested_ctx),
                    (vram[1] if vram is not None else 0),
                )
            self._effective_ctx = effective_ctx

            llama_options: dict[str, Any] = {
                "model_path": str(self.model),
                "chat_handler": handler,
                "n_gpu_layers": -1,
                "n_ctx": effective_ctx,
                "verbose": False,
            }
            try:
                llama_parameters = inspect.signature(Llama).parameters
            except (TypeError, ValueError):
                llama_parameters = {}
            if "chat_template_kwargs" in llama_parameters:
                llama_options["chat_template_kwargs"] = {
                    "enable_thinking": think_mode,
                    "preserve_thinking": False,
                }
            try:
                self.llm = Llama(**llama_options)
            except (TypeError, ValueError, RuntimeError, OSError) as error:
                self._stop_inner()
                raise PythonRuntimeError(
                    "llama-cpp-python failed to load the selected GGUF. "
                    f"{type(error).__name__}: {error}"
                ) from error
            self.chat_handler = handler
            self._loaded_think_mode = bool(think_mode)

    def _ensure_think_mode(self, think_mode: bool) -> None:
        if self._loaded_think_mode != bool(think_mode):
            self.start(think_mode=bool(think_mode))

    def chat(
        self,
        messages: list[dict[str, Any]],
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
    ) -> tuple[str, dict[str, Any]]:
        del reasoning_effort  # llama-cpp-python 由 enable_thinking 控制思考强度
        with self._lock:
            if self.llm is None:
                raise PythonRuntimeError("llama-cpp-python model is not loaded.")
            self._ensure_think_mode(think_mode)
            if self.llm is None:
                raise PythonRuntimeError("llama-cpp-python model is not loaded.")

            options: dict[str, Any] = {
                "messages": messages,
                "seed": normalize_llama_seed(seed),
                "max_tokens": int(max_tokens),
                "temperature": float(temperature),
                "top_p": float(top_p),
                "top_k": int(top_k),
                "min_p": float(min_p),
                "presence_penalty": float(presence_penalty),
                "repeat_penalty": float(repetition_penalty),
                "stream": False,
            }
            for name in ("presence_penalty", "repeat_penalty", "min_p"):
                if not _supports_keyword_argument(self.llm.create_chat_completion, name):
                    options.pop(name, None)
            try:
                result = self.llm.create_chat_completion(**options)
            except (TypeError, ValueError, RuntimeError, OSError) as error:
                raise PythonRuntimeError(
                    f"llama-cpp-python inference failed. {type(error).__name__}: {error}"
                ) from error
            try:
                content = result["choices"][0]["message"].get("content") or ""
            except (KeyError, IndexError, TypeError, AttributeError) as error:
                raise PythonRuntimeError(
                    "llama-cpp-python response is missing choices[0].message.content."
                ) from error
            if isinstance(content, list):
                content = "".join(
                    str(part.get("text") or "")
                    for part in content
                    if isinstance(part, dict) and part.get("type") in (None, "text")
                )
            content = str(content).strip()
            if not content:
                raise PythonRuntimeError(
                    "The local GGUF returned no final answer. Disable thinking or "
                    "increase max output tokens."
                )
            return content, dict(result.get("usage") or {})
