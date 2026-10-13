from __future__ import annotations

import atexit
import gc
import logging
import re
import threading
import time
from pathlib import Path

import comfy.model_management
import folder_paths
from comfy_api.latest import ComfyExtension, io

from .media import (
    image_content,
    image_grid_content,
    sample_indices_per_second,
    text_content,
)
from .runtime import LlamaServerManager
from .skills import (
    SKILL_NAMES,
    detect_h3_mode,
    mode_router_prompt,
    output_issues,
    parse_mode_selection,
    parse_skill_selection,
    router_prompt,
    system_prompt,
)

MODEL_DIR = Path(folder_paths.models_dir) / "LLM"
LLM_FOLDER = "LLM"
DEFAULT_MODEL = "Qwen3.8-27B-Q4_K_M.gguf"
DEFAULT_MMPROJ = "mmproj-F16.gguf"
REFERENCE_VIDEO_FPS = 24.0


def _register_llm_folder() -> None:
    """Register the local LLM model directory as a ComfyUI model folder.

    Registers ``ComfyUI/models/LLM`` under the ``LLM`` model-folder key and
    ensures the ``.gguf`` extension filter is attached. ComfyUI's
    ``get_filename_list`` then recursively scans the folder (including
    subdirectories such as ``Qwen3.8/``), returns relative paths, and
    automatically invalidates its cache when the folder's mtime changes, so
    newly dropped GGUF files show up without a restart.
    """
    try:
        folder_paths.add_model_folder_path(LLM_FOLDER, str(MODEL_DIR))
        paths, extensions = folder_paths.folder_names_and_paths.get(
            LLM_FOLDER, ([], set())
        )
        if not paths:
            folder_paths.folder_names_and_paths[LLM_FOLDER] = ([str(MODEL_DIR)], {".gguf"})
        elif hasattr(extensions, "add"):
            extensions.add(".gguf")
        else:
            folder_paths.folder_names_and_paths[LLM_FOLDER] = (
                paths,
                {*extensions, ".gguf"},
            )
    except (ImportError, AttributeError, KeyError, TypeError, ValueError):
        pass


_register_llm_folder()
INFERENCE_LOCK = threading.Lock()
SERVER_MANAGER = LlamaServerManager()
LOGGER = logging.getLogger("ComfyUI.QwenH3Prompt")
GREEN = "\033[92m"
RED = "\033[91m"
RESET = "\033[0m"

atexit.register(SERVER_MANAGER.release)


def _success(message: str) -> None:
    LOGGER.info("%s%s%s", GREEN, message, RESET)


def _failure(message: str) -> None:
    LOGGER.error("%s%s%s", RED, message, RESET)


def _usage_summary(usage: dict[str, int]) -> str:
    prompt_tokens = usage.get("prompt_tokens")
    completion_tokens = usage.get("completion_tokens")
    total_tokens = usage.get("total_tokens")
    if total_tokens is None:
        return "token usage unavailable"
    return (
        f"prompt {prompt_tokens or 0}, completion {completion_tokens or 0}, "
        f"total {total_tokens} tokens"
    )


def _release_node_resources() -> None:
    SERVER_MANAGER.release()
    gc.collect()
    comfy.model_management.soft_empty_cache(force=True)


def _scan_gguf_models() -> list[str]:
    """Return every GGUF file under the local LLM model directory.

    Uses ComfyUI's registered ``LLM`` folder list, which recursively scans
    ``ComfyUI/models/LLM`` (including subdirectories such as ``Qwen3.8/``),
    filters by the ``.gguf`` extension, and refreshes its cache when new
    files are added. Returns sorted forward-slash relative paths.
    """
    try:
        names = folder_paths.get_filename_list(LLM_FOLDER)
    except (KeyError, AttributeError):
        _register_llm_folder()
        try:
            names = folder_paths.get_filename_list(LLM_FOLDER)
        except (KeyError, AttributeError):
            return []
    result = []
    for name in names:
        normalized = str(name).replace("\\", "/")
        if normalized.lower().endswith(".gguf"):
            result.append(normalized)
    return sorted(result)


def _model_options(projector: bool) -> list[str]:
    files = _scan_gguf_models()
    if projector:
        options = [name for name in files if "mmproj" in name.lower()]
        fallback = DEFAULT_MMPROJ
    else:
        options = [name for name in files if "mmproj" not in name.lower()]
        fallback = DEFAULT_MODEL
    return options or [fallback]


def _resolve_model(name: str) -> Path:
    root = MODEL_DIR.resolve()
    normalized = str(name).replace("\\", "/")
    path = (root / normalized).resolve()
    try:
        path.relative_to(root)
    except ValueError:
        raise FileNotFoundError(f"Invalid GGUF path outside {root}: {name}")
    if not path.is_file() or path.suffix.lower() != ".gguf":
        raise FileNotFoundError(f"Invalid or missing GGUF model in {root}: {name}")
    return path


def _autogrow_values(inputs) -> list:
    if not inputs:
        return []

    def index(item):
        match = re.search(r"(\d+)$", item[0])
        return int(match.group(1)) if match else 0

    return [value for _, value in sorted(inputs.items(), key=index) if value is not None]


def _validate_reference_images(images: list) -> None:
    if len(images) > 9:
        raise ValueError(f"At most 9 reference images are supported; got {len(images)}.")
    for image_index, image in enumerate(images, 1):
        image_count = int(image.shape[0])
        if image_count != 1:
            raise ValueError(
                f"Reference image {image_index} must contain exactly one image; got a batch of {image_count}."
            )


def _reference_video_details(
    videos: list,
    sample_frames_per_second: int,
) -> list[tuple]:
    if len(videos) > 3:
        raise ValueError(f"At most 3 reference videos are supported; got {len(videos)}.")
    details = []
    total_duration = 0.0
    for video_index, frames in enumerate(videos, 1):
        frame_count = int(frames.shape[0])
        if frame_count < 1:
            raise ValueError(
                f"Reference video {video_index} must contain at least 1 frame; got {frame_count}."
            )
        source_duration = frame_count / REFERENCE_VIDEO_FPS
        if source_duration > 15.0:
            raise ValueError(
                f"Reference video {video_index} must be at most 15 seconds at 24 fps; "
                f"got {source_duration:.2f}s."
            )
        total_duration += source_duration
        details.append(
            (
                frames,
                frame_count,
                source_duration,
                sample_indices_per_second(
                    frame_count,
                    REFERENCE_VIDEO_FPS,
                    sample_frames_per_second,
                ),
            )
        )
    if total_duration > 15.0:
        raise ValueError(
            f"Reference videos may total at most 15 seconds; got {total_duration:.2f}s."
        )
    return details


def _picture_role(mode: str, picture_index: int) -> str:
    if mode == "i2va":
        return "the target video's first frame"
    if mode == "l2va":
        return "the target video's last frame"
    if mode == "fl2va":
        return "the target video's first frame" if picture_index == 1 else "the target video's last frame"
    return "a general visual reference whose role follows the user request"


def _sampling_settings(think_mode: bool) -> dict[str, float | int]:
    if think_mode:
        return {
            "temperature": 1.0,
            "top_p": 0.95,
            "top_k": 20,
            "min_p": 0.0,
            "presence_penalty": 0.0,
            "repetition_penalty": 1.0,
        }
    return {
        "temperature": 0.7,
        "top_p": 0.8,
        "top_k": 20,
        "min_p": 0.0,
        "presence_penalty": 1.5,
        "repetition_penalty": 1.0,
    }


def _user_content(
    prompt: str,
    duration: float,
    images: list,
    video_details: list[tuple],
    mode: str,
) -> tuple[list[dict[str, object]], str]:
    content: list[dict[str, object]] = [
        text_content(f"User request:\n{prompt}\n\nTarget duration: {duration:.2f} seconds."),
    ]
    assets = []
    for picture_index, image in enumerate(images, 1):
        role = _picture_role(mode, picture_index)
        content.append(text_content(f"<Picture {picture_index}>: connected as {role}."))
        content.append(image_content(image))
        assets.append(f"Picture {picture_index}={role}")

    for video_index, details in enumerate(video_details, 1):
        frames, frame_count, source_duration, sample_groups = details
        sampled_frame_count = sum(len(indices) for indices in sample_groups)
        content.append(
            text_content(
                f"<Video {video_index}>: {frame_count} ordered frames at {REFERENCE_VIDEO_FPS:.3f} fps "
                f"({source_duration:.2f} seconds), represented by {sampled_frame_count} sampled frames "
                f"grouped into {len(sample_groups)} chronological one-second contact sheets."
            )
        )
        for second_index, indices in enumerate(sample_groups):
            timestamps = ", ".join(
                f"{frame_index / REFERENCE_VIDEO_FPS:.3f}s" for frame_index in indices
            )
            content.append(
                text_content(
                    f"<Video {video_index}> second {second_index + 1}/{len(sample_groups)} contact sheet. "
                    f"Read cells in row-major chronological order at: {timestamps}."
                )
            )
            content.append(image_grid_content(frames, indices))
        assets.append(
            f"Video {video_index}={source_duration:.2f}s/{sampled_frame_count} sampled frames"
        )
    return content, ", ".join(assets) if assets else "none"


class QwenH3Prompt(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        models = _model_options(False)
        projectors = _model_options(True)
        return io.Schema(
            node_id="QwenH3PromptLocal",
            display_name="Qwen H3 Prompt (Local)",
            category="😺dzNodes/Qwen_H3_Prompt",
            description="Runs a local llama.cpp server with Qwen3.8 and a discovered H3 Skill from the bundled or custom Skill directories.",
            inputs=[
                io.String.Input(
                    "prompt",
                    multiline=True,
                    dynamic_prompts=True,
                    default="Describe the video or production task.",
                ),
                io.Combo.Input(
                    "skill",
                    options=["auto", *SKILL_NAMES],
                    default="auto",
                    tooltip="Auto routes across bundled and custom Skills, or choose one directly.",
                ),
                io.Float.Input(
                    "duration",
                    default=10.0,
                    min=1.0,
                    max=60.0,
                    step=0.1,
                    tooltip="Target H3 video duration in seconds.",
                ),
                io.Combo.Input(
                    "llm_model",
                    options=models,
                    default=DEFAULT_MODEL if DEFAULT_MODEL in models else models[0],
                    remote=io.RemoteOptions(
                        route=f"/models/{LLM_FOLDER}",
                        refresh_button=True,
                        control_after_refresh="first",
                    ),
                    tooltip="GGUF language model from ComfyUI/models/LLM. Click the refresh button to re-scan the local directory for newly added GGUF files.",
                ),
                io.Combo.Input(
                    "vision_model",
                    options=projectors,
                    default=DEFAULT_MMPROJ if DEFAULT_MMPROJ in projectors else projectors[0],
                    remote=io.RemoteOptions(
                        route=f"/models/{LLM_FOLDER}",
                        refresh_button=True,
                        control_after_refresh="first",
                    ),
                    tooltip="GGUF multimodal projector from ComfyUI/models/LLM. Click the refresh button to re-scan the local directory for newly added GGUF files.",
                ),
                io.Boolean.Input(
                    "think_mode",
                    display_name="think_mode",
                    default=False,
                    tooltip="Off uses Qwen's official instruct settings. On enables thinking and uses Qwen's official thinking settings.",
                ),
                io.Combo.Input(
                    "reasoning_effort",
                    # 兼容旧工作流：部分旧工作流把该值存成整数 2，部分存成字符串 "2"，
                    # ComfyUI 服务端校验是严格类型比较，因此整数 2 与字符串 "2" 都要收录，
                    # 二者在 execute() 中都会被归一化为 medium。
                    options=["low", "medium", "xhigh", "2", 2],
                    default="medium",
                    tooltip="Applied only in thinking mode. Medium is the balanced RTX 3090 preset. '2' is kept for backward compatibility with older workflows and is treated as medium.",
                ),
                io.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=0xFFFFFFFFFFFFFFFF,
                    step=1,
                    control_after_generate=True,
                    tooltip="ComfyUI seed. It is mapped deterministically to llama.cpp's 32-bit seed range.",
                ),
                io.Int.Input(
                    "max_tokens",
                    default=8192,
                    min=256,
                    max=8192,
                    step=128,
                    tooltip="Maximum generated tokens, including thinking when thinking mode is enabled.",
                ),
                io.Int.Input(
                    "video_sample_frames_per_sec",
                    default=2,
                    min=1,
                    max=8,
                    step=1,
                    advanced=True,
                    tooltip="Frames sampled from each second of every reference video. Frames from the same second are sent as one chronological contact sheet.",
                ),
                io.Boolean.Input(
                    "force_unload_model",
                    display_name="force unload model",
                    default=True,
                    tooltip="On stops the node's llama.cpp server and clears its VRAM and system memory after each run. Errors always force an unload.",
                ),
                # --- 运行时/推理参数（带帮助说明） ---
                io.Combo.Input(
                    "runtime_backend",
                    options=["auto", "vulkan", "cuda", "llama-cpp-python", True],
                    default="auto",
                    tooltip=(
                        "推理后端选择。\n"
                        "auto（推荐）：自动检测并选择——优先独立 llama-server（standalone，当前为 "
                        "Vulkan，配合 --fit 可支撑大 context），其次 llama-cpp-python。\n"
                        "vulkan：强制使用独立 Vulkan 运行时（--fit 显存自适应，低显存下把部分层溢出到 "
                        "系统内存以支撑大 context，8GB 显存推荐；当前已安装）。\n"
                        "cuda（NV，英伟达 CUDA）：强制使用独立 CUDA 运行时（需先运行 "
                        "install_runtime.py --backend cuda 安装；未安装时自动回退到 llama-cpp-python）。\n"
                        "llama-cpp-python：强制使用进程内 CUDA 后端（借用 T8 方案；CUDA 快，但全量上 "
                        "GPU 时 context 受显存限制，超大工作流可能回退 CPU）。"
                    ),
                ),
                io.Int.Input(
                    "context_size",
                    default=32768,
                    min=2048,
                    max=131072,
                    step=2048,
                    advanced=True,
                    tooltip=(
                        "上下文长度（token）。H3 带图工作流（图片 token + 长系统提示 + 8192 输出）需 8K+。"
                        "越大越能容纳长输入/长输出，但显存/内存占用越高、推理越慢。\n"
                        "standalone（--fit）下推荐保持 32768；llama-cpp-python 后端会按可用显存自动调整，"
                        "此处设大值不会被浪费。8GB 显存 + 32GB 内存建议默认 32768。"
                    ),
                ),
                io.Boolean.Input(
                    "fit_enabled",
                    display_name="fit_enabled",
                    default=True,
                    tooltip=(
                        "显存自适应（--fit，仅独立运行时生效）：把部分层溢出到系统内存，低显存也能支撑大 "
                        "context。8GB 显存强烈建议保持开启；关闭后全部依赖显存，大 context 可能放不下。"
                    ),
                ),
                io.Int.Input(
                    "fit_target_mib",
                    default=1536,
                    min=256,
                    max=8192,
                    step=128,
                    advanced=True,
                    tooltip=(
                        "--fit 的显存目标（MiB，仅独立运行时生效）：预留这么多显存给 KV 缓存与计算，其余 "
                        "层溢出到系统内存。值越小越激进（溢出更多、更稳但更慢）；值越大越依赖显存（更快但 "
                        "8GB 卡可能放不下大 context）。8GB 显存推荐 1536。"
                    ),
                ),
                io.Combo.Input(
                    "n_gpu_layers",
                    options=["auto", "-1"],
                    default="auto",
                    advanced=True,
                    tooltip=(
                        "GPU 卸载层数。auto（推荐）：由运行时根据显存自动决定；-1：全部层上 GPU（最快，"
                        "但显存占用最高，8GB 卡下大 context 可能溢出）。standalone 与 llama-cpp-python 均生效。"
                    ),
                ),
                io.Combo.Input(
                    "flash_attention",
                    options=["auto", "on", "off"],
                    default="auto",
                    advanced=True,
                    tooltip=(
                        "Flash Attention（仅独立运行时生效）。auto（推荐）：由运行时自动检测；on：加速并降低 "
                        "显存占用；off：兼容性最好但更慢（且会同时关闭量化 KV 缓存，内存占用更高）。"
                    ),
                ),
                io.Autogrow.Input(
                    "reference_images",
                    optional=True,
                    template=io.Autogrow.TemplatePrefix(
                        input=io.Image.Input(
                            "reference_image",
                            tooltip="One reference image. State first-frame or last-frame intent explicitly in the prompt when needed.",
                        ),
                        prefix="reference_image_",
                        min=0,
                        max=9,
                    ),
                ),
                io.Autogrow.Input(
                    "reference_videos",
                    optional=True,
                    template=io.Autogrow.TemplatePrefix(
                        input=io.Image.Input(
                            "reference_video",
                            tooltip="Ordered video frames as an IMAGE batch, compatible with VHS Load Video.",
                        ),
                        prefix="reference_video_",
                        min=0,
                        max=3,
                    ),
                ),
            ],
            outputs=[
                io.String.Output("h3_prompt"),
                io.String.Output("selected_skill"),
                io.String.Output("detected_mode"),
            ],
        )

    @classmethod
    def execute(
        cls,
        prompt,
        skill,
        duration,
        llm_model,
        vision_model,
        think_mode,
        reasoning_effort,
        seed,
        max_tokens,
        video_sample_frames_per_sec,
        runtime_backend,
        context_size,
        fit_enabled,
        fit_target_mib,
        n_gpu_layers,
        flash_attention,
        force_unload_model,
        reference_images=None,
        reference_videos=None,
    ) -> io.NodeOutput:
        total_started = time.perf_counter()
        # Sanitize legacy/stale reasoning_effort values (e.g. "2" from older
        # workflows) down to a valid llama.cpp effort level.
        _VALID_EFFORTS = ("low", "medium", "xhigh")
        if reasoning_effort not in _VALID_EFFORTS:
            LOGGER.info(
                "[Qwen H3] Normalizing legacy reasoning_effort=%r to 'medium'",
                reasoning_effort,
            )
            reasoning_effort = "medium"
        # 兼容旧工作流错位/遗留值：把运行时选项归一化到合法值，避免错位布尔值导致崩溃。
        if runtime_backend not in ("auto", "vulkan", "cuda", "llama-cpp-python"):
            LOGGER.info("[Qwen H3] Normalizing runtime_backend=%r to 'auto'", runtime_backend)
            runtime_backend = "auto"
        if flash_attention not in ("auto", "on", "off"):
            LOGGER.info("[Qwen H3] Normalizing flash_attention=%r to 'auto'", flash_attention)
            flash_attention = "auto"
        if n_gpu_layers not in ("auto", "-1"):
            LOGGER.info("[Qwen H3] Normalizing n_gpu_layers=%r to 'auto'", n_gpu_layers)
            n_gpu_layers = "auto"
        try:
            context_size = int(context_size)
            fit_target_mib = int(fit_target_mib)
        except (TypeError, ValueError):
            context_size = 32768
            fit_target_mib = 1536
        fit_enabled = bool(fit_enabled)
        content = None
        messages = None
        repair_messages = None
        LOGGER.info(
            "[Qwen H3] Node execution started | skill=%s | think_mode=%s | seed=%d | force_unload_model=%s",
            skill,
            think_mode,
            seed,
            force_unload_model,
        )
        try:
            if not prompt.strip():
                raise ValueError("prompt must not be empty")

            stage_started = time.perf_counter()
            LOGGER.info("[Qwen H3] Preparing multimodal input")
            images = _autogrow_values(reference_images)
            videos = _autogrow_values(reference_videos)
            _validate_reference_images(images)
            video_details = _reference_video_details(
                videos,
                video_sample_frames_per_sec,
            )
            mode = detect_h3_mode(len(images), len(videos))
            model = _resolve_model(llm_model)
            projector = _resolve_model(vision_model)
            settings = _sampling_settings(think_mode)
            LOGGER.info(
                "[Qwen H3] Sampling configuration selected | mode=%s | temperature=%.2f | top_p=%.2f | top_k=%d | min_p=%.2f | presence_penalty=%.2f | repetition_penalty=%.2f",
                "thinking" if think_mode else "instruct",
                settings["temperature"],
                settings["top_p"],
                settings["top_k"],
                settings["min_p"],
                settings["presence_penalty"],
                settings["repetition_penalty"],
            )
            LOGGER.info(
                "[Qwen H3] Multimodal input parsed | images=%d | videos=%d | initial_mode=%s | elapsed %.2f s",
                len(images),
                len(videos),
                mode or "automatic",
                time.perf_counter() - stage_started,
            )

            with INFERENCE_LOCK:
                stage_started = time.perf_counter()
                LOGGER.info("[Qwen H3] Unloading ComfyUI models for Qwen")
                comfy.model_management.unload_all_models()
                comfy.model_management.soft_empty_cache()
                LOGGER.info(
                    "[Qwen H3] ComfyUI model cleanup complete | elapsed %.2f s",
                    time.perf_counter() - stage_started,
                )

                stage_started = time.perf_counter()
                LOGGER.info(
                    "[Qwen H3] Model loading started | model=%s | mmproj=%s",
                    model.name,
                    projector.name,
                )
                server, loaded_new = SERVER_MANAGER.acquire(
                    model,
                    projector,
                    context_size=int(context_size),
                    backend=runtime_backend,
                    fit=bool(fit_enabled),
                    fit_target_mib=int(fit_target_mib),
                    n_gpu_layers=n_gpu_layers,
                    flash_attention=flash_attention,
                )
                LOGGER.info(
                    "[Qwen H3] Model loading complete | %s | platform=%s | backend=%s | port=%d | elapsed %.2f s",
                    "new model loaded" if loaded_new else "resident model reused",
                    server.runtime_spec.platform_name,
                    server.backend,
                    server.port,
                    time.perf_counter() - stage_started,
                )

                if mode is None:
                    stage_started = time.perf_counter()
                    LOGGER.info(
                        "[Qwen H3] Starting automatic H3 mode routing | images=%d",
                        len(images),
                    )
                    mode_selection, mode_usage = server.chat(
                        mode_router_prompt(prompt, len(images)),
                        seed=seed,
                        max_tokens=16,
                        temperature=0.0,
                        top_p=1.0,
                        top_k=1,
                        min_p=0.0,
                        presence_penalty=0.0,
                        repetition_penalty=1.0,
                        think_mode=False,
                        reasoning_effort="low",
                    )
                    mode = parse_mode_selection(mode_selection, len(images))
                    LOGGER.info(
                        "[Qwen H3] H3 mode routing complete | selected=%s | %s | elapsed %.2f s",
                        mode,
                        _usage_summary(mode_usage),
                        time.perf_counter() - stage_started,
                    )
                else:
                    LOGGER.info(
                        "[Qwen H3] H3 mode routing result | selected=%s (deterministic from media counts)",
                        mode,
                    )

                stage_started = time.perf_counter()
                content, asset_summary = _user_content(
                    prompt,
                    duration,
                    images,
                    video_details,
                    mode,
                )
                LOGGER.info(
                    "[Qwen H3] Multimodal input ready | mode=%s | assets=%s | elapsed %.2f s",
                    mode,
                    asset_summary,
                    time.perf_counter() - stage_started,
                )

                selected = skill
                if selected == "auto":
                    stage_started = time.perf_counter()
                    LOGGER.info("[Qwen H3] Starting automatic Skill routing")
                    selection, routing_usage = server.chat(
                        router_prompt(prompt, mode, asset_summary),
                        seed=seed,
                        max_tokens=48,
                        temperature=0.0,
                        top_p=1.0,
                        top_k=1,
                        min_p=0.0,
                        presence_penalty=0.0,
                        repetition_penalty=1.0,
                        think_mode=False,
                        reasoning_effort="low",
                    )
                    selected = parse_skill_selection(selection)
                    LOGGER.info(
                        "[Qwen H3] Skill routing complete | selected=%s | %s | elapsed %.2f s",
                        selected,
                        _usage_summary(routing_usage),
                        time.perf_counter() - stage_started,
                    )
                else:
                    LOGGER.info(
                        "[Qwen H3] Skill routing result | selected=%s (manual)",
                        selected,
                    )

                messages = [
                    {
                        "role": "system",
                        "content": system_prompt(selected, mode, duration),
                    },
                    {"role": "user", "content": content},
                ]
                stage_started = time.perf_counter()
                LOGGER.info(
                    "[Qwen H3] Inference started | mode=%s | skill=%s | max_tokens=%d",
                    mode,
                    selected,
                    max_tokens,
                )
                result, inference_usage = server.chat(
                    messages,
                    seed=seed,
                    max_tokens=max_tokens,
                    think_mode=think_mode,
                    reasoning_effort=reasoning_effort,
                    **settings,
                )
                LOGGER.info(
                    "[Qwen H3] Inference complete | %s | elapsed %.2f s",
                    _usage_summary(inference_usage),
                    time.perf_counter() - stage_started,
                )

                LOGGER.info("[Qwen H3] Validating node output")
                issues = output_issues(
                    result,
                    mode,
                    duration,
                )
                if issues:
                    LOGGER.warning(
                        "[Qwen H3] Detected %d output issue(s); starting automatic repair: %s",
                        len(issues),
                        "; ".join(issues),
                    )
                    stage_started = time.perf_counter()
                    repair_messages = [
                        *messages,
                        {"role": "assistant", "content": result},
                        {
                            "role": "user",
                            "content": (
                                "Repair the output and return the complete requested content only. "
                                "Do not add a wrapper or impose an output schema that the selected Skill does not "
                                "request. Problems: " + "; ".join(issues)
                            ),
                        },
                    ]
                    result, repair_usage = server.chat(
                        repair_messages,
                        seed=seed,
                        max_tokens=max_tokens,
                        think_mode=think_mode,
                        reasoning_effort=reasoning_effort,
                        **settings,
                    )
                    LOGGER.info(
                        "[Qwen H3] Automatic repair inference complete | %s | elapsed %.2f s",
                        _usage_summary(repair_usage),
                        time.perf_counter() - stage_started,
                    )
                    remaining = output_issues(
                        result,
                        mode,
                        duration,
                    )
                    if remaining:
                        raise RuntimeError(
                            "Qwen output still fails the basic output contract: "
                            + "; ".join(remaining)
                        )
                LOGGER.info("[Qwen H3] Node output validation passed")

                content = None
                messages = None
                repair_messages = None
                if force_unload_model:
                    stage_started = time.perf_counter()
                    LOGGER.info("[Qwen H3] Force-releasing node model, VRAM, and system memory")
                    _release_node_resources()
                    LOGGER.info(
                        "[Qwen H3] Forced resource release complete | elapsed %.2f s",
                        time.perf_counter() - stage_started,
                    )
                else:
                    LOGGER.info(
                        "[Qwen H3] Model unload disabled; the bundled llama.cpp model will remain resident for reuse"
                    )

            _success(
                f"[Qwen H3] Execution completed successfully | skill={selected} | mode={mode} | "
                f"total elapsed {time.perf_counter() - total_started:.2f} s"
            )
            return io.NodeOutput(result, selected, mode)
        except Exception as error:
            content = None
            messages = None
            repair_messages = None
            try:
                with INFERENCE_LOCK:
                    _release_node_resources()
            except Exception as cleanup_error:  # noqa: BLE001
                _failure(
                    f"[Qwen H3] Cleanup after failure also failed: "
                    f"{type(cleanup_error).__name__}: {cleanup_error}"
                )
            _failure(
                f"[Qwen H3] Execution failed | {type(error).__name__}: {error} | "
                f"total elapsed {time.perf_counter() - total_started:.2f} s"
            )
            raise


class QwenH3PromptExtension(ComfyExtension):
    async def get_node_list(self):
        return [QwenH3Prompt]


async def comfy_entrypoint() -> QwenH3PromptExtension:
    from .runtime_api import register_runtime_api_routes

    register_runtime_api_routes()
    return QwenH3PromptExtension()
