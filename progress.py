"""推理进度上报：把 token 级进度通过 ComfyUI websocket 推送到前端节点。"""
from __future__ import annotations

import logging
from typing import Any

LOGGER = logging.getLogger("ComfyUI.QwenH3Prompt.progress")


def _prompt_server() -> Any:
    """安全获取 ComfyUI PromptServer 单例；不可用返回 None。"""
    try:
        from server import PromptServer  # type: ignore[import-not-found]

        return PromptServer.instance
    except Exception:  # noqa: BLE001
        return None


def publish(stage: str, current: int, total: int, label: str = "") -> None:
    """向 ComfyUI 前端推送一条 ``qwen_h3_progress`` websocket 事件。

    ``stage`` 取值：routing / inference / repair / done。
    前端扩展会监听该事件并把进度条渲染到节点上的 ``runtime_status`` 控件。
    """
    if total <= 0:
        return
    percent = max(0.0, min(1.0, float(current) / float(total)))
    server = _prompt_server()
    if server is None:
        return
    try:
        server.send_sync(
            "qwen_h3_progress",
            {
                "stage": stage,
                "current": int(current),
                "total": int(total),
                "percent": round(percent, 3),
                "label": label,
            },
        )
    except Exception as exc:  # noqa: BLE001
        LOGGER.debug("progress publish failed: %s", exc)
