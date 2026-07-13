"""Thin client for ComfyUI's HTTP API (upload, queue, poll, fetch, free)."""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path

import httpx

from studio.config import settings

WORKFLOWS_DIR = Path(__file__).resolve().parent / "comfy_workflows"


class ComfyError(Exception):
    pass


# Model filenames inside the bundled templates, remapped to whatever the user
# configured in .env (LDS_QWEN_EDIT_MODEL etc.) so renamed files just work.
_MODEL_INPUTS = {
    "unet_name": "qwen_edit_model",
    "lora_name": "angles_lora",
    "ckpt_name": "sam3_checkpoint",
}
_UPSCALE_SETTINGS = ("dejpg_model", "upscale_model")  # in template node order


_combo_cache: dict[tuple[str, str], list[str]] = {}


def _server_filename(class_type: str, input_key: str, value: str) -> str:
    """Match `value` against the server's file list ignoring path separators —
    ComfyUI enumerates subfolder files with the server OS's separator, and a
    graph value must match that list exactly."""
    if "/" not in value and "\\" not in value:
        return value
    key = (class_type, input_key)
    if key not in _combo_cache:
        try:
            info = httpx.get(f"{settings.comfy_url}/object_info/{class_type}",
                             timeout=10).json()
            options = info[class_type]["input"]["required"][input_key][0]
            _combo_cache[key] = options if isinstance(options, list) else []
        except Exception:
            _combo_cache[key] = []
    norm = value.replace("\\", "/")
    for opt in _combo_cache[key]:
        if opt.replace("\\", "/") == norm:
            return opt
    return value


def load_template(name: str) -> dict:
    graph = json.loads((WORKFLOWS_DIR / f"{name}.json").read_text(encoding="utf-8"))
    upscale_iter = iter(_UPSCALE_SETTINGS)
    for node in graph.values():
        for input_key, setting_attr in _MODEL_INPUTS.items():
            if input_key in node.get("inputs", {}):
                node["inputs"][input_key] = _server_filename(
                    node["class_type"], input_key, getattr(settings, setting_attr))
        if node.get("class_type") == "UpscaleModelLoader":
            node["inputs"]["model_name"] = getattr(settings, next(upscale_iter))
    return graph


def is_up(timeout: float = 3.0) -> bool:
    try:
        httpx.get(f"{settings.comfy_url}/system_stats", timeout=timeout).raise_for_status()
        return True
    except Exception:
        return False


def upload_image(path: Path) -> str:
    """Upload into ComfyUI's input folder; returns the stored filename."""
    name = f"lds_{uuid.uuid4().hex[:10]}{path.suffix.lower()}"
    with path.open("rb") as f:
        r = httpx.post(
            f"{settings.comfy_url}/upload/image",
            files={"image": (name, f, "image/png")},
            data={"overwrite": "true"},
            timeout=60,
        )
    r.raise_for_status()
    return r.json()["name"]


def queue_backlog() -> int:
    """Number of pending prompts other work has queued in ComfyUI."""
    try:
        q = httpx.get(f"{settings.comfy_url}/queue", timeout=10).json()
        return len(q.get("queue_pending", []))
    except Exception:
        return 0


def run_prompt(graph: dict, timeout: float = 600.0) -> list[dict]:
    """Queue an API-format graph, wait for completion, return output image refs."""
    backlog = queue_backlog()
    if backlog > 10:
        raise ComfyError(
            f"ComfyUI queue is busy: {backlog} jobs already pending. This app's jobs "
            f"would wait behind them. Open ComfyUI ({settings.comfy_url}) and clear the "
            f"queue (Queue panel → Clear, or the Manager's 'Clear Queue'), or let it "
            f"finish, then retry. To generate without ComfyUI, switch the engine to the "
            f"Cloud (Gemini) option, or set the isolation/restore backend to Built-in/Basic."
        )
    r = httpx.post(f"{settings.comfy_url}/prompt", json={"prompt": graph}, timeout=30)
    if r.status_code != 200:
        raise ComfyError(f"queue rejected: {r.text[:500]}")
    prompt_id = r.json()["prompt_id"]

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        h = httpx.get(f"{settings.comfy_url}/history/{prompt_id}", timeout=30).json()
        if prompt_id in h:
            entry = h[prompt_id]
            status = entry.get("status", {})
            if status.get("status_str") == "error":
                msgs = [
                    m[1].get("exception_message", "")
                    for m in status.get("messages", [])
                    if m[0] == "execution_error"
                ]
                raise ComfyError(f"execution error: {'; '.join(msgs) or 'unknown'}")
            images = []
            for node_output in entry.get("outputs", {}).values():
                images.extend(node_output.get("images", []))
            if images:
                return images
            if status.get("completed"):
                raise ComfyError("run completed but produced no images")
        time.sleep(1.5)
    raise ComfyError(f"timed out after {timeout}s waiting for prompt {prompt_id}")


def fetch_image(ref: dict, out_path: Path) -> Path:
    r = httpx.get(
        f"{settings.comfy_url}/view",
        params={
            "filename": ref["filename"],
            "subfolder": ref.get("subfolder", ""),
            "type": ref.get("type", "output"),
        },
        timeout=120,
    )
    r.raise_for_status()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(r.content)
    return out_path


def free_vram() -> None:
    """Ask ComfyUI to unload models + free memory (before loading the captioner)."""
    try:
        httpx.post(
            f"{settings.comfy_url}/free",
            json={"unload_models": True, "free_memory": True},
            timeout=30,
        )
    except Exception:
        pass  # best effort; captioner load will fail loudly if VRAM is truly short
