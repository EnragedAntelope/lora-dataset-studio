"""Settings and captioner/engine registries.

Everything here is overridable via environment variables prefixed LDS_
(or a .env file in the repo root — see .env.example).
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parent.parent

# Load .env into the process environment so unprefixed keys (GEMINI_API_KEY,
# GROQ_API_KEY, HF_TOKEN for the gated SAM3 download) work from there too.
load_dotenv(REPO_ROOT / ".env")

# Base captioning instruction, tuned for LoRA training captions: describe what
# VARIES between images (pose/angle/setting/lighting), not the subject's fixed
# identity — identity is what the trigger token learns.
_BASE_PROMPT = (
    "Describe this image in one flowing paragraph of natural language for use as an "
    'image-generation training caption. Refer to the main subject as "{subject}". '
    "Describe the subject's pose, the camera angle and framing, the setting, and the "
    "lighting. If the background is plain, just say so briefly. Do not describe the "
    "subject's fixed physical appearance in exhaustive detail (identity is learned "
    "separately); focus on what varies between images. Do not mention that this is "
    "AI-generated, a film still, or training data. Output only the caption paragraph, "
    "nothing else."
)

# JoyCaption was trained on a specific instruction format and supports a
# documented "refer to them as X" directive — use its convention, not ours.
_JOYCAPTION_PROMPT = (
    "Write a descriptive caption for this image in a formal tone, as one flowing "
    "paragraph. If there is a person or character in the image you must refer to "
    "them as {subject}. Describe the pose, the camera angle and framing, the "
    "setting, and the lighting, but do not describe the subject's fixed physical "
    "appearance in exhaustive detail. Do not mention that the image is AI-generated, "
    "a film still, or training data."
)

# The NSFW-Caption finetune responds best when explicitness is requested plainly.
_NSFW_PROMPT = _BASE_PROMPT + (
    " If the image contains nudity or explicit content, describe it plainly and "
    "directly without euphemism."
)


class CaptionerSpec(BaseModel):
    key: str
    label: str
    # "transformers" (local GPU) | "openai" (Groq / LM Studio / Ollama) | "gemini" (Google)
    backend: str = "transformers"
    hf_id: str = ""  # transformers backend
    model: str = ""  # openai/gemini backend model id ("" = server default / first listed)
    base_url: str = ""  # openai backend endpoint
    api_key_env: str = ""  # env var holding the key ("" = no key needed)
    min_interval_s: float = 0.0  # request spacing (free-tier rate limits)
    prompt_template: str = _BASE_PROMPT  # must contain {subject}
    # "qwen_vl" and "llava" share one transformers code path but need
    # different chat-template quirks (JoyCaption wants its system prompt).
    prompt_style: str = "qwen_vl"
    vram_note: str = ""
    nsfw_capable: bool = True
    cost_note: str = "free"


CAPTIONERS: list[CaptionerSpec] = [
    CaptionerSpec(
        key="qwen3vl",
        label="Local: Qwen3-VL-8B Instruct (heretic)",
        hf_id="heretic-org/Qwen-3-VL-8B-Instruct-heretic",
        vram_note="~17 GB bf16",
    ),
    CaptionerSpec(
        key="joycaption",
        label="Local: JoyCaption Beta One",
        hf_id="fancyfeast/llama-joycaption-beta-one-hf-llava",
        prompt_style="llava",
        prompt_template=_JOYCAPTION_PROMPT,
        vram_note="~17 GB bf16",
    ),
    CaptionerSpec(
        key="qwen3vl-nsfw",
        label="Local: Qwen3-VL-8B NSFW-Caption V4.5",
        hf_id="Disty0/Qwen3-VL-8B-NSFW-Caption-V4.5",
        prompt_template=_NSFW_PROMPT,
        vram_note="~17 GB bf16",
    ),
    CaptionerSpec(
        key="gemini-flash",
        label="Cloud: Gemini 2.5 Flash (Google API key, SFW)",
        backend="gemini",
        model="gemini-2.5-flash",
        api_key_env="GEMINI_API_KEY",
        nsfw_capable=False,
        cost_note="~$0.001/image (estimate at build time — check current pricing)",
    ),
    CaptionerSpec(
        key="groq-llama4-scout",
        label="Cloud: Groq Llama 4 Scout (free tier, SFW)",
        backend="openai",
        base_url="https://api.groq.com/openai/v1",
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        api_key_env="GROQ_API_KEY",
        min_interval_s=2.5,
        nsfw_capable=False,
        cost_note="free tier (rate-limited; 30K TPM)",
    ),
    CaptionerSpec(
        key="groq-qwen3.6",
        label="Cloud: Groq Qwen3.6 27B (free tier, SFW)",
        backend="openai",
        base_url="https://api.groq.com/openai/v1",
        model="qwen/qwen3.6-27b",
        api_key_env="GROQ_API_KEY",
        min_interval_s=3.0,  # lower free-tier TPM (8K) than Scout
        nsfw_capable=False,
        cost_note="free tier (rate-limited; 8K TPM, lower throughput than Scout)",
    ),
    CaptionerSpec(
        key="lmstudio",
        label="Advanced: LM Studio (whatever vision model is loaded)",
        backend="openai",
        base_url="http://127.0.0.1:1234/v1",
    ),
    CaptionerSpec(
        key="ollama",
        label="Advanced: Ollama (first vision model served)",
        backend="openai",
        base_url="http://127.0.0.1:11434/v1",
    ),
]

CAPTIONERS_BY_KEY = {c.key: c for c in CAPTIONERS}

ENGINES = {
    "gemini": "Cloud - Gemini image model (best identity fidelity, SFW only)",
    "comfyui": "Local - ComfyUI Qwen Image Edit 2511 (free, private, uncensored)",
}

# Known Gemini image models -> USD per standard-resolution (1K) image.
# ESTIMATES captured at build time — actual costs are billed by Google to the
# user's own API key; the UI can live-pull the current model list.
CLOUD_IMAGE_PRICES = {
    "gemini-3-pro-image-preview": 0.134,  # Nano Banana Pro (1K-2K)
    "gemini-3.1-flash-image-preview": 0.067,  # Nano Banana 2 (1K)
    "gemini-2.5-flash-image": 0.039,  # Nano Banana (1K)
}

# Local cache for the live Gemini model list so the dropdown can be populated
# without an API call on every UI load. 24-hour TTL; falls back to stale cache,
# then to CLOUD_IMAGE_PRICES if the API is unreachable.
CACHE_DIR = REPO_ROOT / ".cache"
MODEL_CACHE_FILE = CACHE_DIR / "gemini_image_models.json"
MODEL_CACHE_TTL_HOURS = 24


def load_cloud_model_cache() -> list[dict] | None:
    """Return cached model entries if the cache exists and is fresh."""
    if not MODEL_CACHE_FILE.exists():
        return None
    try:
        data = json.loads(MODEL_CACHE_FILE.read_text(encoding="utf-8"))
        cached_at = datetime.fromisoformat(data["cached_at"])
        age = datetime.now(tz=timezone.utc) - cached_at
        if age > timedelta(hours=MODEL_CACHE_TTL_HOURS):
            return None
            return None
        return data.get("models", [])
    except Exception:
        return None


def save_cloud_model_cache(models: list[dict]) -> None:
    """Persist the live model list with a timestamp."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_CACHE_FILE.write_text(
        json.dumps(
            {"cached_at": datetime.now(tz=timezone.utc).isoformat(), "models": models},
            indent=2,
        ),
        encoding="utf-8",
    )

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="LDS_", env_file=REPO_ROOT / ".env",
                                      extra="ignore")

    comfy_url: str = "http://127.0.0.1:8188"
    gemini_api_key: str = ""  # also read from GEMINI_API_KEY if unset
    groq_api_key: str = ""  # also read from GROQ_API_KEY if unset
    gemini_image_model: str = "gemini-3-pro-image-preview"

    target_long_side: int = 1024
    default_engine: str = "gemini"
    default_captioner: str = "qwen3vl"
    # "builtin" = SAM3 via transformers in-process; "comfyui" = SAM3 workflow
    isolation_backend: str = "builtin"
    sam3_hf_id: str = "facebook/sam3"  # gated: accept license + `hf auth login`
    # "auto" = model restore through ComfyUI when reachable, else basic Lanczos;
    # "comfyui" = require ComfyUI; "basic" = never call ComfyUI
    restore_backend: str = "auto"

    output_root: Path = REPO_ROOT / "datasets"
    runs_dir: Path = REPO_ROOT / "runs"

    # ComfyUI model filenames used by the optional workflow templates
    # (relative to your ComfyUI models folders — see docs/comfyui-setup.md)
    qwen_edit_model: str = "qwen_image_edit_2511_int8_convrot.safetensors"
    angles_lora: str = "qwen/Qwen-Image-Edit-2511-Multiple-Angles-LoRA.safetensors"
    upscale_model: str = "4xNomosWebPhoto_RealPLKSR.safetensors"
    dejpg_model: str = "1xDeJPG_OmniSR.pth"
    sam3_checkpoint: str = "sam3.1_multiplex_fp16.safetensors"

    def resolved_key(self, name: str) -> str:
        import os

        attr = getattr(self, name.lower(), "")
        return attr or os.environ.get(name.upper(), "")

    def resolved_gemini_key(self) -> str:
        return self.resolved_key("gemini_api_key") or self.resolved_key("GEMINI_API_KEY")

    def resolved_groq_key(self) -> str:
        return self.resolved_key("groq_api_key") or self.resolved_key("GROQ_API_KEY")


settings = Settings()


def list_images(folder: Path) -> list[Path]:
    """All images directly inside `folder`, sorted by name."""
    if not folder.is_dir():
        return []
    return sorted(p for p in folder.iterdir()
                  if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
