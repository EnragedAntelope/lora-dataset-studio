"""Captioning: natural-language training captions behind one Captioner interface.

Backends (see CAPTIONERS in config.py):
- "transformers" — local VLM on your GPU (Qwen3-VL, JoyCaption, NSFW finetune)
- "gemini"       — Google Gemini via your API key (costs billed to you)
- "openai"       — any OpenAI-compatible server (Groq cloud, LM Studio, Ollama)

Each captioner spec carries its own prompt template, tuned to how that model
was trained (JoyCaption has a documented instruction convention; the NSFW
finetune wants explicitness requested plainly).

Fully standalone: `caption_folder()` tags any folder of images with .txt
sidecars — no other pipeline stage required.
"""

from __future__ import annotations

import base64
import re
import time
from pathlib import Path
from typing import Callable

import httpx
from PIL import Image

from studio.config import CAPTIONERS_BY_KEY, CaptionerSpec, settings

_JOYCAPTION_SYSTEM = "You are a helpful image captioner."


class Captioner:
    def __init__(self, key: str, model_override: str = "",
                 spec_overrides: dict | None = None) -> None:
        self.spec: CaptionerSpec = CAPTIONERS_BY_KEY[key]
        # spec_overrides carries runtime config for the "custom" endpoint
        # (base_url / api_key_env / min_interval_s / model) so users can point
        # at any OpenAI-compatible server without editing config.py.
        if spec_overrides:
            self.spec = self.spec.model_copy(update=spec_overrides)
        # Let the UI pick a specific model (e.g. a live-refreshed Gemini model,
        # or a custom OpenAI-compatible endpoint's model) without editing config.
        self.model = (model_override or self.spec.model).strip()
        self._model = None
        self._processor = None
        self._last_request = 0.0

    # ---------- shared ----------

    def caption(self, image_path: Path, subject: str = "the character") -> str:
        instruction = self.spec.prompt_template.format(subject=subject)
        if self.spec.backend == "openai":
            return _clean(self._caption_openai(image_path, instruction))
        if self.spec.backend == "gemini":
            return _clean(self._caption_gemini(image_path, instruction))
        return _clean(self._caption_transformers(image_path, instruction))

    def load(self) -> None:
        if self.spec.backend == "transformers":
            self._load_transformers()

    def unload(self) -> None:
        if self._model is None:
            return
        import gc

        import torch

        del self._model, self._processor
        self._model = self._processor = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ---------- local transformers backend ----------

    def _load_transformers(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        self._processor = AutoProcessor.from_pretrained(self.spec.hf_id)
        self._model = AutoModelForImageTextToText.from_pretrained(
            self.spec.hf_id, dtype=dtype, device_map="auto"
        )
        self._model.eval()

    def _caption_transformers(self, image_path: Path, instruction: str) -> str:
        import torch

        self._load_transformers()
        image = Image.open(image_path).convert("RGB")
        messages = []
        if self.spec.prompt_style == "llava":  # JoyCaption expects its system prompt
            messages.append({"role": "system", "content": _JOYCAPTION_SYSTEM})
        messages.append(
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": instruction},
                ],
            }
        )
        inputs = self._processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self._model.device)
        with torch.no_grad():
            out = self._model.generate(
                **inputs, max_new_tokens=300, do_sample=True, temperature=0.6, top_p=0.9
            )
        generated = out[0][inputs["input_ids"].shape[1]:]
        return self._processor.decode(generated, skip_special_tokens=True)

    # ---------- Gemini backend (costs billed to your Google API key) ----------

    def _caption_gemini(self, image_path: Path, instruction: str) -> str:
        key = settings.resolved_gemini_key()
        if not key:
            raise RuntimeError(
                f"{self.spec.label} needs GEMINI_API_KEY (or LDS_GEMINI_API_KEY in .env). "
                f"Get one at https://aistudio.google.com/apikey"
            )
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=key)
        resp = client.models.generate_content(
            model=self.model or "gemini-flash-latest",
            contents=[
                types.Part.from_bytes(data=image_path.read_bytes(), mime_type="image/png"),
                instruction,
            ],
        )
        if not resp.text:
            raise RuntimeError(f"{self.spec.label}: empty response (content refusal?)")
        return resp.text

    # ---------- OpenAI-compatible backend (Groq, LM Studio, Ollama) ----------

    def _caption_openai(self, image_path: Path, instruction: str) -> str:
        if not self.spec.base_url:
            raise RuntimeError(
                f"{self.spec.label} has no endpoint URL. Enter a base URL "
                f"(e.g. https://openrouter.ai/api/v1) and save it first."
            )
        if self.spec.min_interval_s:  # respect free-tier rate limits
            wait = self.spec.min_interval_s - (time.monotonic() - self._last_request)
            if wait > 0:
                time.sleep(wait)

        headers = {"Content-Type": "application/json"}
        if self.spec.api_key_env:
            key = settings.resolved_key(self.spec.api_key_env)
            if not key:
                raise RuntimeError(
                    f"{self.spec.label} needs the {self.spec.api_key_env} env var (or "
                    f"LDS_{self.spec.api_key_env} in .env)."
                )
            headers["Authorization"] = f"Bearer {key}"

        model = self.model or self._first_served_model(headers)
        b64 = base64.b64encode(image_path.read_bytes()).decode()
        payload = {
            "model": model,
            "max_tokens": self.spec.max_tokens,
            "temperature": 0.6,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url",
                         "image_url": {"url": f"data:image/png;base64,{b64}"}},
                        {"type": "text", "text": instruction},
                    ],
                }
            ],
            **self.spec.extra_params,  # e.g. {"reasoning_effort": "none"} for Groq Qwen
        }
        for attempt in range(4):
            r = httpx.post(f"{self.spec.base_url}/chat/completions", json=payload,
                           headers=headers, timeout=120)
            self._last_request = time.monotonic()
            if r.status_code == 429:  # rate limited: back off and retry
                time.sleep(5 * (attempt + 1))
                continue
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"]
        raise RuntimeError(f"{self.spec.label}: still rate-limited after 4 attempts")

    def _first_served_model(self, headers: dict) -> str:
        try:
            r = httpx.get(f"{self.spec.base_url}/models", headers=headers, timeout=15)
            r.raise_for_status()
        except httpx.HTTPError as e:
            raise RuntimeError(
                f"Could not reach {self.spec.label} at {self.spec.base_url} — is the "
                f"server running with a vision model loaded? ({e})"
            )
        models = r.json().get("data", [])
        if not models:
            raise RuntimeError(f"No model loaded at {self.spec.base_url} — load one first.")
        return models[0]["id"]


def _clean(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)  # reasoning models
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"^(here is|here's|caption:|sure[,!.]?)\s*", "", text, flags=re.I)
    return text.strip().strip('"')


def finalize_caption(raw: str, trigger: str, character_name: str, aliases: list[str]) -> str:
    """Apply the dataset caption template: trigger first, consistent naming."""
    caption = raw
    if character_name:
        # Replace generic nouns the VLM used for the subject with the real name
        for alias in aliases:
            caption = re.sub(rf"\bthe {re.escape(alias)}\b", character_name, caption, flags=re.I)
    if trigger and not caption.lower().startswith(trigger.lower()):
        if caption and not (character_name and caption.startswith(character_name.split()[0])):
            caption = f"{caption[0].lower()}{caption[1:]}"  # don't lowercase a proper name
        caption = f"{trigger}, {caption}" if caption else trigger
    return caption


SUBJECT_ALIASES = ["character", "creature", "figure", "subject", "person", "woman", "man"]


def caption_images(
    images: list[Path],
    captioner_key: str,
    character_name: str = "",
    trigger: str = "",
    progress: Callable[[str], None] = print,
    model_override: str = "",
    spec_overrides: dict | None = None,
) -> list[tuple[Path, str]]:
    """Caption a list of images. Standalone — no run/pipeline state needed.

    Returns (image_path, finalized_caption) pairs; the captioner model is
    loaded once and freed afterwards.
    """
    subject = character_name or "the character"
    cap = Captioner(captioner_key, model_override=model_override,
                    spec_overrides=spec_overrides)
    if cap.spec.backend == "transformers":
        from studio import comfy_api

        if comfy_api.is_up():
            progress("Freeing ComfyUI VRAM for the captioner...")
            comfy_api.free_vram()
        from studio import isolate

        isolate.unload()  # SAM3 and a 17 GB captioner don't co-fit on most GPUs
        progress(f"Loading captioner {cap.spec.label} (first run downloads weights)...")
        cap.load()
    else:
        progress(f"Captioning via {cap.spec.label}...")

    items: list[tuple[Path, str]] = []
    try:
        for i, p in enumerate(images, 1):
            progress(f"Captioning {i}/{len(images)}: {p.name}")
            raw = cap.caption(p, subject=subject)
            items.append((p, finalize_caption(raw, trigger, character_name, SUBJECT_ALIASES)))
    finally:
        cap.unload()
    return items


def caption_folder(
    folder: Path,
    captioner_key: str,
    character_name: str = "",
    trigger: str = "",
    only: list[Path] | None = None,
    progress: Callable[[str], None] = print,
    model_override: str = "",
    spec_overrides: dict | None = None,
) -> list[tuple[Path, str]]:
    """Caption images in `folder` (all, or the subset in `only`) and write
    .txt sidecars next to each image. The classic 'point at a folder and tag
    it' mode."""
    from studio.config import list_images

    images = only if only else list_images(folder)
    if not images:
        raise RuntimeError(f"No images found in {folder}")
    items = caption_images(images, captioner_key, character_name, trigger, progress,
                           model_override=model_override, spec_overrides=spec_overrides)
    for img, caption in items:
        img.with_suffix(".txt").write_text(caption, encoding="utf-8")
    progress(f"Wrote {len(items)} .txt sidecar(s) in {folder}")
    return items
