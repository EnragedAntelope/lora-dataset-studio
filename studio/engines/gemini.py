"""Cloud engine: Gemini image models (Nano Banana family) via google-genai.

Costs are billed by Google to YOUR API key. Prices shown in the app are
estimates captured at build time — always check current Google pricing.
"""

from __future__ import annotations

from pathlib import Path

from studio.config import settings
from studio.engines.base import GenerationError
from studio.shotplan import Shot

MAX_REFERENCE_IMAGES = 14


def list_image_models() -> list[tuple[str, str]]:
    """Live-pull image-capable Gemini models: [(model_id, price_label), ...]."""
    from google import genai

    from studio.config import CLOUD_IMAGE_PRICES

    key = settings.resolved_gemini_key()
    if not key:
        return [(m, f"${p:.3f}/img") for m, p in CLOUD_IMAGE_PRICES.items()]
    client = genai.Client(api_key=key)
    found = []
    for m in client.models.list():
        name = m.name.removeprefix("models/")
        if "image" not in name or "imagen" in name:  # imagen = t2i only, no reference edit
            continue
        price = CLOUD_IMAGE_PRICES.get(name)
        found.append((name, f"${price:.3f}/img" if price else "price unknown"))
    return found or [(m, f"${p:.3f}/img") for m, p in CLOUD_IMAGE_PRICES.items()]


class GeminiEngine:
    name = "gemini"

    def __init__(self, model: str = "") -> None:
        key = settings.resolved_gemini_key()
        if not key:
            raise GenerationError(
                "No Gemini API key found. Set GEMINI_API_KEY (or LDS_GEMINI_API_KEY in "
                ".env) to use the cloud engine, or switch to the local ComfyUI engine. "
                "Get a key at https://aistudio.google.com/apikey"
            )
        from google import genai  # deferred so local-only installs never need it configured

        self._client = genai.Client(api_key=key)
        self._model = model or settings.gemini_image_model

    def generate(self, sources: list[Path], shot: Shot, out_path: Path, seed: int) -> Path:
        from google.genai import types

        parts: list = [
            types.Part.from_bytes(data=p.read_bytes(), mime_type="image/png")
            for p in sources[:MAX_REFERENCE_IMAGES]
        ]
        parts.append(shot.cloud_prompt)

        last_err: Exception | None = None
        for _ in range(2):
            try:
                resp = self._client.models.generate_content(
                    model=self._model,
                    contents=parts,
                    config=types.GenerateContentConfig(
                        response_modalities=["TEXT", "IMAGE"],
                    ),
                )
                for cand in resp.candidates or []:
                    for part in cand.content.parts or []:
                        if getattr(part, "inline_data", None) and part.inline_data.data:
                            out_path.parent.mkdir(parents=True, exist_ok=True)
                            out_path.write_bytes(part.inline_data.data)
                            return out_path
                # No image part usually means a content refusal
                text = "".join(
                    p.text or ""
                    for c in (resp.candidates or [])
                    for p in (c.content.parts or [])
                    if getattr(p, "text", None)
                )
                raise GenerationError(f"no image returned ({text[:200] or 'refused'})")
            except GenerationError as e:
                last_err = e
                break  # refusals don't get better with a retry
            except Exception as e:  # transient API errors
                last_err = e
        raise GenerationError(f"shot {shot.id}: {last_err}")
