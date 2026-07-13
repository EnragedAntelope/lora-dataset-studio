"""Cloud engine: Gemini image models (Nano Banana family) via google-genai.

Costs are billed by Google to YOUR API key. Prices shown in the app are
estimates captured at build time — always check current Google pricing.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from studio.config import (
    CLOUD_IMAGE_PRICES,
    MODEL_CACHE_FILE,
    load_cloud_model_cache,
    save_cloud_model_cache,
    settings,
)
from studio.engines.base import GenerationError
from studio.shotplan import Shot

MAX_REFERENCE_IMAGES = 14

# Re-exported so tests can monkeypatch a single module-level path.
MODEL_CACHE_FILE = MODEL_CACHE_FILE


def _load_model_cache() -> list[dict] | None:
    """Thin wrapper around the config cache, kept here for testability."""
    return load_cloud_model_cache()


def _save_model_cache(models: list[dict]) -> None:
    """Thin wrapper around the config cache, kept here for testability."""
    save_cloud_model_cache(models)


def _model_label(name: str, price: float | None) -> str:
    if price is None:
        return f"{name}  (price unknown)"
    return f"{name}  (${price:.3f}/img est.)"


def _fallback_models() -> list[tuple[str, str]]:
    """Static fallback when live listing is impossible."""
    return [(_model_label(m, p), m) for m, p in CLOUD_IMAGE_PRICES.items()]


def list_image_models(force_refresh: bool = False) -> list[tuple[str, str]]:
    """Return image-capable Gemini models as [(display_label, model_id), ...].

    Uses a 24-hour local cache so the UI dropdown loads instantly. If the
    cache is stale/missing, live-pull from the API and persist. Falls back to
    the static price table if the API is unreachable or no key is configured.
    """
    if not force_refresh:
        cached = _load_model_cache()
        if cached:
            return [
                (_model_label(m["model_id"], m.get("price")), m["model_id"])
                for m in cached
            ]

    key = settings.resolved_gemini_key()
    if not key:
        return _fallback_models()

    from google import genai

    try:
        client = genai.Client(api_key=key)
        found: list[dict] = []
        for m in client.models.list():
            name = m.name.removeprefix("models/")
            if "image" not in name or "imagen" in name:  # imagen = t2i only, no reference edit
                continue
            # Skip deprecated / shutdown models when the API lists them.
            if any(tag in name for tag in ("-shut-down", "deprecated", "-experimental")):
                continue
            price = CLOUD_IMAGE_PRICES.get(name)
            found.append(
                {
                    "model_id": name,
                    "display_name": getattr(m, "display_name", name),
                    "price": price,
                    "cached_at": datetime.now(tz=timezone.utc).isoformat(),
                }
            )
        if found:
            _save_model_cache(found)
            return [
                (_model_label(f["model_id"], f["price"]), f["model_id"])
                for f in found
            ]
            _save_model_cache(found)
            return [(_model_label(f["model_id"], f["price"]), f["model_id"]) for f in found]
    except Exception:
        # Live pull failed; try stale cache as a last resort before falling
        # back to the static dict.
        stale = _load_model_cache()
        if stale:
            return [
                (_model_label(m["model_id"], m.get("price")), m["model_id"])
                for m in stale
            ]
            return [(_model_label(m["model_id"], m.get("price")), m["model_id"]) for m in stale]

    return _fallback_models()


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
