"""Subject isolation: cut the subject out onto a plain white background.

Two interchangeable backends:
- "builtin"  — SAM3 (facebook/sam3) in-process via transformers. No ComfyUI
  needed. The model is gated on Hugging Face: accept the license on the model
  page and authenticate (`hf auth login` or HF_TOKEN) before first use.
- "comfyui"  — the bundled SAM3 workflow templates, run on your ComfyUI.

Isolation matters twice: backgrounds/props can't leak into generations or the
final dataset, and edit models trained on clean renders (the Multiple-Angles
LoRA) behave far better on isolated subjects.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

from studio.config import settings


class IsolationError(Exception):
    pass


# ---------- builtin backend: SAM3 via transformers ----------

_model = None
_processor = None


def _load_sam3():
    global _model, _processor
    if _model is None:
        try:
            import torch  # noqa: F401
            from transformers import Sam3Model, Sam3Processor
        except ImportError as e:
            raise IsolationError(f"transformers/torch not available: {e}")
        try:
            _model = Sam3Model.from_pretrained(settings.sam3_hf_id, device_map="auto")
            _processor = Sam3Processor.from_pretrained(settings.sam3_hf_id)
        except Exception as e:
            raise IsolationError(
                f"Could not load {settings.sam3_hf_id} — it is a gated model: accept the "
                f"license at https://huggingface.co/{settings.sam3_hf_id} and authenticate "
                f"(`hf auth login` or set HF_TOKEN). Original error: {e}"
            )
    return _model, _processor


def unload() -> None:
    """Free the SAM3 weights (e.g. before loading a large captioner)."""
    global _model, _processor
    if _model is None:
        return
    import gc

    import torch

    _model = _processor = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _segment(image: Image.Image, prompt: str, threshold: float = 0.5) -> np.ndarray:
    """Union of all instance masks matching the text prompt (bool HxW array)."""
    import torch

    model, processor = _load_sam3()
    inputs = processor(images=image, text=prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model(**inputs)
    results = processor.post_process_instance_segmentation(
        outputs, threshold=threshold, mask_threshold=0.5,
        target_sizes=inputs.get("original_sizes").tolist(),
    )[0]
    masks = results["masks"]
    if len(masks) == 0:
        return np.zeros((image.height, image.width), dtype=bool)
    union = np.zeros((image.height, image.width), dtype=bool)
    for m in masks:
        union |= np.asarray(m.cpu(), dtype=bool)
    return union


def _dilate(mask: np.ndarray, px: int) -> np.ndarray:
    """Grow a boolean mask by ~px pixels (used on the exclusion mask so prop
    edges don't survive as halos around the removed object)."""
    if px <= 0 or not mask.any():
        return mask
    img = Image.fromarray(mask.astype(np.uint8) * 255)
    size = max(3, px * 2 + 1)
    if size % 2 == 0:
        size += 1
    return np.asarray(img.filter(ImageFilter.MaxFilter(size))) > 127


def isolate_builtin(image_path: Path, out_path: Path, subject_prompt: str = "character",
                    exclude_prompt: str = "") -> Path:
    image = Image.open(image_path).convert("RGB")
    subject = _segment(image, subject_prompt)
    if not subject.any():
        raise IsolationError(
            f"SAM3 found no '{subject_prompt}' in {image_path.name} — adjust the "
            f"subject prompt or turn isolation off for this image."
        )
    if exclude_prompt.strip():
        # SAM3 merges props the subject grips into the subject segment, so held
        # objects need their own segmentation, grown slightly, then subtracted.
        exclude = _segment(image, exclude_prompt)
        exclude = _dilate(exclude, px=max(2, max(image.size) // 200))
        subject &= ~exclude

    arr = np.asarray(image)
    white = np.full_like(arr, 255)
    out = np.where(subject[..., None], arr, white)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(out).save(out_path, "PNG")
    return out_path


# ---------- comfyui backend ----------

def isolate_comfyui(image_path: Path, out_path: Path, subject_prompt: str = "character",
                    exclude_prompt: str = "") -> Path:
    from studio import comfy_api

    uploaded = comfy_api.upload_image(image_path)
    if exclude_prompt.strip():
        graph = comfy_api.load_template("isolate_exclude")
        graph["122"]["inputs"]["text"] = exclude_prompt
    else:
        graph = comfy_api.load_template("isolate_subject")
    graph["1"]["inputs"]["image"] = uploaded
    graph["121"]["inputs"]["text"] = subject_prompt
    refs = comfy_api.run_prompt(graph, timeout=420)
    return comfy_api.fetch_image(refs[0], out_path)


# ---------- dispatch ----------

def isolate_subject(image_path: Path, out_path: Path, subject_prompt: str = "character",
                    exclude_prompt: str = "", backend: str = "") -> Path:
    backend = backend or settings.isolation_backend
    if backend == "comfyui":
        return isolate_comfyui(image_path, out_path, subject_prompt, exclude_prompt)
    return isolate_builtin(image_path, out_path, subject_prompt, exclude_prompt)
