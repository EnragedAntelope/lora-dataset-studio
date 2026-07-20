"""Dedicated image-tagger backend (SmilingWolf WD v3, ONNX).

Where the VLM captioners *instruct* a general model to approximate booru tags,
this runs a purpose-built tagger that emits **canonical Danbooru tags** straight
from image features — higher tag fidelity for SDXL / Illustrious / NoobAI
datasets. It is exposed as a captioner backend (`backend="wd_tagger"`), so it
plugs into the same ③ Caption flow as everything else.

Heavy, optional deps (`onnxruntime`, `huggingface_hub`) are imported lazily
inside `WDTagger.load()`, so nobody who doesn't pick a tagger pays for them.

The pure selection/formatting logic (`select_tag_names`, `format_tag`) is kept
free of any model or file I/O so it can be unit-tested without downloading a
1 GB model or running inference.
"""

from __future__ import annotations

import csv
from pathlib import Path

# Danbooru tag categories as stored in a WD tagger's selected_tags.csv.
CATEGORY_GENERAL = 0
CATEGORY_CHARACTER = 4
CATEGORY_RATING = 9

# Tags that are literally kaomoji — underscores are part of the face, so they
# must NOT be turned into spaces the way an ordinary tag ("long_hair") is.
_KAOMOJI = {
    "0_0", "(o)_(o)", "+_+", "+_-", "._.", "<o>_<o>", "<|>_<|>", "=_=", ">_<",
    "3_3", "6_9", ">_o", "@_@", "^_^", "o_o", "u_u", "x_x", "|_|", "||_||",
}


def format_tag(name: str) -> str:
    """Danbooru raw tag -> caption form: underscores become spaces (kaomoji kept)."""
    return name if name in _KAOMOJI else name.replace("_", " ")


def select_tag_names(
    labels: list[tuple[str, int, float]],
    general_threshold: float,
    character_threshold: float,
    include_ratings: bool = False,
) -> list[str]:
    """Pick tag names above threshold, ordered general-then-character by confidence.

    `labels` is (name, category, probability). General and character tags use
    separate thresholds (character tags are specific existing-character names, so
    they need a high bar). Rating tags (safe/questionable/explicit) are dropped
    unless `include_ratings`, in which case only the single most-likely rating is
    appended. Pure function: no model, no I/O — this is the unit-tested seam.
    """
    general: list[tuple[str, float]] = []
    character: list[tuple[str, float]] = []
    ratings: list[tuple[str, float]] = []
    for name, category, prob in labels:
        if category == CATEGORY_RATING:
            ratings.append((name, prob))
        elif category == CATEGORY_CHARACTER:
            if prob >= character_threshold:
                character.append((name, prob))
        elif prob >= general_threshold:
            general.append((name, prob))

    general.sort(key=lambda x: x[1], reverse=True)
    character.sort(key=lambda x: x[1], reverse=True)
    names = [n for n, _ in general] + [n for n, _ in character]
    if include_ratings and ratings:
        names.append(max(ratings, key=lambda x: x[1])[0])
    return names


def _read_selected_tags(csv_path: Path) -> tuple[list[str], list[int]]:
    """Parse a WD tagger's selected_tags.csv -> (tag names, category ids)."""
    names: list[str] = []
    categories: list[int] = []
    with csv_path.open(encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            names.append(row["name"])
            categories.append(int(row["category"]))
    if not names:
        raise RuntimeError(f"{csv_path} has no tags — is it the right file?")
    return names, categories


class WDTagger:
    """A SmilingWolf WD v3 ONNX tagger, loaded on demand from Hugging Face.

    `hf_id` is the model repo (e.g. 'SmilingWolf/wd-eva02-large-tagger-v3');
    the model.onnx and selected_tags.csv are downloaded and cached by
    huggingface_hub on first use.
    """

    MODEL_FILE = "model.onnx"
    TAGS_FILE = "selected_tags.csv"

    def __init__(self, hf_id: str, general_threshold: float = 0.35,
                 character_threshold: float = 0.85) -> None:
        self.hf_id = hf_id
        self.general_threshold = general_threshold
        self.character_threshold = character_threshold
        self._session = None
        self._tag_names: list[str] = []
        self._categories: list[int] = []
        self._target_size = 448

    def load(self) -> None:
        if self._session is not None:
            return
        try:
            import onnxruntime as ort
        except ImportError as e:  # optional dep — only taggers need it
            raise RuntimeError(
                "The WD tagger needs onnxruntime. Install it with "
                "`pip install onnxruntime` (or onnxruntime-gpu for CUDA)."
            ) from e
        from huggingface_hub import hf_hub_download

        model_path = hf_hub_download(self.hf_id, self.MODEL_FILE)
        tags_path = hf_hub_download(self.hf_id, self.TAGS_FILE)
        self._tag_names, self._categories = _read_selected_tags(Path(tags_path))
        # Prefer CUDA if the GPU build is present; always fall back to CPU.
        providers = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider")
                     if p in ort.get_available_providers()]
        self._session = ort.InferenceSession(model_path, providers=providers)
        # WD v3 models take NHWC input; the spatial size is the 2nd axis.
        shape = self._session.get_inputs()[0].shape
        if isinstance(shape[1], int):
            self._target_size = shape[1]

    def _preprocess(self, image_path: Path):
        """SmilingWolf's recipe: white-pad to square, resize, RGB->BGR, float32 0-255."""
        import numpy as np
        from PIL import Image

        img = Image.open(image_path)
        if img.mode in ("RGBA", "LA", "P"):
            canvas = Image.new("RGBA", img.size, (255, 255, 255, 255))
            canvas.alpha_composite(img.convert("RGBA"))
            img = canvas.convert("RGB")
        else:
            img = img.convert("RGB")
        side = max(img.size)
        square = Image.new("RGB", (side, side), (255, 255, 255))
        square.paste(img, ((side - img.width) // 2, (side - img.height) // 2))
        square = square.resize((self._target_size, self._target_size), Image.BICUBIC)
        arr = np.asarray(square, dtype="float32")[:, :, ::-1]  # RGB -> BGR
        return np.expand_dims(arr, 0)

    def tag(self, image_path: Path) -> list[str]:
        """Return caption-form tag names for one image (loads on first call)."""
        self.load()
        inputs = {self._session.get_inputs()[0].name: self._preprocess(image_path)}
        preds = self._session.run(None, inputs)[0][0]  # already sigmoid probs
        labels = [(self._tag_names[i], self._categories[i], float(preds[i]))
                  for i in range(len(self._tag_names))]
        names = select_tag_names(labels, self.general_threshold, self.character_threshold)
        return [format_tag(n) for n in names]

    def unload(self) -> None:
        self._session = None
        self._tag_names = []
        self._categories = []
