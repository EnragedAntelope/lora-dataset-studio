"""Dedicated image-tagger backend (ONNX booru taggers).

Where the VLM captioners *instruct* a general model to approximate booru tags,
this runs a purpose-built tagger that emits **canonical tags** straight from
image features — higher tag fidelity for tag-trained checkpoints. Two vocabularies
are supported through one code path:

- **Danbooru** — SmilingWolf WD v3 (`selected_tags.csv`), for SDXL / Illustrious /
  NoobAI.
- **e621** — Z3D-E621-ConvNeXt (`tags-selected.csv`), for Pony / furry checkpoints.

They share preprocessing and the CSV columns we read (`name`, `category`); only
the tag file name and the meaning of the category ids differ, captured in
`SCHEMES`. Exposed as the "wd_tagger" captioner backend.

Heavy, optional deps (`onnxruntime`, `huggingface_hub`) are imported lazily inside
`Tagger.load()`. The pure selection/formatting logic (`select_tag_names`,
`format_tag`) is free of any model or file I/O so it can be unit-tested without a
download or inference.
"""

from __future__ import annotations

import csv
from pathlib import Path

# Category-id sets per tagging scheme. A tagger's tag CSV stores a numeric
# category per tag; the meaning differs between Danbooru and e621. Tags whose
# category is in neither set (Danbooru ratings; e621 meta/invalid) are dropped.
SCHEMES = {
    # Danbooru (WD taggers): 0=general, 4=character, 9=rating.
    "danbooru": {"general": frozenset({0}), "character": frozenset({4})},
    # e621 (Z3D): 0=general, 1=artist, 3=copyright, 4=character, 5=species,
    # 7=meta, 8=lore. Species reads like a general descriptor for training, so it
    # rides the general threshold; artist/copyright/character/lore are specific
    # labels held to the higher character threshold; meta/invalid are dropped.
    "e621": {"general": frozenset({0, 5}), "character": frozenset({1, 3, 4, 8})},
}

# Tags that are literally kaomoji — underscores are part of the face, so they
# must NOT be turned into spaces the way an ordinary tag ("long_hair") is.
_KAOMOJI = {
    "0_0", "(o)_(o)", "+_+", "+_-", "._.", "<o>_<o>", "<|>_<|>", "=_=", ">_<",
    "3_3", "6_9", ">_o", "@_@", "^_^", "o_o", "u_u", "x_x", "|_|", "||_||",
}


def format_tag(name: str) -> str:
    """Raw booru tag -> caption form: underscores become spaces (kaomoji kept)."""
    return name if name in _KAOMOJI else name.replace("_", " ")


def select_tag_names(
    labels: list[tuple[str, int, float]],
    general_threshold: float,
    character_threshold: float,
    scheme: str = "danbooru",
) -> list[str]:
    """Pick tag names above threshold, ordered general-then-character by confidence.

    `labels` is (name, category, probability). General-category tags use
    `general_threshold`; specific-label categories (character/artist/copyright…)
    use the higher `character_threshold` so an original subject isn't mislabelled
    as a known character. Categories outside the scheme (ratings, meta, invalid)
    are dropped. Pure function: no model, no I/O — this is the unit-tested seam.
    """
    cats = SCHEMES.get(scheme, SCHEMES["danbooru"])
    general: list[tuple[str, float]] = []
    character: list[tuple[str, float]] = []
    for name, category, prob in labels:
        if category in cats["general"]:
            if prob >= general_threshold:
                general.append((name, prob))
        elif category in cats["character"]:
            if prob >= character_threshold:
                character.append((name, prob))
    general.sort(key=lambda x: x[1], reverse=True)
    character.sort(key=lambda x: x[1], reverse=True)
    return [n for n, _ in general] + [n for n, _ in character]


def _read_selected_tags(csv_path: Path) -> tuple[list[str], list[int]]:
    """Parse a tagger's tag CSV -> (tag names, category ids).

    Handles both the WD (`tag_id,name,category,count`) and Z3D
    (`id,name,category,post_count`) layouts — we only read `name` and `category`,
    which both share.
    """
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


class Tagger:
    """An ONNX booru tagger, loaded on demand from Hugging Face.

    `hf_id` is the model repo; `tags_file` and `scheme` select the vocabulary
    (Danbooru WD vs e621 Z3D). The model + tag CSV are downloaded and cached by
    huggingface_hub on first use.
    """

    MODEL_FILE = "model.onnx"

    def __init__(self, hf_id: str, general_threshold: float = 0.35,
                 character_threshold: float = 0.85,
                 tags_file: str = "selected_tags.csv", scheme: str = "danbooru") -> None:
        self.hf_id = hf_id
        self.general_threshold = general_threshold
        self.character_threshold = character_threshold
        self.tags_file = tags_file
        self.scheme = scheme
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
                "The tagger backend needs onnxruntime. Install it with "
                "`pip install onnxruntime` (or onnxruntime-gpu for CUDA)."
            ) from e
        from huggingface_hub import hf_hub_download

        model_path = hf_hub_download(self.hf_id, self.MODEL_FILE)
        tags_path = hf_hub_download(self.hf_id, self.tags_file)
        self._tag_names, self._categories = _read_selected_tags(Path(tags_path))
        providers = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider")
                     if p in ort.get_available_providers()]
        self._session = ort.InferenceSession(model_path, providers=providers)
        # WD/Z3D models take NHWC input; the spatial size is the 2nd axis.
        shape = self._session.get_inputs()[0].shape
        if isinstance(shape[1], int):
            self._target_size = shape[1]

    def _preprocess(self, image_path: Path):
        """White-pad to square, resize, RGB->BGR, float32 0-255 (the WD/Z3D recipe)."""
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
        names = select_tag_names(labels, self.general_threshold,
                                 self.character_threshold, self.scheme)
        return [format_tag(n) for n in names]

    def unload(self) -> None:
        self._session = None
        self._tag_names = []
        self._categories = []
