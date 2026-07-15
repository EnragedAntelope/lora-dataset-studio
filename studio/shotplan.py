"""Default shot plan: curated camera angles, poses, emotions, and settings.

Instead of generating standalone "scene/lighting" shots that repeat the same
standing pose with different backgrounds, each shot here is a unique
combination of angle/pose/emotion/setting. This keeps the dataset size
manageable (24 shots) while maximizing the diversity the LoRA actually learns.

The roadmap (done + deferred items) lives in `docs/ARCHITECTURE.md` under
"Roadmap / deferred", not here.
"""

from __future__ import annotations

from pydantic import BaseModel


class Shot(BaseModel):
    id: str
    kind: str  # "angle" | "pose" | "emotion"
    local_prompt: str  # Qwen-Image-Edit-2511 prompt (angles use the <sks> LoRA grammar)
    cloud_prompt: str  # plain-English instruction for Nano Banana
    # Rear views hallucinate when generated straight from a front reference;
    # chain them off a generated side view instead (stepwise rotation).
    chain_from: str = ""
    # Emotion and setting are stored explicitly so the dataframe is readable
    # and so future tooling can filter/group shots by these dimensions.
    emotion: str = ""
    setting: str = ""
    # Wardrobe/outfit override. Empty = keep the reference's default clothing
    # (no identity drift). When set, "wearing {outfit}" is injected into both
    # the local and cloud prompts so clothing can vary across the dataset.
    outfit: str = ""


# Each tuple is: (id_suffix, kind, <sks> grammar or pose stub, plain-English
# description, chain_from, emotion, setting).
#
# Design goals:
# - 9 angles: the core turnaround, each with a different setting/lighting so no
#   two are the same generic standing shot.
# - 8 poses: each pose is paired with a setting and emotion; lighting is part
#   of the setting rather than a separate repeated pose.
# - 7 emotions: close-up expression shots with varied angles and settings.
# - Total: 24 shots (down from 28) while improving per-shot diversity.
#
# Settings are written as "natural" lighting/environment phrases so both the
# local and cloud prompts read like plain English.
_SHOTS = [
    # ---------- angles ----------
    (
        "front",
        "angle",
        "front view eye-level shot medium shot",
        "seen directly from the front at eye level, full body visible",
        "",
        "neutral",
        "against a plain neutral gray studio background with soft even lighting",
    ),
    (
        "front-right",
        "angle",
        "front-right quarter view eye-level shot medium shot",
        "seen from a front-right three-quarter angle at eye level, full body visible",
        "",
        "neutral",
        "outdoors in daylight in an open field",
    ),
    (
        "right",
        "angle",
        "right side view eye-level shot medium shot",
        "seen directly from the right side in full profile, full body visible",
        "",
        "neutral",
        "in a warmly lit interior room",
    ),
    (
        "back-right",
        "angle",
        "back-right quarter view eye-level shot medium shot",
        "seen from a back-right three-quarter angle, full body visible",
        "angle-right",
        "neutral",
        "on a city street at dusk",
    ),
    (
        "back",
        "angle",
        "back view eye-level shot medium shot",
        "seen directly from behind, full body visible",
        "angle-right",
        "neutral",
        "against a plain neutral gray studio background with soft even lighting",
    ),
    (
        "back-left",
        "angle",
        "back-left quarter view eye-level shot medium shot",
        "seen from a back-left three-quarter angle, full body visible",
        "angle-left",
        "neutral",
        "standing in a forest with dappled sunlight",
    ),
    (
        "left",
        "angle",
        "left side view eye-level shot medium shot",
        "seen directly from the left side in full profile, full body visible",
        "",
        "neutral",
        "outdoors at golden hour with warm backlighting",
    ),
    (
        "front-left",
        "angle",
        "front-left quarter view eye-level shot medium shot",
        "seen from a front-left three-quarter angle at eye level, full body visible",
        "",
        "neutral",
        "lit by dramatic hard side lighting against a dark background",
    ),
    (
        "low",
        "angle",
        "front view low-angle shot medium shot",
        "photographed from a low camera angle looking up",
        "",
        "confident",
        "outdoors at night under cool moonlight",
    ),
    # ---------- poses ----------
    (
        "seated",
        "pose",
        "sitting down on a simple wooden stool, hands resting naturally",
        "sitting down on a simple wooden stool, hands resting naturally",
        "",
        "relaxed",
        "in a warmly lit interior room",
    ),
    (
        "lying",
        "pose",
        "lying down on the ground on its side, relaxed",
        "lying down on the ground on its side, relaxed",
        "",
        "peaceful",
        "outdoors in daylight in an open field",
    ),
    (
        "walking",
        "pose",
        "walking forward mid-stride",
        "walking forward mid-stride",
        "",
        "determined",
        "on a city street at dusk",
    ),
    (
        "crouching",
        "pose",
        "crouching low to the ground",
        "crouching low to the ground",
        "",
        "alert",
        "standing in a forest with dappled sunlight",
    ),
    (
        "arms-raised",
        "pose",
        "with both arms raised overhead",
        "with both arms raised overhead",
        "",
        "triumphant",
        "lit by dramatic hard side lighting against a dark background",
    ),
    (
        "leaning",
        "pose",
        "leaning against a wall casually",
        "leaning against a wall casually",
        "",
        "casual",
        "on a city street at dusk",
    ),
    (
        "action",
        "pose",
        "in a dynamic action pose, mid-movement",
        "in a dynamic action pose, mid-movement",
        "",
        "intense",
        "outdoors in daylight in an open field",
    ),
    (
        "looking-back",
        "pose",
        "standing and looking back over one shoulder",
        "standing and looking back over one shoulder",
        "",
        "playful",
        "outdoors at golden hour with warm backlighting",
    ),
    # ---------- emotions (close-ups) ----------
    (
        "smiling",
        "emotion",
        "close-up of the face, smiling expression",
        "a close-up of the face and upper shoulders, smiling warmly",
        "",
        "smiling",
        "against a soft neutral studio background",
    ),
    (
        "serious",
        "emotion",
        "close-up of the face, serious expression",
        "a close-up of the face and upper shoulders, serious expression",
        "",
        "serious",
        "lit by dramatic hard side lighting against a dark background",
    ),
    (
        "surprised",
        "emotion",
        "close-up of the face, surprised expression",
        "a close-up of the face and upper shoulders, surprised expression",
        "",
        "surprised",
        "in a warmly lit interior room",
    ),
    (
        "laughing",
        "emotion",
        "close-up of the face, laughing expression",
        "a close-up of the face and upper shoulders, laughing openly",
        "",
        "laughing",
        "outdoors in daylight in an open field",
    ),
    (
        "contemplative",
        "emotion",
        "close-up of the face, contemplative expression",
        "a close-up of the face and upper shoulders, contemplative gaze",
        "",
        "contemplative",
        "outdoors at night under cool moonlight",
    ),
    (
        "confident",
        "emotion",
        "close-up of the face, confident expression",
        "a close-up of the face and upper shoulders, confident expression",
        "",
        "confident",
        "outdoors at golden hour with warm backlighting",
    ),
    (
        "sad",
        "emotion",
        "close-up of the face, sad expression",
        "a close-up of the face and upper shoulders, sad expression",
        "",
        "sad",
        "in a warmly lit interior room",
    ),
]


def _build_local_prompt(
    kind: str, grammar_or_pose: str, setting: str, emotion: str, outfit: str = ""
) -> str:
    """Build the ComfyUI/Qwen-Edit prompt.

    Angle shots keep the tight <sks> Multiple-Angles LoRA grammar so the LoRA
    can do its job; pose/emotion shots are plain English with setting/lighting
    folded in. The emotion is appended so it influences expression without
    breaking the LoRA grammar for angles. An explicit outfit, when given, is
    appended after the grammar/pose so clothing can vary.
    """
    wardrobe = f", wearing {outfit}" if outfit else ""
    if kind == "angle":
        prompt = f"<sks> {grammar_or_pose}"
        if emotion and emotion != "neutral":
            prompt += f", {emotion} expression"
        return prompt + wardrobe
    return (
        f"the same {{subject}}, {grammar_or_pose}{wardrobe}, in {setting}, "
        f"{emotion} mood, photorealistic, consistent identity"
    )


def _build_cloud_prompt(
    subject: str, kind: str, description: str, setting: str, emotion: str, outfit: str = ""
) -> str:
    """Build the plain-English Nano Banana instruction."""
    parts = [
        f"Generate a photorealistic image of exactly the same {subject} "
        "from the reference image(s), identical in every physical detail",
    ]
    if kind == "emotion":
        parts.append(f", {description}")
    elif kind == "pose":
        parts.append(f", {description}")
        parts.append(f", in {setting}")
        if emotion and emotion != "neutral":
            parts.append(f", with a {emotion} expression")
    else:  # angle
        parts.append(f", {description}")
        parts.append(f", in {setting}")
        if emotion and emotion != "neutral":
            parts.append(f", with a {emotion} expression")
    if outfit:
        parts.append(f", wearing {outfit}")
    parts.append(". Keep the same overall style and realism as the reference.")
    return "".join(parts)


def apply_wardrobe(shot: Shot) -> Shot:
    """Return a copy of `shot` with its outfit folded into the prompts.

    The outfit column is the source of truth: whatever the user types there is
    injected as "wearing {outfit}" at generation time, so the column stays
    functional even if the prompt cells were edited by hand. Idempotent — a
    prompt that already mentions the outfit is left untouched.
    """
    if not shot.outfit:
        return shot
    phrase = f"wearing {shot.outfit}"
    local = shot.local_prompt
    cloud = shot.cloud_prompt
    if phrase.lower() not in local.lower():
        local = f"{local}, {phrase}"
    if phrase.lower() not in cloud.lower():
        # Insert before the trailing "Keep the same..." sentence when present.
        if ". Keep the same" in cloud:
            head, _, tail = cloud.partition(". Keep the same")
            cloud = f"{head}, {phrase}. Keep the same{tail}"
        else:
            cloud = f"{cloud}, {phrase}"
    return shot.model_copy(update={"local_prompt": local, "cloud_prompt": cloud})


# Props carried in a reference image get copied into every generated shot, and a
# dataset where 20/24 images show the same backpack teaches the LoRA that the
# backpack IS the character. These clauses ask the generator to drop them.
#
# Deliberately NOT applied to `kind="angle"` local prompts: those use the <sks>
# Multiple-Angles LoRA grammar, which is trained on clean splat renders and
# degrades when prose is appended (see ARCHITECTURE.md). Diffusion models also
# handle negation poorly in a positive prompt — naming "backpack" can summon one.
# Angle shots rely on isolation instead, which removes props from the reference
# itself and is the mechanism that actually works.
_CLOUD_NO_PROPS = (
    " Show only the character and the clothing worn on their body — do not "
    "include any backpacks, bags, straps, held objects, tools, props, or "
    "accessories that appear in the reference image."
)
_LOCAL_NO_PROPS = ", without any bags or carried accessories"


def apply_prop_exclusion(shot: Shot) -> Shot:
    """Return a copy of `shot` asking the generator to omit reference props.

    Applied at generation time (like `apply_wardrobe`) rather than baked into the
    plan, so the column stays honest and hand-edited prompt cells still get the
    clause. Idempotent.
    """
    cloud = shot.cloud_prompt
    if _CLOUD_NO_PROPS.strip() not in cloud:
        cloud = f"{cloud}{_CLOUD_NO_PROPS}"
    local = shot.local_prompt
    if shot.kind != "angle" and _LOCAL_NO_PROPS not in local:
        local = f"{local}{_LOCAL_NO_PROPS}"
    return shot.model_copy(update={"local_prompt": local, "cloud_prompt": cloud})


def default_plan(subject: str = "the character") -> list[Shot]:
    """Return the curated 24-shot default plan."""
    shots: list[Shot] = []
    for suffix, kind, grammar_or_pose, description, chain, emotion, setting in _SHOTS:
        shots.append(
            Shot(
                id=f"{kind}-{suffix}",
                kind=kind,
                local_prompt=_build_local_prompt(kind, grammar_or_pose, setting, emotion),
                cloud_prompt=_build_cloud_prompt(subject, kind, description, setting, emotion),
                chain_from=chain,
                emotion=emotion,
                setting=setting,
                outfit="",
            )
        )
    return shots
