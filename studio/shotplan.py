"""Default shot plan: camera angles, poses, and scene/lighting variants."""

from __future__ import annotations

from pydantic import BaseModel


class Shot(BaseModel):
    id: str
    kind: str  # "angle" | "pose" | "scene"
    local_prompt: str  # Qwen-Image-Edit-2511 prompt (angles use the <sks> LoRA grammar)
    cloud_prompt: str  # plain-English instruction for Nano Banana Pro
    # Rear views hallucinate when generated straight from a front reference;
    # chain them off a generated side view instead (stepwise rotation).
    chain_from: str = ""


# (id_suffix, <sks> grammar phrase, plain-English equivalent, chain_from)
_ANGLES = [
    ("front", "front view eye-level shot medium shot", "seen directly from the front at eye level, full body visible", ""),
    ("front-right", "front-right quarter view eye-level shot medium shot", "seen from a front-right three-quarter angle at eye level, full body visible", ""),
    ("right", "right side view eye-level shot medium shot", "seen directly from the right side in full profile, full body visible", ""),
    ("back-right", "back-right quarter view eye-level shot medium shot", "seen from a back-right three-quarter angle, full body visible", "angle-right"),
    ("back", "back view eye-level shot medium shot", "seen directly from behind, full body visible", "angle-right"),
    ("back-left", "back-left quarter view eye-level shot medium shot", "seen from a back-left three-quarter angle, full body visible", "angle-left"),
    ("left", "left side view eye-level shot medium shot", "seen directly from the left side in full profile, full body visible", ""),
    ("front-left", "front-left quarter view eye-level shot medium shot", "seen from a front-left three-quarter angle at eye level, full body visible", ""),
    ("low-angle", "front view low-angle shot medium shot", "photographed from a low camera angle looking up", ""),
    ("high-angle", "front-right quarter view high-angle shot medium shot", "photographed from a high camera angle looking down", ""),
    ("closeup", "front view eye-level shot close-up", "a close-up of the head and upper body, front view", ""),
    ("wide", "front-left quarter view eye-level shot wide shot", "a wide shot showing the entire body with room around it", ""),
]

_POSES = [
    ("seated", "sitting down on a simple wooden stool, hands resting naturally"),
    ("lying", "lying down on the ground on its side, relaxed"),
    ("walking", "walking forward mid-stride"),
    ("crouching", "crouching low to the ground"),
    ("arms-raised", "with both arms raised overhead"),
    ("leaning", "leaning against a wall casually"),
    ("action", "in a dynamic action pose, mid-movement"),
    ("looking-back", "standing and looking back over one shoulder"),
]

_SCENES = [
    ("outdoor-day", "standing outdoors in daylight in an open field"),
    ("city-street", "standing on a city street at dusk"),
    ("indoor-warm", "in a warmly lit interior room"),
    ("studio-neutral", "against a plain neutral gray studio background with soft even lighting"),
    ("dramatic-light", "lit by dramatic hard side lighting against a dark background"),
    ("forest", "standing in a forest with dappled sunlight"),
    ("night", "outdoors at night under cool moonlight"),
    ("golden-hour", "outdoors at golden hour with warm backlighting"),
]


def default_plan(subject: str = "the character") -> list[Shot]:
    shots: list[Shot] = []
    for suffix, grammar, plain, chain in _ANGLES:
        shots.append(
            Shot(
                id=f"angle-{suffix}",
                kind="angle",
                local_prompt=f"<sks> {grammar}",
                cloud_prompt=(
                    f"Generate a photorealistic image of exactly the same {subject} from the reference image(s), "
                    f"identical in every physical detail, {plain}. Keep the same overall style and realism as the reference."
                ),
                chain_from=chain,
            )
        )
    for suffix, pose in _POSES:
        shots.append(
            Shot(
                id=f"pose-{suffix}",
                kind="pose",
                local_prompt=f"the same {subject}, {pose}, photorealistic, consistent identity",
                cloud_prompt=(
                    f"Generate a photorealistic image of exactly the same {subject} from the reference image(s), "
                    f"identical in every physical detail, {pose}."
                ),
            )
        )
    for suffix, scene in _SCENES:
        shots.append(
            Shot(
                id=f"scene-{suffix}",
                kind="scene",
                local_prompt=f"the same {subject}, {scene}, photorealistic, consistent identity",
                cloud_prompt=(
                    f"Generate a photorealistic image of exactly the same {subject} from the reference image(s), "
                    f"identical in every physical detail, {scene}."
                ),
            )
        )
    return shots
