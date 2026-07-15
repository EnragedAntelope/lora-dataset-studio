"""Headless CLI. Every stage is its own subcommand and runs standalone:

  python cli.py preprocess ./sources --out ./prepped
  python cli.py generate ./prepped --name "Sy Snootles" --engine comfyui
  python cli.py caption ./any/folder --trigger sysnootles      # .txt sidecars
  python cli.py export ./prepped ./generated --name "Sy Snootles"
  python cli.py build img.png --name "Sy Snootles" --trigger sysnootles  # all four
"""

from __future__ import annotations

from pathlib import Path

import typer

from studio import pipeline
from studio.config import CAPTIONERS_BY_KEY, list_images, settings
from studio.shotplan import default_plan

app = typer.Typer(add_completion=False, help=__doc__)


def _expand(paths: list[Path]) -> list[Path]:
    """Accept image files and/or folders of images."""
    out: list[Path] = []
    for p in paths:
        out.extend(list_images(p) if p.is_dir() else [p])
    if not out:
        raise typer.BadParameter("No images found in the given paths.")
    return out


def _echo_cloud_estimate(engine: str, cloud_model: str, n_shots: int) -> None:
    """Warn what a cloud run will cost before it starts billing the user."""
    if engine != "gemini":
        return
    from studio.config import CLOUD_IMAGE_PRICES, load_cloud_model_cache

    model_id = cloud_model or settings.gemini_image_model
    price = CLOUD_IMAGE_PRICES.get(model_id)
    for m in load_cloud_model_cache() or []:
        if m.get("model_id") == model_id and m.get("price") is not None:
            price = m["price"]
            break
    if price:
        typer.echo(f"Cloud engine: ~${n_shots * price:.2f} estimated for {n_shots} "
                   f"images (build-time estimate, billed to your Google API key).")


def _dress(shots: list) -> list:
    """Fill angle/pose shots with random unisex outfits (close-ups stay blank)."""
    from studio.wardrobe import OUTFIT_SHOT_KINDS, random_outfits

    targets = [s for s in shots if s.kind in OUTFIT_SHOT_KINDS]
    outfits = random_outfits(len(targets))
    dressed = dict(zip((s.id for s in targets), outfits))
    return [s.model_copy(update={"outfit": dressed[s.id]}) if s.id in dressed else s
            for s in shots]


@app.command()
def preprocess(
    inputs: list[Path] = typer.Argument(..., exists=True, help="Image files and/or folders"),
    out: Path = typer.Option(None, help="Output folder (default: new runs/ subfolder)"),
    target: int = typer.Option(settings.target_long_side, help="Long-side resolution"),
    restore: bool = typer.Option(None, "--restore/--no-restore",
                                 help="Force restoration on/off (default: auto)"),
    restore_backend: str = typer.Option(settings.restore_backend, help="auto | comfyui | basic"),
    isolate: bool = typer.Option(True, "--isolate/--no-isolate",
                                 help="Cut out subject, drop background/props"),
    isolation_backend: str = typer.Option(settings.isolation_backend, help="builtin | comfyui"),
    subject_prompt: str = typer.Option("character", help="SAM3 prompt for what to keep"),
    exclude_prompt: str = typer.Option("", help="SAM3 prompt for held props to remove"),
):
    """Restore/upscale/isolate images (standalone)."""
    out = out or pipeline.new_run_dir("prepped")
    pipeline.preprocess_sources(
        _expand(inputs), out, target=target, force_restore=restore, isolate=isolate,
        subject_prompt=subject_prompt, exclude_prompt=exclude_prompt,
        restore_backend=restore_backend, isolation_backend=isolation_backend,
        progress=typer.echo)
    typer.echo(f"Done: {out}")


@app.command()
def generate(
    references: list[Path] = typer.Argument(..., exists=True,
                                            help="Reference image files and/or folders"),
    out: Path = typer.Option(None, help="Output folder (default: new runs/ subfolder)"),
    name: str = typer.Option("", help="Character name used in prompts"),
    engine: str = typer.Option(settings.default_engine, help="gemini (cloud) or comfyui (local)"),
    cloud_model: str = typer.Option("", help=f"Cloud image model (default {settings.gemini_image_model})"),
    max_shots: int = typer.Option(0, help="Limit number of shots (0 = full plan)"),
    isolate_angles: bool = typer.Option(False, help="Isolate generated angle shots onto white"),
    subject_prompt: str = typer.Option("character", help="SAM3 prompt if isolating"),
    exclude_prompt: str = typer.Option("", help="Props to remove when isolating"),
    exclude_props: bool = typer.Option(
        True, "--exclude-props/--keep-props",
        help="Ask the generator to omit bags/held objects from the reference"),
    randomize_outfits: bool = typer.Option(
        False, help="Dress angle/pose shots in random unisex outfits"),
    front: bool = typer.Option(False, help="Jump ComfyUI's pending queue"),
):
    """Generate the shot set from reference image(s) (standalone)."""
    out = out or pipeline.new_run_dir("generated")
    shots = default_plan(subject=f"character {name}" if name else "the character")
    if max_shots:
        shots = shots[:max_shots]
    if randomize_outfits:
        shots = _dress(shots)
    _echo_cloud_estimate(engine, cloud_model, len(shots))
    results = pipeline.generate_shots(
        _expand(references), shots, engine, out, cloud_model=cloud_model,
        isolate_angles=isolate_angles, subject_prompt=subject_prompt,
        exclude_prompt=exclude_prompt, exclude_props=exclude_props, front=front,
        progress=typer.echo)
    if not any(r.path for r in results):
        raise typer.Exit(1)
    typer.echo(f"Done: {out}")


@app.command()
def caption(
    folder: Path = typer.Argument(..., exists=True, file_okay=False,
                                  help="Folder of images to tag"),
    captioner: str = typer.Option(settings.default_captioner,
                                  help=f"one of {list(CAPTIONERS_BY_KEY)}"),
    name: str = typer.Option("", help="Character name used in captions"),
    trigger: str = typer.Option("", help="Trigger word placed first in every caption"),
    model: str = typer.Option("", help="Model id (gemini captioner; blank = default)"),
):
    """Write .txt caption sidecars for every image in a folder (standalone).

    The `custom` captioner reuses the endpoint saved by the UI's Caption tab, so
    the CLI and UI behave identically.
    """
    from studio.captioner import (
        CaptionerConfigError,
        caption_folder,
        resolve_captioner_config,
    )

    try:
        model_override, spec_overrides = resolve_captioner_config(captioner, model)
    except CaptionerConfigError as e:
        typer.echo(str(e))
        raise typer.Exit(1)
    caption_folder(folder, captioner, name, trigger, progress=typer.echo,
                   model_override=model_override, spec_overrides=spec_overrides)


@app.command()
def export(
    folders: list[Path] = typer.Argument(..., exists=True, file_okay=False,
                                         help="Folder(s) of captioned images"),
    name: str = typer.Option("", help="Character name"),
    trigger: str = typer.Option("", help="Trigger word"),
    output_root: Path = typer.Option(settings.output_root),
):
    """Package captioned folders into a flat NN.png/NN.txt dataset (standalone)."""
    from studio.package import package_dataset

    items, missing = [], []
    for folder in folders:
        for img in list_images(folder):
            txt = img.with_suffix(".txt")
            (items.append((img, txt.read_text(encoding="utf-8").strip()))
             if txt.exists() else missing.append(img.name))
    if not items:
        typer.echo("No captioned images found — run `caption` first.")
        raise typer.Exit(1)
    if missing:
        typer.echo(f"Skipping {len(missing)} uncaptioned image(s): {', '.join(missing)}")
    metadata = {"character_name": name, "trigger": trigger,
                "source_folders": [str(f) for f in folders],
                "skipped_uncaptioned": missing}
    ds = package_dataset(items, output_root, name, trigger, metadata)
    typer.echo(f"Dataset written to {ds}")


@app.command()
def build(
    images: list[Path] = typer.Argument(..., exists=True, readable=True),
    name: str = typer.Option("", help="Character name used in captions"),
    trigger: str = typer.Option("", help="LoRA trigger word placed first in every caption"),
    engine: str = typer.Option(settings.default_engine, help="gemini (cloud) or comfyui (local)"),
    captioner: str = typer.Option(settings.default_captioner,
                                  help=f"one of {list(CAPTIONERS_BY_KEY)}"),
    target: int = typer.Option(settings.target_long_side, help="Long-side resolution"),
    output_root: Path = typer.Option(settings.output_root),
    restore: bool = typer.Option(None, "--restore/--no-restore",
                                 help="Force restoration on/off (default: auto)"),
    max_shots: int = typer.Option(0, help="Limit number of shots (0 = full plan)"),
    isolate: bool = typer.Option(True, "--isolate/--no-isolate",
                                 help="Cut out subject, drop background/props"),
    subject_prompt: str = typer.Option("character", help="SAM3 prompt for what to keep"),
    exclude_prompt: str = typer.Option("", help="SAM3 prompt for held props to remove"),
    cloud_model: str = typer.Option("", help=f"Cloud image model (default {settings.gemini_image_model})"),
    exclude_props: bool = typer.Option(
        True, "--exclude-props/--keep-props",
        help="Ask the generator to omit bags/held objects from the reference"),
    randomize_outfits: bool = typer.Option(
        False, help="Dress angle/pose shots in random unisex outfits"),
):
    """Full pipeline: preprocess -> generate -> caption -> export."""
    from studio.captioner import caption_images
    from studio.package import package_dataset

    run_dir = pipeline.new_run_dir(name or trigger)
    typer.echo(f"Run dir: {run_dir}")

    reports = pipeline.preprocess_sources(
        _expand(images), run_dir / "prepped", target=target, force_restore=restore,
        isolate=isolate, subject_prompt=subject_prompt, exclude_prompt=exclude_prompt,
        progress=typer.echo)

    shots = default_plan(subject=f"character {name}" if name else "the character")
    if max_shots:
        shots = shots[:max_shots]
    if randomize_outfits:
        shots = _dress(shots)
    _echo_cloud_estimate(engine, cloud_model, len(shots))

    results = pipeline.generate_shots(
        [r.output for r in reports], shots, engine, run_dir / "generated",
        cloud_model=cloud_model, isolate_angles=isolate, subject_prompt=subject_prompt,
        exclude_prompt=exclude_prompt, exclude_props=exclude_props, progress=typer.echo)
    kept = [r.path for r in results if r.path]
    if not kept:
        typer.echo("No shots succeeded; aborting before captioning.")
        raise typer.Exit(1)

    all_images = [r.output for r in reports] + kept
    items = caption_images(all_images, captioner, name, trigger, progress=typer.echo)
    metadata = {
        "character_name": name,
        "trigger": trigger,
        "engine": engine,
        "captioner": captioner,
        "sources": [str(s) for s in images],
        "shots": [{"id": r.shot.id, "seed": r.seed, "error": r.error} for r in results],
    }
    ds = package_dataset(items, output_root, name, trigger, metadata)
    typer.echo(f"\nDone: {ds}")


if __name__ == "__main__":
    app()
