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
from studio.config import CAPTIONERS_BY_KEY, CLOUD_IMAGE_PRICES, list_images, settings
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
):
    """Generate the shot set from reference image(s) (standalone)."""
    out = out or pipeline.new_run_dir("generated")
    shots = default_plan(subject=f"character {name}" if name else "the character")
    if max_shots:
        shots = shots[:max_shots]
    if engine == "gemini":
        price = CLOUD_IMAGE_PRICES.get(cloud_model or settings.gemini_image_model)
        if price:
            typer.echo(f"Cloud engine: ~${len(shots) * price:.2f} estimated for "
                       f"{len(shots)} images (billed to your Google API key).")
    results = pipeline.generate_shots(
        _expand(references), shots, engine, out, cloud_model=cloud_model,
        isolate_angles=isolate_angles, subject_prompt=subject_prompt, progress=typer.echo)
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
):
    """Write .txt caption sidecars for every image in a folder (standalone)."""
    from studio.captioner import caption_folder

    caption_folder(folder, captioner, name, trigger, progress=typer.echo)


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
    if engine == "gemini":
        price = CLOUD_IMAGE_PRICES.get(cloud_model or settings.gemini_image_model)
        if price:
            typer.echo(f"Cloud engine: ~${len(shots) * price:.2f} estimated for "
                       f"{len(shots)} images (billed to your Google API key).")

    results = pipeline.generate_shots(
        [r.output for r in reports], shots, engine, run_dir / "generated",
        cloud_model=cloud_model, isolate_angles=isolate, subject_prompt=subject_prompt,
        exclude_prompt=exclude_prompt, progress=typer.echo)
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
