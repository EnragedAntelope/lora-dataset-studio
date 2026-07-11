"""LoRA Dataset Studio — Gradio UI.

Every tab is standalone: point it at any folder (or upload files) and run just
that stage. When you do run stages in order, each one auto-fills the next
tab's input folder — chaining is a convenience, never a requirement.

Run:  python app.py   then open http://127.0.0.1:7861
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import gradio as gr
import pandas as pd

from studio import pipeline
from studio.captioner import SUBJECT_ALIASES, Captioner, caption_images, finalize_caption
from studio.config import CAPTIONERS, CLOUD_IMAGE_PRICES, list_images, settings
from studio.shotplan import Shot, default_plan

ENGINE_CHOICES = [
    ("Cloud — Gemini image model (best identity fidelity, SFW only)", "gemini"),
    ("Local — ComfyUI Qwen Image Edit 2511 (free, private, uncensored)", "comfyui"),
]
CLOUD_MODEL_CHOICES = [(f"{m}  (~${p:.3f}/img est.)", m) for m, p in CLOUD_IMAGE_PRICES.items()]
CAPTIONER_CHOICES = [(c.label, c.key) for c in CAPTIONERS]
ISOLATION_CHOICES = [
    ("Built-in SAM3 (no ComfyUI needed; gated HF model)", "builtin"),
    ("ComfyUI SAM3 workflow", "comfyui"),
]
RESTORE_BACKEND_CHOICES = [
    ("Auto (ComfyUI models if reachable, else basic)", "auto"),
    ("ComfyUI (DeJPG + photo upscale models)", "comfyui"),
    ("Basic (Lanczos only, no ComfyUI)", "basic"),
]


# ---------- helpers ----------

def _stamped(kind: str) -> Path:
    d = settings.runs_dir / f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-{kind}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _inputs(files: list[str] | None, folder: str) -> list[Path]:
    """Uploaded files win; otherwise list the folder."""
    if files:
        return [Path(f) for f in files]
    if folder.strip():
        images = list_images(Path(folder.strip()))
        if images:
            return images
        raise gr.Error(f"No images found in folder: {folder}")
    raise gr.Error("Upload image(s) or enter an input folder first.")


def _plan_df(subject: str) -> pd.DataFrame:
    shots = default_plan(subject=subject or "the character")
    return pd.DataFrame([s.model_dump() for s in shots])


def _df_to_shots(df: pd.DataFrame) -> list[Shot]:
    def val(row, k):
        v = row[k] if k in row else ""
        return "" if pd.isna(v) else str(v)

    cols = ("id", "kind", "local_prompt", "cloud_prompt", "chain_from")
    return [Shot(**{k: val(row, k) for k in cols})
            for _, row in df.iterrows() if val(row, "id").strip()]


def _gen_gallery(results: list[pipeline.GenResult]):
    ok = [r for r in results if r.path and r.path.exists()]
    gallery = [(str(r.path), r.shot.id) for r in ok]
    ids = [r.shot.id for r in ok]
    return gallery, gr.CheckboxGroup(choices=ids, value=ids)


# ---------- ① preprocess ----------

def do_preprocess(files: list[str], folder: str, target: int, restore_mode: str,
                  restore_backend: str, isolate: bool, isolation_backend: str,
                  subject_prompt: str, exclude_prompt: str):
    sources = _inputs(files, folder)
    out_dir = _stamped("prepped")
    force = {"Auto (only if needed)": None, "Always": True, "Never": False}[restore_mode]
    log: list[str] = []
    try:
        reports = pipeline.preprocess_sources(
            sources, out_dir, target=target, force_restore=force, isolate=isolate,
            subject_prompt=subject_prompt or "character",
            exclude_prompt=exclude_prompt or "", restore_backend=restore_backend,
            isolation_backend=isolation_backend, progress=log.append)
    except Exception as e:
        raise gr.Error(f"Preprocess failed: {e}")
    gallery = [(str(r.output), f"{r.source.name}: {r.reason}") for r in reports]
    note = f"✅ {len(reports)} image(s) preprocessed into {out_dir}"
    # Auto-fill downstream tabs (they can still be pointed anywhere else)
    return gallery, note, "\n".join(log), str(out_dir), str(out_dir)


# ---------- ② generate & curate ----------

def do_generate(files: list[str], folder: str, plan_df: pd.DataFrame, engine: str,
                cloud_model: str, isolate_angles: bool, isolation_backend: str,
                subject_prompt: str, gen_dir_prev: str, results_state,
                progress=gr.Progress()):
    sources = _inputs(files, folder)
    out_dir = Path(gen_dir_prev) if gen_dir_prev.strip() else _stamped("generated")
    shots = _df_to_shots(plan_df)
    log: list[str] = []

    def report(msg: str):
        log.append(msg)
        progress((len(log), len(shots) + 2), desc=msg)

    results = pipeline.generate_shots(
        sources, shots, engine, out_dir, cloud_model=cloud_model,
        isolate_angles=isolate_angles, subject_prompt=subject_prompt or "character",
        isolation_backend=isolation_backend, progress=report)
    gallery, keep = _gen_gallery(results)
    return results, gallery, keep, "\n".join(log), str(out_dir), str(out_dir)


def do_regenerate(files: list[str], folder: str, plan_df: pd.DataFrame, engine: str,
                  cloud_model: str, isolate_angles: bool, isolation_backend: str,
                  subject_prompt: str, gen_dir: str, results_state, keep_ids: list[str],
                  progress=gr.Progress()):
    if not results_state:
        raise gr.Error("Nothing generated yet.")
    all_ids = {r.shot.id for r in results_state}
    redo = all_ids - set(keep_ids or [])
    if not redo:
        raise gr.Error("Uncheck the shots you want regenerated, then click again.")
    sources = _inputs(files, folder)
    log: list[str] = []
    results = pipeline.generate_shots(
        sources, _df_to_shots(plan_df), engine, Path(gen_dir), cloud_model=cloud_model,
        isolate_angles=isolate_angles, subject_prompt=subject_prompt or "character",
        isolation_backend=isolation_backend, existing=results_state, only_ids=redo,
        progress=log.append)
    gallery, keep = _gen_gallery(results)
    return results, gallery, keep, "\n".join(log)


def do_refresh_disk(results_state, gen_dir: str):
    """Re-sync with the output folder — files you deleted externally drop out."""
    if not results_state:
        raise gr.Error("No generation results in this session.")
    before = len(results_state)
    results = [r for r in results_state if r.path is None or r.path.exists()]
    gallery, keep = _gen_gallery(results)
    note = f"Re-synced with {gen_dir}: {before - len(results)} externally deleted shot(s) dropped."
    return results, gallery, keep, note


def send_kept_to_caption(results_state, keep_ids: list[str], gen_dir: str):
    if not gen_dir.strip():
        raise gr.Error("Nothing generated yet.")
    kept = {r.path.name for r in (results_state or [])
            if r.path and r.path.exists() and r.shot.id in set(keep_ids or [])}
    if not kept:
        raise gr.Error("No kept shots selected.")
    images = list_images(Path(gen_dir.strip()))
    names = [p.name for p in images]
    gallery = [(str(p), p.name) for p in images]
    preselected = [n for n in names if n in kept]
    note = (f"{len(names)} image(s) loaded from ② — {len(preselected)} kept shot(s) "
            f"preselected for captioning.")
    return gen_dir, gallery, gr.CheckboxGroup(choices=names, value=preselected), note


# ---------- ③ caption ----------

def load_caption_folder(folder: str):
    images = list_images(Path(folder.strip())) if folder.strip() else []
    if not images:
        raise gr.Error(f"No images found in folder: {folder or '(empty)'}")
    captioned = {p.name for p in images if p.with_suffix(".txt").exists()}
    gallery = [(str(p), f"{p.name}{' ✓ captioned' if p.name in captioned else ''}")
               for p in images]
    names = [p.name for p in images]
    note = (f"{len(images)} image(s) loaded, {len(captioned)} already have .txt "
            f"sidecars (re-captioning overwrites them).")
    return gallery, gr.CheckboxGroup(choices=names, value=names), note


def do_test_caption(folder: str, selected: list[str], captioner_key: str,
                    name: str, trigger: str):
    if not folder.strip() or not selected:
        raise gr.Error("Load a folder and select at least one image first.")
    path = Path(folder.strip()) / selected[0]
    cap = Captioner(captioner_key)
    try:
        raw = cap.caption(path, subject=name or "the character")
    except Exception as e:
        raise gr.Error(str(e))
    finally:
        cap.unload()
    return finalize_caption(raw, trigger, name, SUBJECT_ALIASES)


def do_caption(folder: str, selected: list[str], captioner_key: str,
               name: str, trigger: str, progress=gr.Progress()):
    if not folder.strip() or not selected:
        raise gr.Error("Load a folder and select the images to caption first.")
    base = Path(folder.strip())
    images = [base / s for s in selected]
    log: list[str] = []

    def report(msg: str):
        log.append(msg)
        progress((len(log), len(images) + 2), desc=msg)

    try:
        items = caption_images(images, captioner_key, name, trigger, progress=report)
    except Exception as e:
        raise gr.Error(f"Captioning failed: {e}")
    for img, caption in items:
        img.with_suffix(".txt").write_text(caption, encoding="utf-8")
    gallery, boxes, note = load_caption_folder(folder)
    result = f"✅ Wrote {len(items)} caption sidecar(s) in {base}"
    return gallery, boxes, result, "\n".join(log), str(base)


# ---------- ④ export ----------

def do_export(folders_text: str, name: str, trigger: str, output_root: str):
    folders = [Path(line.strip()) for line in folders_text.splitlines() if line.strip()]
    if not folders:
        raise gr.Error("Enter at least one folder of captioned images (one per line).")
    from studio.package import package_dataset

    items: list[tuple[Path, str]] = []
    missing: list[str] = []
    for folder in folders:
        for img in list_images(folder):
            txt = img.with_suffix(".txt")
            if txt.exists():
                items.append((img, txt.read_text(encoding="utf-8").strip()))
            else:
                missing.append(f"{folder.name}/{img.name}")
    if not items:
        raise gr.Error("No captioned images found — run ③ Caption first "
                       "(each exported image needs a .txt sidecar).")
    metadata = {"character_name": name, "trigger": trigger,
                "source_folders": [str(f) for f in folders],
                "skipped_uncaptioned": missing}
    ds = package_dataset(items, Path(output_root), name, trigger, metadata)
    sample = next(ds.glob("*.txt")).read_text(encoding="utf-8")
    skipped = f"\n⚠️ Skipped (no caption): {', '.join(missing)}" if missing else ""
    return f"✅ Dataset ready: {ds}  ({len(items)} image/caption pairs){skipped}\n\nSample caption:\n{sample}"


# ---------- misc ----------

def refresh_plan(name: str) -> pd.DataFrame:
    return _plan_df(f"character {name}" if name else "the character")


def estimate_cost(engine: str, cloud_model: str, df: pd.DataFrame) -> str:
    n = len(df)
    if engine == "gemini":
        price = CLOUD_IMAGE_PRICES.get(cloud_model)
        if price is None:
            return f"**Cost:** {n} images on `{cloud_model}` (price unknown — billed to your API key)"
        return (f"**Cost:** ~${n * price:.2f} for {n} images on `{cloud_model}` "
                f"(estimate at build time — billed to your own Google API key)")
    return f"**Cost:** {n} images, $0 (local generation)"


def refresh_cloud_models():
    from studio.engines.gemini import list_image_models

    try:
        models = list_image_models()
    except Exception as e:
        raise gr.Error(f"Could not list models: {e}")
    return gr.Dropdown(choices=[(f"{m}  ({p})", m) for m, p in models],
                       value=models[0][0] if models else settings.gemini_image_model)


# ---------- layout ----------

with gr.Blocks(title="LoRA Dataset Studio") as demo:
    gr.Markdown(
        "# LoRA Dataset Studio\n"
        "One image → ready-to-train character LoRA dataset. Every tab works standalone "
        "on any folder — or run them in order and each step auto-fills the next: "
        "**① Preprocess → ② Generate & curate → ③ Caption → ④ Export**."
    )
    results_state = gr.State([])

    with gr.Tabs():
        with gr.Tab("① Preprocess (optional)"):
            gr.Markdown("Restore / upscale / isolate source images. Skip this tab entirely "
                        "if your images are already clean.")
            with gr.Row():
                with gr.Column(scale=1):
                    pre_files = gr.File(label="Source image(s)", file_count="multiple",
                                        file_types=["image"])
                    pre_folder = gr.Textbox(label="…or input folder",
                                            placeholder="path/to/images (used if no upload)")
                    target = gr.Slider(512, 2048, value=settings.target_long_side, step=64,
                                       label="Dataset resolution (long side, px)")
                    restore_mode = gr.Radio(["Auto (only if needed)", "Always", "Never"],
                                            value="Auto (only if needed)", label="Restoration")
                    restore_backend = gr.Dropdown(RESTORE_BACKEND_CHOICES,
                                                  value=settings.restore_backend,
                                                  label="Restoration backend")
                    isolate = gr.Checkbox(value=True,
                                          label="Isolate subject (cutout onto white background)")
                    isolation_backend = gr.Dropdown(ISOLATION_CHOICES,
                                                    value=settings.isolation_backend,
                                                    label="Isolation backend")
                    subject_prompt = gr.Textbox(label="Subject to keep (SAM3 prompt)",
                                                value="character")
                    exclude_prompt = gr.Textbox(
                        label="Objects to remove (props the subject holds/touches)",
                        placeholder="microphone, microphone stand")
                    btn_pre = gr.Button("① Preprocess", variant="primary")
                with gr.Column(scale=2):
                    pre_note = gr.Markdown()
                    prep_gallery = gr.Gallery(label="Preprocessed output", columns=4, height=340)

        with gr.Tab("② Generate & Curate"):
            gr.Markdown("Turn reference image(s) into a full shot set. Each plan row becomes "
                        "one generated image; `chain_from` makes rear views build on a "
                        "generated side view.")
            with gr.Row():
                with gr.Column(scale=1):
                    gen_files = gr.File(label="Reference image(s)", file_count="multiple",
                                        file_types=["image"])
                    gen_src_folder = gr.Textbox(label="…or reference folder (auto-filled by ①)")
                    gen_name = gr.Textbox(label="Character name (used in prompts)",
                                          placeholder="Sy Snootles")
                    refresh = gr.Button("Rebuild default plan with character name")
                    engine = gr.Radio(ENGINE_CHOICES, value=settings.default_engine,
                                      label="Generation engine")
                    cloud_model = gr.Dropdown(CLOUD_MODEL_CHOICES,
                                              value=settings.gemini_image_model,
                                              label="Cloud image model")
                    refresh_models = gr.Button("🔄 Refresh model list from API")
                    cost = gr.Markdown()
                    gen_isolate = gr.Checkbox(value=False,
                                              label="Isolate generated angle shots (white background)")
                    gen_iso_backend = gr.Dropdown(ISOLATION_CHOICES,
                                                  value=settings.isolation_backend,
                                                  label="Isolation backend")
                    gen_subject = gr.Textbox(label="Subject prompt for isolation", value="character")
                with gr.Column(scale=2):
                    plan = gr.Dataframe(value=_plan_df("the character"), label="Shot plan",
                                        interactive=True, wrap=True)
            with gr.Row():
                btn_gen = gr.Button("② Generate all shots", variant="primary")
                btn_regen = gr.Button("♻️ Regenerate UNCHECKED shots (new seeds)")
                btn_disk = gr.Button("🔃 Re-sync with output folder")
                btn_send = gr.Button("➡ Send kept shots to ③ Caption")
            gen_out_dir = gr.Textbox(label="Output folder (blank = new run folder)", value="")
            gen_gallery = gr.Gallery(label="Generated shots", columns=6, height=420)
            keep = gr.CheckboxGroup(label="✅ Kept shots — UNCHECK to reject", choices=[])

        with gr.Tab("③ Caption"):
            gr.Markdown("Tag any folder of images with natural-language caption `.txt` "
                        "sidecars — the folder does **not** need to come from ① or ②. "
                        "Each captioner uses a prompt tuned to that model.")
            with gr.Row():
                with gr.Column(scale=1):
                    cap_folder = gr.Textbox(label="Image folder (auto-filled by ①/②)")
                    btn_load = gr.Button("📂 Load folder")
                    cap_name = gr.Textbox(label="Character name (optional)",
                                          placeholder="Sy Snootles")
                    cap_trigger = gr.Textbox(label="Trigger word (optional, placed first)",
                                             placeholder="sysnootles")
                    captioner = gr.Dropdown(CAPTIONER_CHOICES, value=settings.default_captioner,
                                            label="Captioner")
                    cap_cost = gr.Markdown()
                    btn_test = gr.Button("🧪 Test caption on first selected image")
                    btn_caption = gr.Button("③ Caption selected images", variant="primary")
                with gr.Column(scale=2):
                    cap_note = gr.Markdown()
                    cap_gallery = gr.Gallery(label="Folder contents", columns=6, height=340)
                    cap_select = gr.CheckboxGroup(label="Images to caption", choices=[])
            test_caption = gr.Textbox(label="Test caption output", lines=4)
            cap_result = gr.Markdown()

        with gr.Tab("④ Export"):
            gr.Markdown("Package captioned images into a flat `NN.png` + `NN.txt` dataset "
                        "folder (ai-toolkit / OneTrainer ready), with `metadata.json` and "
                        "`README.txt`. List one or more folders (one per line) — e.g. the "
                        "preprocessed sources **and** the generated shots.")
            exp_folders = gr.Textbox(label="Folders of captioned images (one per line)", lines=3)
            with gr.Row():
                exp_name = gr.Textbox(label="Character name", placeholder="Sy Snootles")
                exp_trigger = gr.Textbox(label="Trigger word", placeholder="sysnootles")
            output_root = gr.Textbox(label="Output folder", value=str(settings.output_root))
            btn_export = gr.Button("④ Export dataset", variant="primary")
            exp_result = gr.Textbox(label="Result", lines=8)

    log_box = gr.Textbox(label="Log", lines=8)

    # ---------- wiring ----------

    btn_pre.click(
        do_preprocess,
        [pre_files, pre_folder, target, restore_mode, restore_backend, isolate,
         isolation_backend, subject_prompt, exclude_prompt],
        [prep_gallery, pre_note, log_box, gen_src_folder, cap_folder])

    refresh.click(refresh_plan, [gen_name], [plan])
    refresh_models.click(refresh_cloud_models, [], [cloud_model])
    engine.change(estimate_cost, [engine, cloud_model, plan], [cost])
    cloud_model.change(estimate_cost, [engine, cloud_model, plan], [cost])
    plan.change(estimate_cost, [engine, cloud_model, plan], [cost])

    gen_inputs = [gen_files, gen_src_folder, plan, engine, cloud_model, gen_isolate,
                  gen_iso_backend, gen_subject]
    btn_gen.click(do_generate, gen_inputs + [gen_out_dir, results_state],
                  [results_state, gen_gallery, keep, log_box, gen_out_dir, cap_folder])
    btn_regen.click(do_regenerate, gen_inputs + [gen_out_dir, results_state, keep],
                    [results_state, gen_gallery, keep, log_box])
    btn_disk.click(do_refresh_disk, [results_state, gen_out_dir],
                   [results_state, gen_gallery, keep, log_box])
    btn_send.click(send_kept_to_caption, [results_state, keep, gen_out_dir],
                   [cap_folder, cap_gallery, cap_select, cap_note])

    btn_load.click(load_caption_folder, [cap_folder], [cap_gallery, cap_select, cap_note])
    captioner.change(
        lambda key: f"**Cost:** {next(c.cost_note for c in CAPTIONERS if c.key == key)}"
                    + (f"  |  VRAM: {next(c.vram_note for c in CAPTIONERS if c.key == key)}"
                       if next(c.vram_note for c in CAPTIONERS if c.key == key) else ""),
        [captioner], [cap_cost])
    btn_test.click(do_test_caption, [cap_folder, cap_select, captioner, cap_name, cap_trigger],
                   [test_caption])
    btn_caption.click(do_caption, [cap_folder, cap_select, captioner, cap_name, cap_trigger],
                      [cap_gallery, cap_select, cap_result, log_box, exp_folders])

    btn_export.click(do_export, [exp_folders, exp_name, exp_trigger, output_root], [exp_result])

if __name__ == "__main__":
    # Bound to localhost on purpose: no auth layer, and .env keys are reachable
    # through the process. Do not expose publicly / use share=True.
    demo.launch(server_name="127.0.0.1", server_port=7861, inbrowser=True)
