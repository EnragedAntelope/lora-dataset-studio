# Architecture

Version: 0.3.0

```
app.py                  Gradio UI — thin wiring over the stage functions (5 tabs)
cli.py                  Typer CLI — one subcommand per stage + `build` for all four
studio/
  config.py             Settings (env/.env overridable), captioner registry with
                        per-model prompt templates, engine/price tables
  pipeline.py           Stage orchestration: preprocess_sources(), generate_shots()
  preprocess.py         Restore (comfyui|basic|auto) + isolate + resize
  isolate.py            Subject isolation: builtin SAM3 (transformers) | comfyui
  captioner.py          Captioner backends (transformers | gemini | openai-compat),
                        caption finalization, standalone caption_folder()
  quality.py            Advisory sharpness check (variance of Laplacian, numpy-only)
  package.py            Dataset export (NN.png/NN.txt + metadata.json + README.txt)
  shotplan.py           Default shot plan (curated 24 shots: angles, poses, emotions,
                        settings + outfit merged into each shot) + Shot model +
                        apply_wardrobe() outfit injection
  plan_io.py            Save/load shot plans as YAML (user-editable prompt libraries)
  trainer_configs.py    Emit LoRA-trainer configs (ai-toolkit config.yaml / musubi
                        dataset.toml) + run-command builders + model registry
  user_config.py        Persist trainer install paths + last training settings to
                        .cache/user_settings.json (paths only; gitignored)
  comfy_api.py          Thin ComfyUI HTTP client (upload, queue, poll, fetch, free)
  engines/
    base.py             Engine protocol + GenerationError
    gemini.py           Cloud engine (Gemini image models via google-genai) with
                        a cached, force-refreshable model list
    comfyui.py          Local engine (Qwen Image Edit 2511 + Multiple-Angles LoRA)
  comfy_workflows/*.json  API-format ComfyUI graphs (restore, isolate ×2, qwen edit)
```

## Design rules

1. **Every stage is standalone.** Stage functions take explicit input paths and an
   explicit output folder; none of them require state from another stage. The UI's
   "auto-fill the next tab's folder" is convenience wiring in `app.py` only.
2. **Local/cloud is a per-stage choice**, selected at call time (engine key, captioner
   key, isolation/restore backend), never a global mode.
3. **Heavy imports are lazy.** `torch`/`transformers`/`google-genai` are imported inside
   the functions that need them, so cloud-only or export-only use never loads them.
4. **Captions are `.txt` sidecars** next to the images. That makes stage ③ composable
   with anything (hand-written captions, other tools) and lets export (④) work on any
   folder that has pairs.

## Data flow (full pipeline)

```
sources ─① preprocess→ runs/<stamp>-prepped/*_prepped.png
        └─ restore (comfyui models | basic lanczos)  → isolate (SAM3) → resize
prepped ─② generate→ runs/<stamp>-generated/<shot-id>.png   (one per plan row)
any folder ─③ caption→ *.txt sidecars (trigger-first, per-model prompt template)
folders(s) ─④ export→ datasets/<name>-dataset/NN.png + NN.txt + metadata.json
dataset ──⑤ train  → writes ai-toolkit config.yaml OR musubi dataset.toml into the
                     dataset folder + prints the run command (optional; nothing launched)
```

## Gotchas (hard-won)

- **`facebook/sam3` is gated.** Users must accept Meta's license on the model page and
  authenticate (`HF_TOKEN` in `.env` or `hf auth login`) or `isolate.py` raises a clear
  error with instructions.
- **Rear views hallucinate** when generated straight from a front reference. The shot
  plan chains them (`chain_from`) off a generated side view; chained shots always run
  last, and the chained image is passed *first* so single-reference engines rotate
  stepwise.
- **SAM3 merges held props into the subject segment.** An "exclude" prompt runs a second
  segmentation, dilates it slightly (edge halos), and subtracts it. Inverting the subject
  mask does *not* work for this.
- **The Multiple-Angles LoRA is trained on clean splat renders** — isolating the subject
  onto white dramatically improves its output, especially direct back views. The LoRA is
  trigger-based (`<sks>` grammar), so its strength is 0.9 for `kind="angle"` shots and
  zeroed for pose/emotion shots.
- **VRAM choreography:** before loading a ~17 GB local captioner, the pipeline asks
  ComfyUI to `/free` its models (best effort) and unloads the in-process SAM3.
- **ComfyUI queue guard:** if ComfyUI already has >10 pending jobs, the client fails
  fast instead of silently queueing behind them.
- **Gemini refusals return no image part** — that's detected and reported as a refusal;
  retries don't help, so only transient errors are retried.
- **Shot plan is a curated list, not a cartesian product.** Each of the 24 default
  shots combines a unique angle/pose/emotion/setting. Scene/lighting is folded into
  other shot kinds so the dataset is not skewed by many images of the same standing
  pose with different backgrounds.
- **Gemini model list is cached locally.** `studio/engines/gemini.py` persists the
  live model list to `.cache/gemini_image_models.json` with a 24-hour TTL. The UI
  loads from cache; a force-refresh button bypasses the TTL.
- **Groq rate limits are honored per model.** `groq-llama4-scout` and `groq-qwen3.6`
  use different `min_interval_s` values reflecting their free-tier TPM limits; 429
  responses are retried with exponential backoff.
- **ComfyUI caches model combo lists**; a freshly downloaded model file may need a
  ComfyUI restart before the bundled workflows validate.
- **Model filenames are configuration.** The bundled workflow JSONs are patched at load
  time from settings (`LDS_QWEN_EDIT_MODEL` etc.), so users don't edit JSON to match
  their filenames.
- **ai-toolkit configs are one-command; musubi configs are not.** `trainer_configs.py`
  emits a fully runnable ai-toolkit `config.yaml` (model is a HF id, all hyperparameters
  inline → `python run.py config.yaml`). musubi's `dataset.toml` is complete, but its
  *training* invocation needs the user's local DiT/VAE/text-encoder paths, so the musubi
  run command is a template with `<<FILL: ...>>` placeholders. The UI says so; it is not
  faked as one-click.
- **Trainer model registry is curated, not exhaustive.** Where a model's canonical HF id
  or musubi training script isn't something we can guarantee, the preset carries a
  `<<FILL>>` placeholder so the emitted config is honest rather than silently wrong.
- **Sharpness is advisory.** `quality.py` labels blurry shots (`⚠ blurry (score)`) in the
  curate gallery and lists them at export; it never deletes or blocks. Threshold is
  tunable via `LDS_SHARPNESS_BLUR_THRESHOLD`.
- **The outfit column is applied at generation time.** `apply_wardrobe()` folds a shot's
  outfit into both prompts just before the engine runs (idempotent), so the column stays
  functional even if prompt cells were hand-edited. Default shots leave outfit empty to
  avoid identity drift.
- **`user_settings.json` holds paths only.** Trainer install paths + last hyperparameters
  live in `.cache/user_settings.json` (gitignored). API keys never go there — they stay
  in `.env`/environment.

## Roadmap / deferred

Relocated here from `shotplan.py`. Done this revision (0.3.0):

- ✅ musubi-tuner `dataset.toml` + ostris ai-toolkit `config.yaml` generators (⑤ Train tab).
- ✅ Outfit/wardrobe control (per-shot `outfit`, folded into prompts without identity drift).
- ✅ Sharpness quality flag before export (advisory Laplacian-variance).
- ✅ Bulk/inline caption editor in the UI (edit any `.txt` sidecar by hand).
- ✅ User-editable prompt libraries — save/load shot plans as YAML.

Deferred (with rationale):

- **In-GUI training launch** (subprocess + streamed logs): configs + saved paths + the run
  command cover the workflow; launching is fragile across trainer venvs/CUDA setups, so it
  is intentionally out of scope. The command is displayed, never executed.
- **Automated aesthetic scoring** beyond sharpness: needs a heavy scoring model; the cheap
  Laplacian check covers the common "is it blurry" case without a new dependency.
- **Regularization-image generation**: niche for single-character LoRAs on these trainers.
- **Face-similarity guard across multiple sources**: would add a face-recognition
  dependency for a narrow multi-reference case.
- **`platformdirs` cache relocation**: the repo-local `.cache/` is consistent with how the
  app resolves `.env`/models and is self-contained; not worth the churn yet.

## Security posture

- UI binds to `127.0.0.1`; no auth layer, so it must not be exposed (`share=True` is
  deliberately not used).
- API keys live in `.env` (gitignored; `setup.sh` chmods it 600) or the environment,
  and are only sent to their own vendor endpoints.
- No telemetry; the only network calls are the ones the selected backends require.
- Local model weights come from Hugging Face as safetensors.
- The ⑤ Train tab **generates and displays** config files and a run command — it never
  executes a subprocess or shell, so there is no command-injection surface. Generated
  configs reference the user's own model ids/paths; no credentials are embedded.
- `.cache/user_settings.json` (gitignored) stores filesystem paths + hyperparameters only.
