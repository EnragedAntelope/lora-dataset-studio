# Architecture

Version: 0.5.0

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
                        caption finalization, standalone caption_folder().
                        Captioner takes model_override (Gemini model picker) and
                        spec_overrides (runtime config for the custom endpoint)
  quality.py            Advisory sharpness check (variance of Laplacian, numpy-only)
  package.py            Dataset export (NN.png/NN.txt + metadata.json + README.txt);
                        resolve_export_items() classifies candidates by caption sidecar
                        state (ready/empty/missing) — shared by the UI gate and the CLI
  shotplan.py           Default shot plan (curated 24 shots: angles, poses, emotions,
                        settings + outfit merged into each shot) + Shot model +
                        apply_wardrobe() outfit injection + apply_prop_exclusion()
  wardrobe.py           Compositional unisex outfit pool for the outfit column
                        (colour x garment; one pool, no gender picker)
  plan_io.py            Save/load shot plans as YAML (user-editable prompt libraries)
  dataset_stats.py      Inspect a dataset folder (count/sizes/aspects) -> suggested
                        steps + bucket ladder, so ⑤'s numbers are derived not guessed
  trainer_configs.py    Emit LoRA-trainer configs (ai-toolkit config.yaml / musubi
                        dataset.toml) + run-command builders + model registry
  user_config.py        Persist trainer install paths, last training settings, and the
                        custom captioner endpoint (URL/model/key-env-NAME/spacing) to
                        .cache/user_settings.json (no secrets; gitignored)
  update_check.py        Best-effort GitHub-release check (cached 24h) -> dismissible
                        UI banner when a newer version is published
  comfy_api.py          Thin ComfyUI HTTP client (upload, queue, poll, fetch, free)
  engines/
    base.py             Engine protocol + GenerationError
    gemini.py           Cloud engine (Gemini image models via google-genai) with
                        cached, force-refreshable image- AND caption-model lists
                        (list_image_models / list_caption_models)
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
- **SAM3 usually does NOT merge props into the subject segment** — and the exclusion
  step must not assume it does. Measured on a reference image (character with a backpack
  and a belt radio): only **3% of prop pixels** fell inside the `character` mask. Because
  `isolate_builtin` composites with `np.where(subject, image, white)`, everything outside
  the subject mask is *already* white, so a prop SAM3 excluded is gone whether or not it
  is subtracted. Subtracting it can therefore only remove genuine subject pixels.
  `isolate_builtin` measures the subject/prop overlap and, below
  `_MERGED_PROP_OVERLAP`, reports the exclusion as a no-op instead of carving. The
  feature still earns its place for genuinely merged props (a microphone gripped in a
  hand), which is the case it was built for.
  *(Prior versions of this file asserted the opposite; that belief produced the bug.)*
- **Never dilate the exclusion mask by default.** Growing it cannot remove out-of-subject
  halos (they are already white) — it can only eat the subject. At the old
  `max(2, long_side // 200)` (7px on a 1480px image) it destroyed **2.75% of the
  character**, versus 0.39% for a raw subtract and 0% for no exclusion at all.
  `LDS_EXCLUDE_DILATE_PX` defaults to 0 and exists only for the merged-prop case.
- **SAM3 scores a text prompt as ONE concept.** `"backpack, walkie talkie"` asks for a
  single object that is both, and found *less* than `"backpack"` alone (40,942 px vs
  42,712 px); segmenting each term and unioning found 60,084 px (1.47x). `split_terms()`
  splits on commas and `_segment_terms()` unions per-term masks. The ComfyUI graph has one
  text encoder, so it joins terms with `" or "` as the closest equivalent.
- **Occlusion holes cannot be masked away.** Where a prop physically covers the body, the
  subject mask has a real hole and the composite renders it white. Only inpainting fixes
  that; no amount of threshold/dilation tuning will.
- **The Multiple-Angles LoRA is trained on clean splat renders** — isolating the subject
  onto white dramatically improves its output, especially direct back views. The LoRA is
  trigger-based (`<sks>` grammar), so its strength is 0.9 for `kind="angle"` shots and
  zeroed for pose/emotion shots.
- **VRAM choreography:** before loading a ~17 GB local captioner, the pipeline asks
  ComfyUI to `/free` its models (best effort) and unloads the in-process SAM3.
- **ComfyUI queue guard:** if ComfyUI already has >10 pending jobs, the client fails
  fast instead of silently queueing behind them — unless `front=True`, which asks
  ComfyUI to put the job at the head of the pending queue (`/prompt` accepts `front`
  and negates the job's priority number). `front` does **not** interrupt the job already
  running, so the UI must not promise "immediate". Polling is always by our own
  `prompt_id`, so another client's jobs are never mistaken for ours.
- **The bundled workflows use ONLY core nodes.** `isolate_*.json` previously needed
  `MaskPreview+` from the third-party `ComfyUI_essentials` pack — used purely as a hack to
  render a white backdrop — which silently broke the workflows for anyone who didn't
  happen to have that pack. Replaced with core `EmptyImage` + `GetImageSize`.
  `EmptyImage` **must** be sized from the source via `GetImageSize`: hardcoding 1024²
  breaks `ImageCompositeMasked` (`resize_source: false` requires matching dimensions).
  `SAM3_Detect` *is* core (`comfy_extras/nodes_sam3.py`). Keep it this way — a
  third-party node in a bundled workflow is a silent portability failure.
- **Gemini refusals return no image part** — that's detected and reported as a refusal;
  retries don't help, so only transient errors are retried.
- **Shot plan is a curated list, not a cartesian product.** Each of the 24 default
  shots combines a unique angle/pose/emotion/setting. Scene/lighting is folded into
  other shot kinds so the dataset is not skewed by many images of the same standing
  pose with different backgrounds.
- **Gemini model lists are cached locally.** `studio/engines/gemini.py` persists the
  live image-model list to `.cache/gemini_image_models.json` and the caption-model list
  to `.cache/gemini_caption_models.json`, both with a 24-hour TTL. The UI seeds each
  dropdown from cache (never a network call on startup); a refresh button force-pulls.
  Both fall back to a static list if the API is unreachable or no key is set.
- **Update check is best-effort by design.** `update_check.py` hits the GitHub releases
  API on `demo.load` (not at process start, so it can't slow down launch), caches the
  result 24h in `.cache/update_check.json`, and falls back to a stale cache on a failed
  refetch. Any exception — offline, rate-limited, no releases published yet — is caught
  and simply shows no banner; it must never raise into the UI. `LDS_UPDATE_CHECK_ENABLED=
  false` skips the network call entirely.
- **The update check reads GitHub Releases, not commits or tags.** It compares
  `studio.__version__` against `GET /repos/.../releases/latest`, which 404s (silently, no
  banner) until a Release object exists — a `git tag` alone is not enough. So **every
  version bump needs both**: (1) update `__version__` in `studio/__init__.py` to match
  the `Version:` line at the top of this file, and (2) publish a Release —
  `gh release create vX.Y.Z --title "vX.Y.Z" --generate-notes` — targeting the commit
  that lands the bump. Skipping step 2 means users on the old version never see a notice.
- **Gemini caption default is a rolling alias.** The Gemini captioner defaults to
  `gemini-flash-latest` so it doesn't 404 when a pinned version is decommissioned (as
  `gemini-2.5-flash` was for new keys). The Caption tab can refresh and pick a specific
  model; `Captioner(model_override=...)` applies it only when the backend is Gemini.
- **Groq Qwen3.6 is a reasoning model.** Its spec sets `extra_params={"reasoning_effort":
  "none"}` to switch off the `<think>` scratchpad (otherwise the token budget is spent
  on reasoning and the caption comes back empty/truncated), plus a higher `max_tokens`
  as insurance and `_clean()` strips any residual `<think>` tags. 429s are retried with
  backoff; `min_interval_s` spaces requests for the free tier.
- **Custom OpenAI-compatible captioner.** The `custom` captioner carries no endpoint in
  config; the Caption tab collects base URL / model / key-env-NAME / request spacing and
  persists them (minus the secret) via `user_config.set_custom_captioner`. At caption
  time `_resolve_captioner_config` loads them into `spec_overrides`, applied with
  `CaptionerSpec.model_copy(update=...)` so the registry spec is never mutated. The same
  429-backoff + spacing path as Groq applies. The API key is read from the named env var.
- **ComfyUI caches model combo lists**; a freshly downloaded model file may need a
  ComfyUI restart before the bundled workflows validate.
- **Model filenames are configuration.** The bundled workflow JSONs are patched at load
  time from settings (`LDS_QWEN_EDIT_MODEL` etc.), so users don't edit JSON to match
  their filenames.
- **Trainer configs are derived from the dataset, not from constants.**
  `dataset_stats.inspect()` reads image count/dimensions (Pillow header parse only) so
  steps scale with the set (`STEPS_PER_IMAGE`, clamped to 1000–4000) and buckets come
  from the images that actually exist. Only config keys attested in each trainer's own
  examples are emitted — inventing plausible-looking keys produces configs that fail
  hours into a run.
- **`musubi_command()` takes the whole `TrainConfig`, not just the `ModelPreset`.** It
  previously took the preset alone and hardcoded `--network_dim 16` /
  `--max_train_steps 2000`, so every ⑤-tab slider was silently discarded (it even emitted
  the literal `{16}` from a broken f-string). Regression-tested in
  `tests/test_trainer_configs_0_4_0.py`.
- **Emitted configs are verified by unit tests on the rendered text, never by a live
  training run.** The ⑤ tab says so. Don't imply otherwise.
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
- **`user_settings.json` holds no secrets.** Trainer install paths, last hyperparameters,
  and the custom endpoint (URL/model/key-env-NAME/spacing) live in
  `.cache/user_settings.json` (gitignored). API keys never go there — only the *name* of
  the env var that holds the key; the secret stays in `.env`/environment.
- **Gradio only serves whitelisted paths.** The galleries display images the app writes
  to user-chosen output folders, which can be on any drive. Gradio refuses paths outside
  the CWD/temp by default (`InvalidPathError`), so `demo.launch(allowed_paths=...)` is
  fed the configured run/dataset roots plus every present drive root. This is acceptable
  only because the server binds to localhost with no auth (see Security posture).
- **User-entered output paths are validated.** `_validate_out_dir` in `app.py` rejects
  paths with characters the OS forbids (`< > : " | ? *`, control chars) with a friendly
  message, and the generate/export handlers wrap writes in `try/except OSError` — a bad
  path (e.g. a stray `|`) surfaces as a GUI hint, not a raw traceback.
- **Export sample caption is deterministic and empty-aware.** `do_export` sorts the
  numbered `.txt` files (excluding `README.txt`), shows the first non-empty one, and
  explicitly flags when captions are empty — so a blank-caption bug (e.g. a thinking
  model burning the token budget) is visible instead of looking like a UI glitch.
- **Export is selection-gated, path-keyed.** ④ requires "Load & preview" before
  export; the `CheckboxGroup` value is the full image path (not a `gr.State`), so
  identically-named files in different folders never collide and the stage stays
  stateless. `resolve_export_items()` (in `package.py`) is the single scan/classify
  seam shared by the UI and `cli.py export`; a checked image with a missing/empty
  caption is skipped and reported, never silently dropped or fatal. Caption status
  (✓ / ⚠ empty / ⚠ no caption) is surfaced per image *before* packaging.

## Roadmap / deferred

Done this revision (0.5.0):

- ✅ **Final export selection gate** — ④ gains "Load & preview" → gallery + all-checked
  `CheckboxGroup` (UNCHECK-to-drop, matching ②/③); export packages only checked images.
  Per-image caption status (✓ / empty / missing) is surfaced *before* packaging, plus
  click-to-zoom review. Shared `resolve_export_items()` de-duplicates the UI/CLI scan
  loop; verified end-to-end incl. same-name cross-folder files and empty-selection guards.

Done in 0.4.0:

- ✅ **Isolation over-cutting fixed** — comma-split exclusion terms, dilation off by
  default, no-op detection + reporting. Verified against a reference image: output is now
  byte-identical to the subject-only baseline (0 character pixels lost, was 6,594).
- ✅ **Bundled ComfyUI workflows are core-only** — no `ComfyUI_essentials` needed.
- ✅ **ComfyUI queue priority** (`front=True`) + relaxed backlog guard.
- ✅ **Wardrobe randomizer** (`wardrobe.py`) — one unisex pool, angle/pose shots only.
- ✅ **Prop-exclusion clause** on generation prompts (`apply_prop_exclusion`).
- ✅ **`exclude_prompt` wired into ②** — the pipeline accepted it but the UI never passed it.
- ✅ **Caption-tab cost estimate** (per-model, scales with selection).
- ✅ **③ → ④ folder list accumulates** instead of overwriting.
- ✅ **musubi honors every hyperparameter**; steps derived from image count; multi-res buckets.
- ✅ **CLI/UI parity** for the custom captioner endpoint (`resolve_captioner_config`).

Done in 0.3.0:

- ✅ musubi-tuner `dataset.toml` + ostris ai-toolkit `config.yaml` generators (⑤ Train tab).
- ✅ Outfit/wardrobe control (per-shot `outfit`, folded into prompts without identity drift).
- ✅ Sharpness quality flag before export (advisory Laplacian-variance).
- ✅ Bulk/inline caption editor in the UI (edit any `.txt` sidecar by hand).
- ✅ User-editable prompt libraries — save/load shot plans as YAML.

Deferred (with rationale):

- **Inpainting occluded regions** (where a prop physically covers the body, isolation
  leaves a real hole): the honest fix, but it needs a generative edit pass per source.
  The app has the machinery (Qwen-Image-Edit) — revisit if the white notch bites users.
- **Live cloud pricing**: Google publishes no pricing API; only scraping the pricing page
  could beat the build-time table, and it would break silently. Estimates are labelled.

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
  deliberately not used). `allowed_paths` is widened to drive roots so galleries can
  display images in arbitrary user output folders — this lets the localhost file route
  serve any file on those drives, which is only acceptable given the localhost bind. Do
  not expose the app; do not add `share=True`.
- **Costs and content are the user's responsibility**, stated in-app (header banner +
  "Costs & your responsibility" accordion) and in the README. Cloud calls bill the user's
  own key; custom endpoints bill whatever provider the user points at. No charges flow
  through this project.
- API keys live in `.env` (gitignored; `setup.sh` chmods it 600) or the environment,
  and are only sent to their own vendor endpoints.
- No telemetry; the only network calls are the ones the selected backends require.
- Local model weights come from Hugging Face as safetensors.
- The ⑤ Train tab **generates and displays** config files and a run command — it never
  executes a subprocess or shell, so there is no command-injection surface. Generated
  configs reference the user's own model ids/paths; no credentials are embedded.
- `.cache/user_settings.json` (gitignored) stores filesystem paths + hyperparameters only.
