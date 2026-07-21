# Architecture

Version: 0.9.0

```
app.py                  Gradio UI — thin wiring over the stage functions (5 tabs)
cli.py                  Typer CLI — one subcommand per stage + `build` for all four
studio/
  config.py             Settings (env/.env overridable), captioner registry with
                        per-model prompt templates, engine/price tables
  pipeline.py           Stage orchestration: preprocess_sources(), generate_shots()
  preprocess.py         Restore (comfyui|basic|auto) + isolate + resize
  isolate.py            Subject isolation: builtin SAM3 (transformers) | comfyui
  tagger.py             ONNX booru-tagger backend: canonical tags straight from
                        image features — WD (Danbooru) and Z3D (e621) via one code
                        path, differing only in tag file + category SCHEMES. Pure
                        select/format logic is I/O-free (unit-tested); onnxruntime +
                        hub download are lazy. The "wd_tagger" captioner backend.
                        Opt-in extras: select_rating() appends the top rating tag
                        (Danbooru cat 9), format_tag(keep_underscores=) keeps raw
                        booru tokens
  dedupe.py             Advisory near-duplicate detection (perceptual dHash,
                        numpy-only) surfaced in the ④ export preview; the Hamming
                        distance (default 5) is a ④ slider
  quality.py            Advisory sharpness (blur) + exposure/contrast flags
                        (dark/bright/low-contrast) for the curate/export views
  caption_lint.py       Advisory caption analysis (pure string logic): health lint
                        (empty/short/trigger-missing/identical captions) + tag-
                        frequency / near-ubiquitous-tag report for tag datasets.
                        Surfaced after ③ and in the ④ preview; CLI `lint`
  hf_publish.py         Optional, opt-in publish of a dataset folder to the HF Hub
                        (private by default; write HF_TOKEN; huggingface_hub lazy)
  captioner.py          Captioner backends (transformers | gemini | openai-compat |
                        wd_tagger),
                        caption finalization, standalone caption_folder().
                        Captioner takes model_override (Gemini model picker) and
                        spec_overrides (runtime config for the custom endpoint).
                        caption(style=) picks prose vs booru-tag template per spec;
                        finalize_caption(style=) normalizes tag output (dedupe,
                        lowercase, trigger-first). apply_affixes() wraps captions
                        with fixed prefix/suffix (quality/score tags);
                        drop_blacklisted_tags() strips a noisy-tag drop-list;
                        merge_tagger_overrides() folds the ③ tag-option controls into
                        spec_overrides (shared by UI + CLI); skip_existing leaves
                        already-captioned images alone
  package.py            Dataset export (NN.png/NN.txt + metadata.json + metadata.jsonl
                        + README.txt); zip_dataset() bundles a folder into a .zip;
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
  trainer_configs.py    Emit LoRA-trainer configs (ai-toolkit config.yaml / kohya
                        sd-scripts kohya-dataset.toml / musubi dataset.toml) +
                        run-command builders + model registry. ModelPreset carries
                        per-arch train/sample knobs (noise_scheduler/sample_guidance/
                        sample_steps) so SDXL (ddpm, higher CFG) renders correctly
                        beside the flux/qwen flow-matching presets
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

## Maintainer principles (standing demands)

These are the project owner's repeated, non-negotiable expectations for every change. They
sit alongside the architectural Design rules above and apply to all future work — treat a
change that violates one as incomplete.

1. **No bloat, no unwarranted complexity.** Add code, dependencies, UI, and abstractions only
   when they earn their place. Prefer reusing an existing seam over inventing a new one; prefer
   a small pure function over a framework. If a feature can't be justified as *genuinely useful*,
   it does not go in — never add things for the sake of adding them.
2. **Efficiency.** Keep startup fast (heavy imports stay lazy), keep the hot paths cheap, and
   don't do work the user didn't ask for. Advisories are numpy/string-only unless a real model
   is unavoidable.
3. **Help/tooltips stay useful and current.** Every non-obvious control carries a concise `info=`
   tooltip; when behaviour changes, update the tooltip in the same change. Guidance must be
   accurate — a stale hint is worse than none.
4. **Sensible defaults, optimized for modern models.** Defaults target current-model dataset
   building (Flux / Krea 2 / Qwen-Image / SDXL): prose captions by default, 1024 long-side, a
   modern flow-matching trainer preset, cloud engine for no-GPU users. New controls default to
   the safe/no-op choice. Re-audit defaults whenever a stage changes.
5. **README stays human-readable and useful without being long.** Features/why-use-it up top,
   marketable but accurate, scannable. Trim before adding; deep detail lives in this file.
6. **Keep every MD file updated in the same change.** `README.md` and `docs/ARCHITECTURE.md`
   (module map, gotchas, roadmap, version line) must reflect the code as it lands, not later.
7. **Log genuinely-useful future ideas here, never build them unprompted.** While working, watch
   for enhancements that fit the project shape and record them under "Further ideas identified"
   as candidate to-dos — *only if legitimately useful*. Noting is free; building without a green
   light is not.
8. **Best practices throughout.** UI/UX and ease of use, security (localhost-only, no secrets on
   disk, opt-in/outbound-flagged network), coding (typed, tested, `ruff`-clean, honest output —
   never fake capability), and GitHub hygiene (clear commits, no PRs unless asked, releases when
   a version bumps). Every version bump updates `studio.__version__` **and** ships a GitHub
   Release, or the in-app update check never fires.

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
- **Caption style is prose / Danbooru tags / e621 tags, per call — not a global mode.**
  `CaptionerSpec` carries three templates (`prompt_template` prose, `tags_template`
  Danbooru, `e621_template` furry/anthro) and `prompt_for(style)` selects one; every backend
  (transformers/gemini/openai) just sends the chosen instruction, so no backend-specific tag
  logic is needed. Tags exist because Danbooru-trained (SDXL / Illustrious / NoobAI) and
  e621-trained (Pony / furry) checkpoints learn poorly from prose — and Danbooru vs e621 are
  **different controlled vocabularies**, hence two separate tag templates, not one. Both tag
  styles share `finalize_caption(style in {"tags","e621"})`, which is deliberately
  *different* from prose: it lowercases, dedupes and comma-joins (`_normalize_tags`), keeps
  the trigger first, and does **not** inject the character name as a tag (identity is the
  trigger). The prose path is unchanged. An unknown style value falls back to prose rather
  than raising. Caveat worth remembering: the backend *instructs* a general VLM to emit
  tags, so output approximates the vocabulary; a dedicated tagger would be needed for a
  canonical tag set (backlog).
- **The tagger is a captioner backend, not a style.** `backend="wd_tagger"` runs an ONNX
  tagger that emits canonical tags directly from image features — so it *ignores* the
  prose/tags/e621 selector, and both `caption_images` and the UI's test path force
  `style="tags"` for it (tag finalization). Its pure selection/formatting logic
  (`select_tag_names`, `format_tag`) lives in `tagger.py` free of model/file I/O and is
  unit-tested; `onnxruntime` and the huggingface download are lazy (inside `Tagger.load`),
  so nobody who doesn't pick a tagger pays for them. Preprocessing is exact and non-obvious
  (white-pad to square, resize, **RGB→BGR**, float32 **0–255, no normalization**, NHWC) — do
  not "tidy" it. Outputs are already sigmoid probabilities.
- **Danbooru and e621 taggers share one code path; only the tag file + categories differ.**
  WD (`selected_tags.csv`) and Z3D-E621 (`tags-selected.csv`) both expose `name`/`category`
  columns, so `_read_selected_tags` reads either. The *meaning* of the category ids differs,
  captured in `SCHEMES`: Danbooru {general:0, character:4}; e621 {general:0+5(species),
  character:1(artist)/3(copyright)/4/8(lore)} with meta/invalid dropped. e621 **species**
  rides the general threshold (it reads like a descriptor for training); artist/character/
  copyright ride the higher character threshold so an original subject isn't stamped with a
  known artist/character. The e621 model (`toynya/Z3D-E621-Convnext`) is a community weight —
  documented as such; its output should be spot-checked. Same white-pad/BGR/0–255 recipe as
  WD (the standard tagger extensions use it for both).
- **Tagger thresholds flow through `spec_overrides`, not a new arg.** The ③ sliders set
  `general_threshold`/`character_threshold` via the same `CaptionerSpec.model_copy(update=…)`
  path the custom endpoint uses — no bespoke plumbing, and the registry spec stays pristine.
  The rating and keep-underscores toggles ride the same seam (`include_rating`,
  `keep_underscores`). `merge_tagger_overrides()` is the single place UI *and* CLI fold those
  four controls in, and it is a no-op for any non-tagger captioner — so a stray threshold from
  the UI can never leak into a Gemini/Groq spec.
- **Rating tags are opt-in and Danbooru-only.** WD taggers expose a rating category (cat 9:
  general/sensitive/questionable/explicit) that is otherwise dropped; `include_rating` appends
  only the single highest-confidence one (ratings are mutually exclusive). The e621/Z3D scheme
  carries **no** rating category — ratings aren't tags there — so `SCHEMES["e621"]["rating"]`
  is empty and the toggle is a documented no-op for Z3D, not a wrong guess at a category id.
- **The tag drop-list runs before affixes, never on prose, never on the trigger.**
  `drop_blacklisted_tags` filters a comma tag list against a normalized blacklist (lowercase,
  underscores→spaces, so `long_hair`/`Long Hair` both match `long hair`). It is applied after
  `finalize_caption` but before `apply_affixes`, so a fixed prefix/suffix survives the drop; it
  is a strict no-op for prose (`style` not a tag style) or an empty list; and the first token
  (the identity trigger) is always kept even if the user lists it. `parse_blacklist` is the
  shared normalizer (comma/newline tolerant).
- **Caption prefix/suffix is a post-finalize wrap, applied once.** `apply_affixes` runs after
  `finalize_caption` (so the trigger is already placed) and joins with a comma for tag styles,
  a space for prose. It exists for constant tags the model shouldn't be asked to invent —
  Pony `score_9, …`, Danbooru `masterpiece, best quality`. Empty prefix/suffix is a strict
  no-op, so existing captions are unchanged.
- **Composition flags are honest heuristics, not a framing model.** `quality.composition_flags`
  labels exposure/contrast outliers (dark/bright/low-contrast) from mean/std of luminance —
  numpy-only, advisory, shown next to the blur flag in ②. It deliberately does **not** claim
  semantic framing (face/bust/body/back); that needs a detector and is left to the backlog.
  Like sharpness it never blocks — the curate gallery wraps it in try/except.
- **Caption lint is advisory string logic, never a rewrite.** `caption_lint.py` mirrors the
  blur/dedup pattern: it flags empty / too-short / trigger-missing / byte-identical captions and
  (for tag datasets only) near-ubiquitous tags, but never edits or blocks. Two deliberate
  choices keep it trustworthy: `min_words` defaults to 2 so only near-junk is called "short"
  (a concise real caption is never nagged), and the tag-frequency report is gated on
  `looks_like_tags()` (average ≤3 words per comma segment) so prose captions don't produce a
  meaningless frequency list. In ④ it runs on captioned pairs only with an empty trigger —
  empties are already summarized by the export note and the trigger is unknown at preview time,
  so it adds only the short/duplicate/ubiquitous signals, no double-counting. All call sites
  wrap it in try/except; an unreadable folder degrades to no report, never an error.
- **Near-duplicate detection is advisory dHash, surfaced in ④.** `dedupe.find_near_duplicate_
  groups` groups images within a Hamming distance (default 5, exposed as a ④ slider so users can
  loosen/tighten it) and the export preview lists them so the user can uncheck over-weighted
  repeats. It never auto-drops. Note a dHash quirk: flat images (solid colour) all hash alike,
  so a set of near-blank frames may group together — acceptable for an advisory.
- **`metadata.jsonl` is written for every export.** `package_dataset` emits the HuggingFace
  `imagefolder` sidecar (`{"file_name","text"}` per image) alongside `metadata.json`, so the
  dataset also loads via `datasets.load_dataset("imagefolder", …)`. `.txt`-based trainers
  ignore it; it carries captions only, no secrets.
- **kohya sd-scripts uses a different dataset layout from musubi.** Even though musubi is a
  kohya fork, sd-scripts wants `[[datasets.subsets]]` with `image_dir` (not musubi's
  `[[datasets]]` + `image_directory`), so `render_kohya_toml` is its own renderer. SDXL base
  is a runnable HF id; a family checkpoint (Pony/Illustrious/NoobAI) is a user-local
  `<<FILL>>`. `kohya_command` threads every hyperparameter from the `TrainConfig` (same
  discipline as the musubi fix) and uses `sdxl_train_network.py`. Verified by unit tests on
  the rendered text, never a live run.
- **SDXL is not flow-matching.** The ai-toolkit renderer used to hardcode `noise_scheduler:
  flowmatch` and `guidance_scale: 4`, which are correct for Flux/Qwen/Z-Image but wrong for
  SDXL (wants `ddpm` and CFG ~7). Those three knobs now live on `ModelPreset`
  (`noise_scheduler`/`sample_guidance`/`sample_steps`) with the old values as defaults, so
  every existing preset renders byte-identically and only SDXL overrides them. `guidance`
  is formatted `:g` so `4.0`→`4` (keeps the flow-matching output unchanged). As always, only
  arch keys attested by ai-toolkit are emitted; the SDXL-family preset (Pony/Illustrious/
  NoobAI) keeps a `<<FILL>>` model path because those checkpoints are user-local.
- **HF publishing is opt-in and private by default.** `hf_publish.publish_dataset` never
  runs on its own — it needs an explicit UI button click or the CLI `--publish-hf` flag.
  `create_repo(private=True)` unless the user deliberately unchecks it; the write token comes
  only from `HF_TOKEN` (env/.env) and is never written to disk. `normalize_repo_id` validates
  the id (I/O-free, unit-tested) and the missing-token / missing-folder guards fail fast
  *before* `huggingface_hub` is imported. Only files inside the dataset folder are uploaded.
- **`zip_dataset` arcnames are derived from the folder, never absolute.** Entries are stored
  under `<dataset-name>/…` (so it extracts tidily) and the archive is non-clobbering
  (`-2.zip`, …) — no path-traversal surface because we only ever add files found *inside*
  the packaged dataset dir.
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

Done this revision (0.9.0) — four of the five "further ideas" logged in the 0.8.0 review
(resolution normalization deferred), plus setup/requirements housekeeping:

- ✅ **Global tag drop-list (③).** `drop_blacklisted_tags` strips noisy tags (`simple
  background`, `signature`, `watermark`) from tag captions across a folder; UI *Tag options*
  field + CLI/`build` `--drop-tags`. Trigger-safe, prose no-op, runs before affixes.
- ✅ **Optional rating tags for taggers (③).** `include_rating` appends the top WD rating tag
  (Danbooru cat 9); UI checkbox + CLI `--rating-tags`. Documented no-op for e621/Z3D.
- ✅ **Keep-underscores toggle for taggers (③).** `format_tag(keep_underscores=)`; UI checkbox
  + CLI `--keep-underscores`. Kaomoji still preserved.
- ✅ **Configurable near-dup sensitivity (④).** The dHash Hamming distance is a ④ slider
  (default 5) threaded into `find_near_duplicate_groups`.
- ✅ **Caption health lint (③/④).** `caption_lint.lint_captions` flags empty / short / trigger-
  missing / identical captions; shown after ③, in the ④ preview, and via CLI `lint`. Advisory.
- ✅ **Tag-frequency / ubiquitous-tag report (③/④).** `caption_lint.tag_frequency` +
  `ubiquitous_tags` surface tags on nearly every image (drop-list candidates), gated to tag
  datasets. Pairs with the drop-list to tell the user *what* to drop.
- ✅ **GPU/CPU setup choice + onnxruntime move.** `setup.bat` now offers an NVIDIA-GPU-optimized
  vs CPU-only install and is re-runnable to switch an existing venv between them (torch +
  onnxruntime). `onnxruntime` moved out of `requirements.txt` into the setup scripts so the CPU
  (`onnxruntime`) or CUDA (`onnxruntime-gpu`) variant tracks the chosen build; `setup.sh` mirrors
  this via nvidia-smi auto-detection.

Done in 0.8.0 — remaining backlog items 2–3 + all six 0.7.0 "further ideas":

- ✅ **e621 tagger (③)** — `toynya/Z3D-E621-Convnext` added through the generalized `Tagger`
  (scheme-driven categories); canonical furry/Pony tags. Shares the WD code path.
- ✅ **Composition advisory in curate (②)** — `quality.composition_flags` (dark/bright/
  low-contrast) shown next to the blur flag; honest exposure heuristics, advisory only.
- ✅ **Tagger threshold controls (③)** — general/character sliders threaded via `spec_overrides`.
- ✅ **Caption prefix/suffix (③/④)** — `apply_affixes` for constant quality/score tags (Pony,
  Danbooru), UI *Tag options* + CLI `--prefix/--suffix`.
- ✅ **Skip already-captioned (③)** — `skip_existing` leaves images with a caption untouched;
  UI checkbox + CLI `--skip-captioned`.
- ✅ **Near-duplicate advisory (④)** — `dedupe.py` (dHash) flags near-identical groups in the
  export preview; never auto-drops.
- ✅ **`metadata.jsonl` (④)** — HuggingFace `imagefolder` sidecar written on every export.
- ✅ **kohya sd-scripts trainer (⑤)** — SDXL LoRA `kohya-dataset.toml` + command (its own
  subsets layout); every hyperparameter threaded, unit-tested.

Done in 0.7.0 — market-gap backlog items 2–5, in one pass:

- ✅ **Dedicated WD tagger backend (③)** — `tagger.py` + the `wd_tagger` captioner backend
  (WD EVA02-Large / ViT v3) emit **canonical Danbooru tags** from the image, for higher tag
  fidelity than instructing a VLM. Lazy `onnxruntime`/hub, unit-tested pure logic.
- ✅ **Expanded trainer registry — SDXL (⑤)** — honest SDXL presets (ai-toolkit) with the
  correct `ddpm` scheduler / CFG via new per-arch `ModelPreset` knobs; flow-matching presets
  render unchanged. A `<<FILL>>` SDXL-family preset covers Pony / Illustrious / NoobAI.
- ✅ **ZIP export (④)** — `package.zip_dataset()` + a UI checkbox and CLI `--zip`; tidy,
  non-clobbering, traversal-safe archive.
- ✅ **Hugging Face dataset publishing (④)** — `hf_publish.py` + a UI accordion and CLI
  `--publish-hf`; opt-in, **private by default**, `HF_TOKEN` never persisted, guards tested.

Done in 0.6.0:

- ✅ **Tag caption styles (Danbooru + e621)** — ③ Caption (and `--caption-style` on the CLI)
  can now emit a comma-separated tag list instead of prose, matching tag-trained base models:
  **Danbooru** for SDXL / Illustrious / NoobAI, **e621** (furry/anthro vocabulary) for Pony
  and furry checkpoints. Per-model templates for each style (generic / JoyCaption / NSFW),
  shared tag normalization (dedupe, lowercase, trigger-first, name-not-injected), UI radio,
  CLI flag with validation, and unit tests. Prose remains the default and is unchanged.

Done in 0.5.0:

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

### Market-gap backlog

Scan of the broader LoRA-dataset-tooling space (some tools go much further and also
orchestrate *training* — cloud GPU rental, checkpoint-comparison studios, model merging,
web scraping, face-recognition curation). Most of that is out of scope for this tool by
design — we deliberately generate and display trainer configs and never launch training,
scrape, or bundle a heavyweight frontend. The items below are the ones that *fit* this
project's shape (standalone stages, per-stage local/cloud choice, lazy heavy imports,
honest output), prioritized by benefit-to-cost. Take from the top.

**Shipped** (items 1–5 + 2–3): tag caption styles (0.6.0); WD tagger, SDXL presets, ZIP
export, HF publishing (0.7.0); e621 tagger + composition advisory (0.8.0). Remaining, still
prioritized by benefit-to-cost:

1. **Concept & Style dataset types (②).** We are character-only; a "style" mode (no
   isolation, prose/tags describe the aesthetic, trigger always-on) and a "concept" mode
   (object/action) are a larger shot-plan change but a genuine capability gap. Design before
   building — it touches ②'s whole shot model. **Now the top remaining item.**
2. **Face-similarity identity guard (② curate).** Flag generated shots that drift from the
   reference's identity. Genuinely useful for character LoRAs but needs a face-recognition
   dependency (InsightFace + onnxruntime); previously deferred for that reason. Revisit if
   identity drift bites users — could be an optional extra like the gated SAM3 download.

*(All six "further ideas" logged in the 0.7.0 review shipped in 0.8.0 — prefix/suffix,
tagger thresholds, near-dup advisory, `metadata.jsonl`, kohya trainer, skip-already-captioned.)*

### Further ideas identified (0.8.0 review)

Four of these five shipped in 0.9.0 (drop-list, rating tags, keep-underscores, near-dup
sensitivity — see the top of this section). The one still open:

- **Resolution normalization at export (④).** Optional pad-to-square / center-crop to a target
  size for trainers that want uniform square inputs — advisory today via buckets, but a real
  transform would help SDXL-at-1024 users. Larger; design first. **Deliberately deferred** —
  the bucket ladder covers most cases and a real transform mutates pixels, so it wants its own
  design pass rather than being rushed into this release.

### Further ideas identified (0.9.0 review)

Surfaced while shipping this round. All fit the project shape (standalone/advisory, no new
heavy deps, honest output). The top two — **caption health lint** and the **tag-frequency /
ubiquitous-tag report** — shipped in 0.9.0 (see "Done this revision" above). Remaining,
prioritized by benefit-to-cost:

- **Caption-style ↔ trainer sanity check (④→⑤).** Record the caption style in export
  `metadata.json` (the CLI `build` already records it; the UI export does not) and warn at ⑤
  when prose captions feed a tag-trained preset (SDXL / Pony / Illustrious) or tags feed a
  prose model (Flux / Krea). A cross-stage honesty check, no new deps.
- **CLIP 77-token truncation warning for tag datasets (③/④).** SDXL / Illustrious silently
  truncate captions past 77 tokens; a cheap comma/word-count heuristic can flag captions that
  will be cut (optional lazy tokenizer for exactness). Advisory.
- **Tighten-to-subject crop after isolation (①).** After the SAM3 cutout, optionally crop to
  the subject's bounding box (numpy on the mask) so framing is consistent across shots and less
  white padding is trained. Small, opt-in.
- **Alpha (RGBA) cutout option (①).** Export the isolated subject on transparency instead of
  white for workflows that composite their own backgrounds. Small toggle; white stays the
  default (the Multiple-Angles LoRA is trained on white).

**Explicitly not pursuing** (conflict with this project's scope): in-app training launch /
cloud GPU rental, Test Studio / checkpoint ranking, Merge Lab, and web scraping. The
first three are the "never launch training" line we hold deliberately (see Deferred); web
scraping carries rights/ToS liability we don't want to put one click away.

**GitHub About description / topics — revisit with the style-support release.** Proposed
marketable copy is held until Concept/Style dataset types land (they change what the tool
"does"), so the About blurb is refreshed once, accurately, alongside that feature rather than
now. Draft to revisit then:
- *Description:* "Turn one image of a character into a ready-to-train LoRA dataset — consistent
  multi-angle shots, smart captions (prose/Danbooru/e621), and trainer-ready configs for Flux,
  SDXL, Krea, Pony & more. Local or cloud, per stage. No training launched, no telemetry."
- *Topics:* lora, lora-training, dataset-generation, stable-diffusion, flux, sdxl,
  pony-diffusion, image-captioning, wd-tagger, sam3, comfyui, qwen-image, ai-toolkit, kohya,
  gradio, huggingface, generative-ai.

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
- **Publishing to Hugging Face is opt-in, private by default, and outbound.** It runs only on
  an explicit button/`--publish-hf`, creates the dataset private unless the user deliberately
  chooses public, and reads the write token only from `HF_TOKEN` (never persisted). It
  uploads every image in the dataset folder to a remote host — the in-app notice states the
  user is responsible for the rights to that content and for HF's terms. Once uploaded,
  content may be cached/indexed even if later deleted.
- Local model weights come from Hugging Face as safetensors; the WD tagger downloads an ONNX
  model + tag CSV from a public repo (no gating). `onnxruntime` is an optional, lazy dep.
- The ⑤ Train tab **generates and displays** config files and a run command — it never
  executes a subprocess or shell, so there is no command-injection surface. Generated
  configs reference the user's own model ids/paths; no credentials are embedded.
- `.cache/user_settings.json` (gitignored) stores filesystem paths + hyperparameters only.
