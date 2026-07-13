# LoRA Dataset Studio

Turn **one image** (or a few) of a character into a **ready-to-train LoRA dataset**:
~24 consistent shots across camera angles, poses, emotions, and settings, each with a
natural-language caption `.txt` (trigger word first), packaged in a flat folder that drops
straight into **ai-toolkit / OneTrainer** — and can emit a ready-to-edit **training config**
for **ai-toolkit** or **musubi-tuner**.

**Every stage is optional and standalone.** Point any tab (or CLI subcommand) at any
folder of images — preprocess only, generate only, caption only ("tag this folder"),
or export only. Run them in order and each step auto-fills the next one's input.

**Every stage offers a local or a cloud path.** Fully-local (private, free, uncensored)
or cloud (zero GPU requirements) — mix and match per stage.

> ## ⚠️ Costs & your responsibility
>
> **Costs.** Local options are **free** (they run on your own hardware). The **cloud**
> options cost money and are **billed directly to the API key you provide**, not to this
> project:
> - **Gemini image generation** and **Gemini captioning** → billed by Google to your key.
>   In-app prices are build-time estimates; always check [current Google pricing](https://ai.google.dev/pricing).
> - **Groq** captioning → free tier (rate-limited).
> - **Custom OpenAI-compatible endpoints** you add → billed to you by that provider.
>
> This tool never charges you and takes no cut; every charge is between you and the
> provider whose key you supply.
>
> **What you make is on you.** You are **solely responsible** for the source images you
> supply and for everything you generate, caption, export, or transmit. Only use images
> you have the rights to, and comply with each model/provider's acceptable-use policy and
> all applicable laws. This software is provided under the [MIT License](LICENSE) **with no
> warranty** — the authors are not liable for your use of it, for provider charges you
> incur, or for content you create with it.

| Stage | Local option | Cloud option |
|---|---|---|
| ① Preprocess: restore/upscale | ComfyUI model restore *(optional)* or basic Lanczos | — |
| ① Preprocess: subject isolation | **Built-in SAM3** (no ComfyUI needed) or ComfyUI SAM3 | — |
| ② Generate shots | ComfyUI: Qwen Image Edit 2511 + Multiple-Angles LoRA | Gemini image models (Nano Banana) |
|| ③ Caption | Local VLMs via `transformers`: Qwen3-VL-8B, JoyCaption, NSFW finetune (also LM Studio / Ollama / any OpenAI-compatible endpoint) | Gemini Flash, Groq free tier (Qwen3.6 27B) |
| ④ Export | Always local | — |
| ⑤ Train config *(optional)* | Generate ai-toolkit / musubi-tuner config for your dataset | — |

## Quick start

```text
# Windows
setup.bat      # one time: venv, PyTorch (CUDA auto-detected), deps, optional API keys
start.bat      # launches the UI at http://127.0.0.1:7861

# Linux / macOS
./setup.sh
./start.sh
```

Setup offers to store API keys in a local `.env` (gitignored, never uploaded).
You can skip all of them and stay fully local, or add them later — copy
[.env.example](.env.example) to `.env`.

### API keys (only for the cloud options)

- **Google Gemini** — cloud image generation and/or Gemini captioning.
  Create a key at <https://aistudio.google.com/apikey>. **Costs are billed by Google
  to your key.** In-app prices (e.g. ~$0.134/image for `gemini-3-pro-image-preview` at 1K,
  ~$0.067 for `gemini-3.1-flash-image-preview`, $0.039 for `gemini-2.5-flash-image`) are
  estimates captured at build time. The model list can be live-refreshed from the API and
  saved locally for instant dropdown loads. Always check current Google pricing.
- **Groq** — free-tier cloud captioning (SFW; requests are auto-spaced and 429s retried).
  Create a key at <https://console.groq.com/keys>.
- **Hugging Face token** — only for the **built-in SAM3 isolation**: `facebook/sam3` is a
  gated model. Accept the license on the [model page](https://huggingface.co/facebook/sam3),
  create a *read* token at <https://huggingface.co/settings/tokens>, and put it in `.env`
  as `HF_TOKEN` (or run `hf auth login`). Weights (~3.4 GB) download once on first use.

### ComfyUI (optional — only for the fully-local generation/restore path)

Cloud generation, built-in SAM3 isolation, local captioning, and export all work
**without ComfyUI**. If you want free/private/uncensored image generation or
model-based photo restoration, see **[docs/comfyui-setup.md](docs/comfyui-setup.md)**
for the required models (Qwen Image Edit 2511, the fal Multiple-Angles LoRA,
restoration models, SAM3 checkpoint) and how the bundled workflow templates find them.

## Using the UI

1. **① Preprocess (optional)** — upload images or point at a folder. Degraded sources
   are restored/upscaled (only when needed, or on demand), the subject is cut out onto
   a white background (SAM3; editable subject prompt, plus an "objects to remove" prompt
   for held props like microphones), and everything is sized to the target resolution.
2. **② Generate & curate** — review/edit the curated shot plan (9 angles, 8 poses,
   7 emotion close-ups; each row combines a unique setting/lighting so the dataset isn't
   skewed toward a single standing pose). An **outfit** column varies wardrobe without
   breaking identity (leave blank to keep the reference's clothing), and you can **save/load
   plans** as reusable prompt libraries. Pick the engine, generate. Uncheck rejects;
   blurry shots are flagged (`⚠ blurry`) so they're easy to drop.
3. **③ Caption** — point at **any** folder (not just pipeline output), select images,
   pick a captioner, and write `.txt` sidecars. 🧪 tests one caption first so you can
   compare captioners cheaply. Each captioner uses a prompt tuned to that model
   (JoyCaption's documented instruction convention, explicitness for the NSFW finetune, …).
   For the **Gemini** captioner, a 🔄 button refreshes the live model list and lets you
   pick a specific model. For a **Custom OpenAI-compatible endpoint**, open *Custom
   endpoint settings* to set a base URL / model / API-key env var / request spacing (saved
   locally, key stays in `.env`) — use OpenRouter, vLLM, a proxy, etc. (**you pay that
   provider**). An **inline editor** lets you tweak any caption by hand and save it back.
4. **④ Export** — list one or more captioned folders (e.g. prepped sources + kept shots);
   get a flat `NN.png` + `NN.txt` dataset folder with `metadata.json` and `README.txt`.
5. **⑤ Train (configs, optional)** — pick a trainer (**ai-toolkit** or **musubi-tuner**)
   and model, tune rank/alpha/steps/lr/resolution, and generate a training config written
   into the dataset folder plus the exact run command. Your trainer install path is saved
   for next time. ai-toolkit is genuinely one-command (`python run.py config.yaml`); musubi
   needs you to fill in your local model paths. **Nothing is launched from the app** — the
   config and command are generated for you to run in the trainer's own environment.

## Using the CLI

```bash
python cli.py preprocess ./sources --out ./prepped
python cli.py generate ./prepped --name "Sy Snootles" --engine comfyui
python cli.py caption ./any/folder --trigger sysnootles     # writes .txt sidecars
python cli.py export ./prepped ./generated --name "Sy Snootles" --trigger sysnootles
python cli.py build source.png --name "Sy Snootles" --trigger sysnootles   # all four
```

Each subcommand is fully standalone; `--help` on any of them shows all options.

## Captioners

| Captioner | Runs | Notes |
|---|---|---|
| Qwen3-VL-8B Instruct (heretic) *(default)* | your GPU, ~17 GB bf16 | best instruction-following, NSFW-capable |
| JoyCaption Beta One | your GPU, ~17 GB bf16 | purpose-built diffusion captioner |
| Qwen3-VL-8B NSFW-Caption V4.5 | your GPU, ~17 GB bf16 | explicit-dataset specialist |
| Gemini Flash | Google API | SFW, billed to your key (~$0.001/img build-time estimate). Defaults to the `gemini-flash-latest` rolling alias; the Caption tab can refresh & pick a specific model. |
|| Groq Qwen3.6 27B | Groq API | SFW, free tier, 8K TPM. A reasoning model — the app disables its thinking scratchpad so you get just the caption. |
| LM Studio / Ollama | your machine | advanced: whatever vision model you serve |
| Custom (OpenAI-compatible) | any endpoint | advanced: your own base URL + model + optional API-key env var (OpenRouter, vLLM, a proxy, …). You pay that provider. |

Local models download automatically from Hugging Face on first use. Add your own in
`studio/config.py` (`CAPTIONERS`) — each entry carries its own prompt template. Cloud
captioners retry `429` rate-limits with backoff and space requests automatically.

## Caption format

`{trigger}, {one-paragraph natural-language description}` — the description covers pose,
camera angle, setting, and lighting (the things that *vary*), not the character's fixed
appearance (identity is what the trigger token learns). Generic nouns like "the creature"
are rewritten to the character's name.

## Good to know

- **No GPU?** Use cloud generation + cloud captioning; skip isolation or use the ComfyUI
  backend on another machine. Local 8B captioners are impractical on CPU.
- **Rear views are chained** (`chain_from` column): back shots build on a generated side
  view — direct front→back generation hallucinates anatomy on unusual characters.
- **Sources are never modified**; every stage writes copies into `runs/` (or your chosen
  output folder).
- The UI binds to `127.0.0.1` only. Don't expose it publicly — there's no authentication,
  and the process can read your `.env` keys.
- Model licenses are your responsibility: `facebook/sam3` is gated under Meta's license;
  check the licenses of any captioner/edit models you download.

More detail on the internals: **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)**.

## License

[MIT](LICENSE)
