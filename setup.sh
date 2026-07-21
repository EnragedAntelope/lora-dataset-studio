#!/usr/bin/env bash
# LoRA Dataset Studio - one-time setup (Linux / macOS)
set -euo pipefail
cd "$(dirname "$0")"

echo "=== LoRA Dataset Studio setup ==="
echo

# --- Python check ---
PY=python3
command -v "$PY" >/dev/null 2>&1 || { echo "[ERROR] python3 not found — install Python 3.10+ first."; exit 1; }
echo "Found $($PY --version)"

# --- venv ---
if [ ! -d .venv ]; then
    echo "Creating virtual environment..."
    "$PY" -m venv .venv
fi
PIP=.venv/bin/pip

# --- torch + ONNX Runtime: CUDA if an NVIDIA GPU is present, else platform default ---
if command -v nvidia-smi >/dev/null 2>&1; then
    WANT=gpu
    echo "NVIDIA GPU detected - installing CUDA build of PyTorch..."
    "$PIP" install torch torchvision --index-url https://download.pytorch.org/whl/cu128
else
    WANT=cpu
    echo "No NVIDIA GPU detected - installing default PyTorch build."
    echo "NOTE: local captioning/isolation models are very slow without a GPU"
    echo "      (Apple Silicon uses MPS and is usable). Cloud captioners"
    echo "      (Gemini/Groq) work fine without a GPU."
    "$PIP" install torch torchvision
fi

echo "Installing dependencies..."
"$PIP" install -r requirements.txt

# ONNX Runtime for the WD/e621 taggers (③), matched to the chosen build. Kept
# out of requirements.txt so the CPU vs CUDA variant tracks the torch install.
echo "Installing ONNX Runtime ($WANT) for the taggers..."
"$PIP" uninstall -y onnxruntime onnxruntime-gpu >/dev/null 2>&1 || true
if [ "$WANT" = gpu ]; then
    "$PIP" install onnxruntime-gpu || echo "[warn] onnxruntime-gpu failed - taggers can fall back to CPU via 'pip install onnxruntime'."
else
    "$PIP" install onnxruntime || echo "[warn] onnxruntime failed - the taggers need it; install later with 'pip install onnxruntime'."
fi

# --- optional API keys -> .env ---
echo
echo "--- Optional API keys (press Enter to skip any of them) ---"
echo "Keys are stored ONLY in the local .env file (gitignored, never uploaded)."
echo
echo "GEMINI_API_KEY : cloud image generation + Gemini captioner."
echo "  Get one at https://aistudio.google.com/apikey - usage is billed by"
echo "  Google to YOUR key. In-app prices are build-time estimates only."
read -r -s -p "GEMINI_API_KEY (Enter to skip): " GKEY; echo
echo
echo "GROQ_API_KEY : free-tier cloud captioning (SFW, rate-limited)."
echo "  Get one at https://console.groq.com/keys"
read -r -s -p "GROQ_API_KEY (Enter to skip): " QKEY; echo
echo
echo "HF_TOKEN : needed for the built-in SAM3 subject isolation (gated model)."
echo "  Accept the license at https://huggingface.co/facebook/sam3 then create a"
echo "  read token at https://huggingface.co/settings/tokens"
read -r -s -p "HF_TOKEN (Enter to skip): " HKEY; echo

touch .env
chmod 600 .env
[ -n "${GKEY:-}" ] && echo "GEMINI_API_KEY=$GKEY" >> .env
[ -n "${QKEY:-}" ] && echo "GROQ_API_KEY=$QKEY" >> .env
[ -n "${HKEY:-}" ] && echo "HF_TOKEN=$HKEY" >> .env

# --- optional ComfyUI check ---
echo
if .venv/bin/python -c "from studio import comfy_api; import sys; sys.exit(0 if comfy_api.is_up() else 1)" >/dev/null 2>&1; then
    echo "ComfyUI detected at the configured URL - the fully-local engine is available."
else
    echo "ComfyUI not detected (optional). Cloud generation + built-in SAM3 isolation"
    echo "work without it. For fully-local image generation, see docs/comfyui-setup.md"
    echo "for the required models and workflow dependencies."
fi

echo
echo "=== Setup complete. Run ./start.sh to launch. ==="
