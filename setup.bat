@echo off
setlocal EnableDelayedExpansion
REM LoRA Dataset Studio - one-time setup (Windows)
cd /d "%~dp0"

echo === LoRA Dataset Studio setup ===
echo.

REM --- Python check ---
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found on PATH. Install Python 3.10+ from https://www.python.org/downloads/
    echo         (check "Add python.exe to PATH" in the installer^), then re-run setup.bat
    exit /b 1
)
for /f "tokens=2" %%v in ('python --version') do echo Found Python %%v

REM --- venv ---
if not exist .venv (
    echo Creating virtual environment...
    python -m venv .venv || exit /b 1
)
set PIP=.venv\Scripts\pip.exe

REM --- torch: CUDA if an NVIDIA GPU is present, else CPU ---
where nvidia-smi >nul 2>&1
if %errorlevel%==0 (
    echo NVIDIA GPU detected - installing CUDA build of PyTorch...
    %PIP% install torch torchvision --index-url https://download.pytorch.org/whl/cu128 || exit /b 1
) else (
    echo No NVIDIA GPU detected - installing CPU build of PyTorch.
    echo NOTE: local captioning/isolation models are very slow on CPU.
    echo       Cloud captioners (Gemini/Groq^) work fine without a GPU.
    %PIP% install torch torchvision || exit /b 1
)

echo Installing dependencies...
%PIP% install -r requirements.txt || exit /b 1

REM --- optional API keys -> .env ---
echo.
echo --- Optional API keys (press Enter to skip any of them) ---
echo Keys are stored ONLY in the local .env file (gitignored, never uploaded).
echo.
echo GEMINI_API_KEY : cloud image generation + Gemini captioner.
echo   Get one at https://aistudio.google.com/apikey - usage is billed by
echo   Google to YOUR key. In-app prices are build-time estimates only.
set "GKEY="
set /p GKEY="GEMINI_API_KEY (Enter to skip): "
echo.
echo GROQ_API_KEY : free-tier cloud captioning (SFW, rate-limited).
echo   Get one at https://console.groq.com/keys
set "QKEY="
set /p QKEY="GROQ_API_KEY (Enter to skip): "
echo.
echo HF_TOKEN : needed for the built-in SAM3 subject isolation (gated model).
echo   Accept the license at https://huggingface.co/facebook/sam3 then create a
echo   read token at https://huggingface.co/settings/tokens
set "HKEY="
set /p HKEY="HF_TOKEN (Enter to skip): "

if not exist .env type nul > .env
if defined GKEY echo GEMINI_API_KEY=!GKEY!>> .env
if defined QKEY echo GROQ_API_KEY=!QKEY!>> .env
if defined HKEY echo HF_TOKEN=!HKEY!>> .env

REM --- optional ComfyUI check ---
echo.
.venv\Scripts\python.exe -c "from studio import comfy_api; import sys; sys.exit(0 if comfy_api.is_up() else 1)" >nul 2>&1
if %errorlevel%==0 (
    echo ComfyUI detected at the configured URL - the fully-local engine is available.
) else (
    echo ComfyUI not detected ^(optional^). Cloud generation + built-in SAM3 isolation
    echo work without it. For fully-local image generation, see docs\comfyui-setup.md
    echo for the required models and workflow dependencies.
)

echo.
echo === Setup complete. Run start.bat to launch. ===
endlocal
