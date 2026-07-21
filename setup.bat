@echo off
setlocal EnableDelayedExpansion
REM LoRA Dataset Studio - one-time setup (Windows). Safe to re-run: use it to
REM switch an existing install between the CPU-only and NVIDIA-GPU PyTorch builds.
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
set PY=.venv\Scripts\python.exe

REM --- choose the compute build (GPU-optimized vs CPU-only) ---
set "GPU_DEFAULT=2"
where nvidia-smi >nul 2>&1 && set "GPU_DEFAULT=1"
echo.
echo Choose your PyTorch / ONNX Runtime build:
echo    [1] NVIDIA GPU (CUDA) - fast local generation, captioning and isolation
echo    [2] CPU only          - no NVIDIA GPU; use the cloud options for heavy stages
if "%GPU_DEFAULT%"=="1" (
    echo    ^(NVIDIA GPU detected - [1] recommended^)
) else (
    echo    ^(No NVIDIA GPU detected - [2] recommended^)
)
set "CHOICE="
set /p CHOICE="Enter 1 or 2 [default %GPU_DEFAULT%]: "
if not defined CHOICE set "CHOICE=%GPU_DEFAULT%"
if "%CHOICE%"=="1" (set "WANT=gpu") else (set "WANT=cpu")

REM --- what is installed now? (so a re-run can switch builds) ---
set "CURRENT=none"
%PY% -c "import torch,sys; sys.stdout.write('gpu' if torch.version.cuda else 'cpu')" > "%TEMP%\lds_torch.txt" 2>nul
if exist "%TEMP%\lds_torch.txt" (
    set /p CURRENT=<"%TEMP%\lds_torch.txt"
    del "%TEMP%\lds_torch.txt" >nul 2>&1
)
if "%CURRENT%"=="" set "CURRENT=none"

if "%CURRENT%"=="%WANT%" (
    echo PyTorch ^(%WANT%^) already installed - skipping reinstall.
) else (
    if not "%CURRENT%"=="none" (
        echo Switching PyTorch from %CURRENT% to %WANT% - reinstalling...
        %PIP% uninstall -y torch torchvision >nul 2>&1
    )
    if "%WANT%"=="gpu" (
        echo Installing CUDA build of PyTorch...
        %PIP% install torch torchvision --index-url https://download.pytorch.org/whl/cu128 || exit /b 1
    ) else (
        echo Installing CPU build of PyTorch.
        echo NOTE: local captioning/isolation models are very slow on CPU.
        echo       Cloud captioners ^(Gemini/Groq^) work fine without a GPU.
        %PIP% install torch torchvision || exit /b 1
    )
)

echo Installing dependencies...
%PIP% install -r requirements.txt || exit /b 1

REM --- ONNX Runtime for the WD/e621 taggers (③), matched to the chosen build ---
REM Kept out of requirements.txt so the CPU vs CUDA variant tracks your choice.
echo Installing ONNX Runtime (%WANT%) for the taggers...
%PIP% uninstall -y onnxruntime onnxruntime-gpu >nul 2>&1
if "%WANT%"=="gpu" (
    %PIP% install onnxruntime-gpu || echo [warn] onnxruntime-gpu failed - taggers will fall back to CPU if you `pip install onnxruntime`.
) else (
    %PIP% install onnxruntime || echo [warn] onnxruntime failed - the taggers need it; install it later with `pip install onnxruntime`.
)

REM --- optional API keys -> .env (skipped if already present, so re-runs are clean) ---
if not exist .env type nul > .env
echo.
echo --- Optional API keys (press Enter to skip any of them) ---
echo Keys are stored ONLY in the local .env file (gitignored, never uploaded).
echo.
findstr /b /c:"GEMINI_API_KEY=" .env >nul 2>&1
if errorlevel 1 (
    echo GEMINI_API_KEY : cloud image generation + Gemini captioner.
    echo   Get one at https://aistudio.google.com/apikey - usage is billed by
    echo   Google to YOUR key. In-app prices are build-time estimates only.
    set "GKEY="
    set /p GKEY="GEMINI_API_KEY (Enter to skip): "
    if defined GKEY echo GEMINI_API_KEY=!GKEY!>> .env
    echo.
) else (
    echo GEMINI_API_KEY already set in .env - skipping.
)
findstr /b /c:"GROQ_API_KEY=" .env >nul 2>&1
if errorlevel 1 (
    echo GROQ_API_KEY : free-tier cloud captioning (SFW, rate-limited).
    echo   Get one at https://console.groq.com/keys
    set "QKEY="
    set /p QKEY="GROQ_API_KEY (Enter to skip): "
    if defined QKEY echo GROQ_API_KEY=!QKEY!>> .env
    echo.
) else (
    echo GROQ_API_KEY already set in .env - skipping.
)
findstr /b /c:"HF_TOKEN=" .env >nul 2>&1
if errorlevel 1 (
    echo HF_TOKEN : built-in SAM3 subject isolation (gated model) + HF dataset publishing.
    echo   Accept the license at https://huggingface.co/facebook/sam3 then create a
    echo   token at https://huggingface.co/settings/tokens
    set "HKEY="
    set /p HKEY="HF_TOKEN (Enter to skip): "
    if defined HKEY echo HF_TOKEN=!HKEY!>> .env
) else (
    echo HF_TOKEN already set in .env - skipping.
)

REM --- optional ComfyUI check ---
echo.
%PY% -c "from studio import comfy_api; import sys; sys.exit(0 if comfy_api.is_up() else 1)" >nul 2>&1
if %errorlevel%==0 (
    echo ComfyUI detected at the configured URL - the fully-local engine is available.
) else (
    echo ComfyUI not detected ^(optional^). Cloud generation + built-in SAM3 isolation
    echo work without it. For fully-local image generation, see docs\comfyui-setup.md
    echo for the required models and workflow dependencies.
)

echo.
echo === Setup complete ^(%WANT% build^). Run start.bat to launch. ===
echo     Re-run setup.bat any time to switch between the GPU and CPU builds.
endlocal
