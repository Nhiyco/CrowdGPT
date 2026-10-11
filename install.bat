@echo off
setlocal EnableExtensions EnableDelayedExpansion

REM ============================================================
REM CrowdGPT Universal Installer for Windows
REM v9.6
REM  - v9.6: pin pip < 25 before installing from AMD's ROCm repo
REM          (pip 25+ asserts on hashless direct URLs, which is
REM          what repo.radeon.com serves — install crashes with
REM          "assert link.hash" in direct_url_helpers.py)
REM  - v9.6: HIP_GFX_OVERRIDE is cleared when falling back to CPU,
REM          so a CPU install doesn't bake a bogus
REM          HSA_OVERRIDE_GFX_VERSION into the launcher
REM  - v9.6: fixed compound `if A if B (...)` in :do_integration
REM          that crashed CMD's parser on `!HIP_GFX_OVERRIDE!`
REM          (CMD parses the block eagerly regardless of A)
REM  - v9.5: AMD ROCm path uses Python 3.12 (AMD publishes cp312
REM          Windows wheels only)
REM  - v9.4: HSA_OVERRIDE_GFX_VERSION baked into launchers; real
REM          ROCm matmul in verify
REM  - v9.3: fresh install; HIP SDK detection
REM  - v9.2: broaden ROCm allow-list to all RDNA3 consumer GPUs
REM ============================================================

set "REPO_URL=https://github.com/Vxtzq/CrowdGPT.git"
set "REPO_DIR=CrowdGPT"
set "BACKEND="
set "TORCH_INDEX="
set "WHEEL_NOTE="
set "LAUNCH=1"
set "INTEGRATE=1"
set "BACKEND_OVERRIDE="
set "AMD_GPU_NAME="
set "FRESH=1"
set "HIP_SDK_FOUND=0"
set "HIP_GFX_OVERRIDE="
set "HIP_OVERRIDE_OPT_OUT=0"
set "HIP_LAUNCH_BLOCKING=0"
set "PY_VER=3.11"

REM ---------- arg parsing ----------
:parse_args
if "%~1"=="" goto args_done
if /I "%~1"=="--backend" (
    set "BACKEND_OVERRIDE=%~2"
    shift & shift
    goto parse_args
)
if /I "%~1"=="--dir" (
    set "REPO_DIR=%~2"
    shift & shift
    goto parse_args
)
if /I "%~1"=="--hip-override" (
    set "HIP_GFX_OVERRIDE=%~2"
    shift & shift
    goto parse_args
)
if /I "%~1"=="--no-hip-override" (
    set "HIP_OVERRIDE_OPT_OUT=1"
    shift
    goto parse_args
)
if /I "%~1"=="--debug" (
    set "HIP_LAUNCH_BLOCKING=1"
    shift
    goto parse_args
)
if /I "%~1"=="--no-launch" (
    set "LAUNCH=0"
    shift
    goto parse_args
)
if /I "%~1"=="--no-integrate" (
    set "INTEGRATE=0"
    shift
    goto parse_args
)
if /I "%~1"=="--no-fresh" (
    set "FRESH=0"
    shift
    goto parse_args
)
if /I "%~1"=="--help" goto show_help
if /I "%~1"=="-h" goto show_help
echo [ERROR] Unknown argument: %~1
echo         Run install.bat --help for usage.
exit /b 1

:show_help
echo.
echo CrowdGPT installer - Windows
echo.
echo Options:
echo   --backend ^<name^>       Force backend: nvidia ^| amd ^| amd-rocm ^| intel ^| cpu
echo   --dir ^<path^>           Directory to clone into. Default: .\CrowdGPT
echo   --hip-override ^<ver^>   Force HSA_OVERRIDE_GFX_VERSION value (e.g. 11.0.0)
echo   --no-hip-override       Skip HSA_OVERRIDE_GFX_VERSION entirely
echo   --debug                 Enable HIP_LAUNCH_BLOCKING for verbose errors
echo   --no-launch            Install only, don't launch client.py
echo   --no-integrate         Skip Start Menu and PATH integration
echo   --no-fresh             Keep existing install (default: wipe and reinstall)
echo   --help                 Show this message
echo.
exit /b 0

:args_done

echo.
echo ============================================================
echo                 CrowdGPT Installer  (v9.6)
echo ============================================================
echo.

REM ------------------------------------------------------------
REM 0. Paths
REM ------------------------------------------------------------
set "APP_DIR=%LOCALAPPDATA%\CrowdGPT"
set "BIN_DIR=%APP_DIR%"
set "START_MENU=%APPDATA%\Microsoft\Windows\Start Menu\Programs"

if "!INTEGRATE!"=="1" (
    set "VENV_DIR=!APP_DIR!\.venv"
) else (
    set "VENV_DIR=%CD%\.venv"
)
set "APP_PY=!VENV_DIR!\Scripts\python.exe"

REM ------------------------------------------------------------
REM 0b. Wipe previous install (preserve caches)
REM ------------------------------------------------------------
if "!INTEGRATE!"=="1" if "!FRESH!"=="1" if exist "!APP_DIR!" (
    echo [INFO] Removing previous installation at !APP_DIR!...
    echo        ^(caches in checkpoints\ and pending_uploads\ are preserved^)

    set "KEEP_DIR=%TEMP%\crowdgpt-keep-%RANDOM%"
    if not exist "!KEEP_DIR!" mkdir "!KEEP_DIR!" >nul 2>nul

    if exist "!APP_DIR!\checkpoints" (
        robocopy "!APP_DIR!\checkpoints" "!KEEP_DIR!\checkpoints" /E /NFL /NDL /NJH /NJS /NC /NS /NP >nul
    )
    if exist "!APP_DIR!\pending_uploads" (
        robocopy "!APP_DIR!\pending_uploads" "!KEEP_DIR!\pending_uploads" /E /NFL /NDL /NJH /NJS /NC /NS /NP >nul
    )

    rmdir /s /q "!APP_DIR!" >nul 2>nul
    if exist "!APP_DIR!" (
        echo [WARN] Could not fully remove !APP_DIR! ^(files may be locked^).
        echo        Close any running CrowdGPT windows and rerun for a clean install.
    )

    mkdir "!APP_DIR!" >nul 2>nul
    if exist "!KEEP_DIR!\checkpoints" (
        robocopy "!KEEP_DIR!\checkpoints" "!APP_DIR!\checkpoints" /E /NFL /NDL /NJH /NJS /NC /NS /NP >nul
    )
    if exist "!KEEP_DIR!\pending_uploads" (
        robocopy "!KEEP_DIR!\pending_uploads" "!APP_DIR!\pending_uploads" /E /NFL /NDL /NJH /NJS /NC /NS /NP >nul
    )
    rmdir /s /q "!KEEP_DIR!" >nul 2>nul
    echo [OK] Previous install cleared.
    echo.
)

REM ------------------------------------------------------------
REM 1. Install uv
REM ------------------------------------------------------------
where uv >nul 2>nul
if errorlevel 1 (
    echo [INFO] Installing uv...
    where winget >nul 2>nul
    if not errorlevel 1 (
        winget install --id=astral-sh.uv -e --source winget --accept-source-agreements --accept-package-agreements
    ) else (
        echo [INFO] Installing uv via official installer...
        powershell -NoProfile -ExecutionPolicy Bypass -Command "irm https://astral.sh/uv/install.ps1 | iex"
    )
    set "PATH=%USERPROFILE%\.local\bin;%LOCALAPPDATA%\uv;%PATH%"
)

where uv >nul 2>nul
if errorlevel 1 (
    echo [ERROR] uv installed but not on PATH. Restart terminal and rerun.
    goto :fatal
)
echo [OK] uv is ready.

REM ------------------------------------------------------------
REM 2. Locate / install Git
REM ------------------------------------------------------------
where git >nul 2>nul
if errorlevel 1 (
    echo [INFO] Git not found. Installing Git for Windows...
    where winget >nul 2>nul
    if not errorlevel 1 (
        winget install --id Git.Git -e --source winget --accept-source-agreements --accept-package-agreements
        if errorlevel 1 goto :git_fallback
    ) else (
        goto :git_fallback
    )
    set "PATH=%ProgramFiles%\Git\cmd;%ProgramFiles%\Git\bin;%PATH%"
)

where git >nul 2>nul
if errorlevel 1 goto :git_fallback
goto :git_found

:git_fallback
echo [INFO] winget unavailable/failed. Downloading Git for Windows directly...
set "GIT_INSTALLER=%TEMP%\Git-64-bit.exe"
powershell -NoProfile -ExecutionPolicy Bypass -Command "$u='https://github.com/git-for-windows/git/releases/latest/download/Git-64-bit.exe'; Invoke-WebRequest -Uri $u -OutFile '%GIT_INSTALLER%'"
if errorlevel 1 goto :fatal

echo [INFO] Installing Git silently...
"%GIT_INSTALLER%" /VERYSILENT /NORESTART
if errorlevel 1 goto :fatal
del /q "%GIT_INSTALLER%" >nul 2>nul
set "PATH=%ProgramFiles%\Git\cmd;%ProgramFiles%\Git\bin;%PATH%"

where git >nul 2>nul
if errorlevel 1 goto :fatal

:git_found
echo [OK] Git found.

REM ------------------------------------------------------------
REM 3. Clone / update repo
REM ------------------------------------------------------------
if exist ".git" if exist "client.py" (
    set "REPO_DIR=."
    goto :repo_ready
)

if exist "%REPO_DIR%\.git" (
    echo [INFO] CrowdGPT already exists. Updating...
    git -C "%REPO_DIR%" pull --ff-only
    if errorlevel 1 echo [WARN] git pull failed; continuing with existing checkout.
    goto :repo_ready
)

if exist "%REPO_DIR%" (
    echo [ERROR] "%REPO_DIR%" exists but is not a Git repository.
    echo         Move/delete it and rerun.
    goto :fatal
)

echo [INFO] Cloning CrowdGPT...
git clone "%REPO_URL%" "%REPO_DIR%"
if errorlevel 1 goto :fatal

:repo_ready
cd /d "%REPO_DIR%"
if not exist "client.py" (
    echo [ERROR] client.py not found in the CrowdGPT repository.
    goto :fatal
)
set "SRC_DIR=%CD%"

REM ------------------------------------------------------------
REM 4. Hardware detection
REM ------------------------------------------------------------
echo.
echo [INFO] Detecting hardware backend...

call :detect_backend
if errorlevel 1 goto :fatal

if not "!BACKEND_OVERRIDE!"=="" call :apply_override
if errorlevel 1 goto :fatal

REM Compute HSA_OVERRIDE_GFX_VERSION for AMD ROCm
if /I "!BACKEND!"=="amd-rocm" call :compute_hip_override

REM Compute Python version based on backend.
REM AMD's Windows ROCm torch wheels are published for cp312 only.
if /I "!BACKEND!"=="amd-rocm" (
    set "PY_VER=3.12"
) else (
    set "PY_VER=3.11"
)

echo.
echo ============================================================
echo                       Backend
echo ============================================================
echo   Detected:    !BACKEND!
if defined TORCH_INDEX echo   Torch index: !TORCH_INDEX!
echo   Note:        !WHEEL_NOTE!
if /I "!BACKEND!"=="amd-rocm" (
    if defined HIP_GFX_OVERRIDE (
        echo   HIP override: HSA_OVERRIDE_GFX_VERSION=!HIP_GFX_OVERRIDE!
    ) else (
        echo   HIP override: none
    )
)
echo   Python:      !PY_VER!
echo ============================================================
echo.

if /I "!BACKEND!"=="amd" call :note_amd
if /I "!BACKEND!"=="amd-rocm" call :note_amd_rocm

REM ------------------------------------------------------------
REM 5. Create venv AT ITS FINAL LOCATION
REM ------------------------------------------------------------
if "!INTEGRATE!"=="1" (
    if not exist "!APP_DIR!" mkdir "!APP_DIR!"
)

echo [INFO] Creating venv with Python !PY_VER! at !VENV_DIR!...
if exist "!VENV_DIR!" rmdir /s /q "!VENV_DIR!" >nul 2>nul

uv venv --python !PY_VER! --seed "!VENV_DIR!"
if errorlevel 1 (
    echo [ERROR] Failed to create venv with uv.
    goto :fatal
)

if not exist "!APP_PY!" (
    echo [ERROR] venv Python missing at !APP_PY!
    goto :fatal
)

"!APP_PY!" -m pip --version >nul 2>nul
if errorlevel 1 (
    echo [WARN] pip missing from venv, bootstrapping with ensurepip...
    "!APP_PY!" -m ensurepip --upgrade >nul 2>nul
    "!APP_PY!" -m pip --version >nul 2>nul
    if errorlevel 1 (
        echo [ERROR] Could not bootstrap pip into the venv.
        goto :fatal
    )
)

echo [INFO] Upgrading pip...
"!APP_PY!" -m pip install --upgrade pip >nul 2>nul

REM ------------------------------------------------------------
REM 6. Install PyTorch
REM ------------------------------------------------------------
echo.
echo ============================================================
echo                   Installing PyTorch
echo ============================================================

"!APP_PY!" -m pip uninstall -y torch torchvision torchaudio torch-directml >nul 2>nul

if /I "!BACKEND!"=="nvidia"   goto :pytorch_nvidia
if /I "!BACKEND!"=="amd"      goto :pytorch_amd
if /I "!BACKEND!"=="amd-rocm" goto :pytorch_amd_rocm
if /I "!BACKEND!"=="intel"    goto :pytorch_intel
goto :pytorch_cpu

:pytorch_nvidia
echo [INFO] Trying PyTorch CUDA 12.1 cu121...
"!APP_PY!" -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
if not errorlevel 1 goto :pytorch_done
echo.
echo [WARN] cu121 failed. Trying CUDA 11.8 cu118...
"!APP_PY!" -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
if not errorlevel 1 goto :pytorch_done
goto :pytorch_cpu

:pytorch_amd
echo [INFO] Trying PyTorch DirectML...
"!APP_PY!" -m pip install torch-directml
if not errorlevel 1 goto :pytorch_done
echo.
echo [WARN] torch-directml install failed. Falling back to CPU.
goto :pytorch_cpu

:pytorch_amd_rocm
echo [INFO] Trying official AMD ROCm PyTorch for Windows...
echo        Requires Radeon RX 9000 series, RX 7000 series,
echo        or Ryzen AI Max/Max+ Strix Halo.
echo.

call :check_hip_sdk
if "!HIP_SDK_FOUND!"=="1" (
    echo [OK] AMD HIP SDK detected.
) else (
    echo [WARN] AMD HIP SDK for Windows was not detected.
    echo        The ROCm PyTorch wheel will install, but
    echo        torch.cuda.is_available^(^) will return False until the
    echo        HIP SDK is present. GPU acceleration will not work.
    echo.
    echo        HIP SDK download page:
    echo          https://www.amd.com/en/developer/resources/rocm-hub/hip-sdk.html
    echo.
    echo        Do you want to open the download page in your browser?
    echo          [Y] Yes  [N] No, continue anyway
    choice /C YN /N /M "  Choose: "
    if errorlevel 2 (
        echo [INFO] Continuing without HIP SDK. ROCm will likely fall back to CPU.
    ) else (
        start "" "https://www.amd.com/en/developer/resources/rocm-hub/hip-sdk.html"
        echo.
        echo [INFO] Browser opened. Install HIP SDK, reboot, then rerun this installer.
        echo        Press any key to continue the install now, or Ctrl+C to abort.
        pause >nul
    )
    echo.
)
echo.

REM pip 25+ asserts on hashless direct URLs. repo.radeon.com serves
REM index pages without #sha256 fragments, so we pin pip below 25
REM for this install. Scoped to the ROCm path only — PyPI wheels
REM install fine on pip 26.
echo [INFO] Pinning pip below 25 (AMD repo wheels lack hash fragments)...
"!APP_PY!" -m pip install --quiet --disable-pip-version-check "pip<25"
if errorlevel 1 (
    echo [WARN] Could not downgrade pip. ROCm install may fail with an assert.
)
echo.

"!APP_PY!" -m pip install torch torchvision torchaudio --index-url https://repo.radeon.com/rocm/windows/rocm-rel-6.4.4/
if not errorlevel 1 goto :pytorch_done
echo.
echo [WARN] ROCm PyTorch install failed. Falling back to CPU PyTorch.
echo        If the error mentioned 'assert link.hash', pip was still
echo        >= 25. Try:  .venv\Scripts\pip install "pip<25"
echo        For GPU acceleration, install the AMD HIP SDK for Windows
echo        and rerun. For DirectML instead, try:  install.bat --backend amd
goto :pytorch_cpu

:pytorch_intel
echo [INFO] Trying PyTorch XPU...
"!APP_PY!" -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/xpu --extra-index-url https://pypi.org/simple
if not errorlevel 1 goto :pytorch_done
goto :pytorch_cpu

:pytorch_cpu
set "BACKEND=cpu"
set "TORCH_INDEX=https://download.pytorch.org/whl/cpu"
set "WHEEL_NOTE=CPU-only fallback"
set "HIP_GFX_OVERRIDE="
echo.
echo [WARN] Installing CPU-only PyTorch. Training will be very slow.
"!APP_PY!" -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
if errorlevel 1 goto :fatal

:pytorch_done

REM ------------------------------------------------------------
REM 7. Install project requirements (torch lines filtered out)
REM ------------------------------------------------------------
if not exist "requirements.txt" (
    echo [WARN] requirements.txt not found; skipping project deps.
    goto :reqs_done
)

echo.
echo ============================================================
echo               Installing project dependencies
echo ============================================================

set "TMP_REQ=%TEMP%\crowdgpt-req-%RANDOM%.txt"
set "FILTER_PS=%TEMP%\crowdgpt-filter-%RANDOM%.ps1"

> "!FILTER_PS!"  echo $in  = '!SRC_DIR!\requirements.txt'
>> "!FILTER_PS!" echo $out = '!TMP_REQ!'
>> "!FILTER_PS!" echo $rx  = '^\s*(torch^|torchvision^|torchaudio^|torch-directml)\b'
>> "!FILTER_PS!" echo Get-Content -LiteralPath $in ^| Where-Object { $_ -notmatch $rx } ^| Set-Content -LiteralPath $out -Encoding ASCII

powershell -NoProfile -ExecutionPolicy Bypass -File "!FILTER_PS!"
del "!FILTER_PS!" >nul 2>nul

if not exist "!TMP_REQ!" (
    echo [WARN] No filtered requirements produced.
    goto :reqs_done
)

for %%A in ("!TMP_REQ!") do set "TMP_SIZE=%%~zA"
if "!TMP_SIZE!"=="0" (
    echo [WARN] requirements.txt contained only torch packages; nothing else to install.
    del "!TMP_REQ!" >nul 2>nul
    goto :reqs_done
)

"!APP_PY!" -m pip install -r "!TMP_REQ!"
if errorlevel 1 (
    del "!TMP_REQ!" >nul 2>nul
    goto :fatal
)
del "!TMP_REQ!" >nul 2>nul

:reqs_done

REM ------------------------------------------------------------
REM 7b. AMD DirectML: pin transformers to 4.x
REM ------------------------------------------------------------
if /I "!BACKEND!"=="amd" (
    echo.
    echo [INFO] AMD DirectML: pinning transformers to 4.x ^(torch 2.4.1 ceiling^)...
    "!APP_PY!" -m pip install --upgrade "transformers>=4.45,<5.0"
    if errorlevel 1 (
        echo [ERROR] Failed to pin transformers to 4.x for AMD DirectML.
        goto :fatal
    )
)

REM ------------------------------------------------------------
REM 8. GUI backend check - WebView2
REM ------------------------------------------------------------
echo.
echo ============================================================
echo           GUI backend for pywebview - WebView2
echo ============================================================

set "WV2_FOUND=0"
reg query "HKLM\SOFTWARE\WOW6432Node\Microsoft\EdgeUpdate\Clients\{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}" >nul 2>nul
if not errorlevel 1 set "WV2_FOUND=1"
if "!WV2_FOUND!"=="0" (
    reg query "HKLM\SOFTWARE\Microsoft\EdgeUpdate\Clients\{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}" >nul 2>nul
    if not errorlevel 1 set "WV2_FOUND=1"
)
if "!WV2_FOUND!"=="1" (
    echo [OK] WebView2 Runtime present.
) else (
    echo [WARN] Microsoft Edge WebView2 Runtime not detected.
    echo        pywebview needs it to display the GUI window.
    echo        Windows 11 and recent Windows 10 already include it.
    echo        If the client fails to open a window, install it from:
    echo          https://developer.microsoft.com/microsoft-edge/webview2/
    echo.
)

REM ------------------------------------------------------------
REM 9. Verify
REM ------------------------------------------------------------
echo.
echo ============================================================
echo                    Verifying install
echo ============================================================
"!APP_PY!" -c "import sys, torch; print('  Python:', sys.version.split()[0]); print('  Torch:  ', torch.__version__); print('  CUDA:   ', torch.cuda.is_available(), ('' if not torch.cuda.is_available() else '(' + torch.cuda.get_device_name(0) + ')'))"
if errorlevel 1 echo [WARN] Torch import failed.

"!APP_PY!" -c "import transformers, torch; assert transformers.utils.is_torch_available(), 'transformers refuses to use installed torch (version mismatch)'; print('  transformers:', transformers.__version__, '-> torch', torch.__version__, 'linked OK')"
if errorlevel 1 (
    echo.
    echo [ERROR] transformers cannot use the installed torch.
    echo         This WILL break model loading at runtime.
    if /I "!BACKEND!"=="amd" (
        echo         Expected fix: transformers should have been pinned to 4.x.
        echo         Try manually:
        echo           pip install "transformers>=4.45,<5.0"
    ) else (
        echo         Try reinstalling transformers:
        echo           pip install --force-reinstall transformers
    )
    goto :fatal
)

if /I "!BACKEND!"=="amd" call :verify_directml_devices

REM ROCm device check + REAL KERNEL TEST
if /I "!BACKEND!"=="amd-rocm" (
    echo [INFO] Running a real ROCm matmul to check kernel compatibility...
    if defined HIP_GFX_OVERRIDE (
        set "HSA_OVERRIDE_GFX_VERSION=!HIP_GFX_OVERRIDE!"
    )
    "!APP_PY!" -c "import torch; assert torch.cuda.is_available(), 'torch.cuda.is_available() is False'; print('  ROCm device:', torch.cuda.get_device_name(0)); a = torch.randn(64, 64, device='cuda'); b = torch.randn(64, 64, device='cuda'); c = (a @ b).sum().item(); print('  matmul OK (sum=' + str(c) + ')')"
    if errorlevel 1 (
        echo.
        echo [ERROR] ROCm PyTorch is installed but cannot execute a kernel.
        if "!HIP_SDK_FOUND!"=="0" (
            echo         The AMD HIP SDK for Windows is missing.
            echo         Install it from AMD, reboot, then rerun:
            echo           https://www.amd.com/en/developer/resources/rocm-hub/hip-sdk.html
        ) else (
            echo         HIP SDK is present but the wheel has no kernels for
            echo         your GPU architecture.
            if defined HIP_GFX_OVERRIDE (
                echo         Current override: HSA_OVERRIDE_GFX_VERSION=!HIP_GFX_OVERRIDE!
                echo         Try a different value, e.g.:
                echo           install.bat --backend amd-rocm --hip-override 11.0.1
                echo         or
                echo           install.bat --backend amd-rocm --hip-override 11.0.2
            ) else (
                echo         HSA_OVERRIDE_GFX_VERSION is not set.
                echo         Try forcing a value:
                echo           install.bat --backend amd-rocm --hip-override 11.0.0
            )
        )
        goto :fatal
    )
    echo [OK] ROCm kernel test passed.
)

"!APP_PY!" -c "import webview; print('  pywebview: OK')" 2>nul
if errorlevel 1 echo [WARN] pywebview import failed.

REM ------------------------------------------------------------
REM 10. Desktop integration
REM ------------------------------------------------------------
if "!INTEGRATE!"=="1" (
    call :do_integration
    if errorlevel 1 goto :fatal
)

REM ------------------------------------------------------------
REM 11. Launch
REM ------------------------------------------------------------
if "!LAUNCH!"=="1" (
    echo Launching client.py...
    echo.
    if defined HIP_GFX_OVERRIDE (
        set "HSA_OVERRIDE_GFX_VERSION=!HIP_GFX_OVERRIDE!"
    )
    if "!HIP_LAUNCH_BLOCKING!"=="1" set "HIP_LAUNCH_BLOCKING=1"
    if "!INTEGRATE!"=="1" (
        "!APP_PY!" "!APP_DIR!\client.py"
    ) else (
        "!APP_PY!" "!SRC_DIR!\client.py"
    )
    set "EXITCODE=!ERRORLEVEL!"
    echo.
    echo CrowdGPT exited with code !EXITCODE!.
    exit /b !EXITCODE!
) else (
    if "!INTEGRATE!"=="1" (
        echo To launch later:  crowdgpt
    ) else (
        echo To launch later:  "!APP_PY!" "!SRC_DIR!\client.py"
    )
    exit /b 0
)

REM ============================================================
REM Subroutine: compute_hip_override
REM ============================================================
:compute_hip_override
if "!HIP_OVERRIDE_OPT_OUT!"=="1" (
    set "HIP_GFX_OVERRIDE="
    exit /b 0
)
if defined HIP_GFX_OVERRIDE (
    echo [INFO] Using user-supplied HSA_OVERRIDE_GFX_VERSION=!HIP_GFX_OVERRIDE!
    exit /b 0
)

set "HIP_GFX_OVERRIDE="
echo !AMD_GPU_NAME! | findstr /I /C:"RX 7800" >nul && set "HIP_GFX_OVERRIDE=11.0.0"
echo !AMD_GPU_NAME! | findstr /I /C:"RX 7700" >nul && set "HIP_GFX_OVERRIDE=11.0.0"
echo !AMD_GPU_NAME! | findstr /I /C:"RX 7650" >nul && set "HIP_GFX_OVERRIDE=11.0.0"
echo !AMD_GPU_NAME! | findstr /I /C:"RX 7600" >nul && set "HIP_GFX_OVERRIDE=11.0.0"

if defined HIP_GFX_OVERRIDE (
    echo [INFO] Auto-selected HSA_OVERRIDE_GFX_VERSION=!HIP_GFX_OVERRIDE! for !AMD_GPU_NAME!
    echo        ^(RDNA3 non-flagship cards need this to hit gfx1100 kernels^)
) else (
    echo [INFO] No HSA_OVERRIDE_GFX_VERSION needed for !AMD_GPU_NAME!
)
exit /b 0

REM ============================================================
REM Subroutine: check_hip_sdk
REM ============================================================
:check_hip_sdk
set "HIP_SDK_FOUND=0"
if defined HIP_PATH if exist "%HIP_PATH%" set "HIP_SDK_FOUND=1"
if "!HIP_SDK_FOUND!"=="0" if exist "%ProgramFiles%\AMD\ROCm" set "HIP_SDK_FOUND=1"
if "!HIP_SDK_FOUND!"=="0" if exist "%ProgramFiles%\AMD\ROCm\6.4" set "HIP_SDK_FOUND=1"
if "!HIP_SDK_FOUND!"=="0" if exist "%ProgramFiles%\AMD\ROCm\6.3" set "HIP_SDK_FOUND=1"
if "!HIP_SDK_FOUND!"=="0" if exist "%ProgramFiles%\AMD\ROCm\6.2" set "HIP_SDK_FOUND=1"
exit /b 0

REM ============================================================
REM Subroutine: note_amd
REM ============================================================
:note_amd
echo [NOTE] AMD DirectML backend selected.
echo        torch-directml is pinned by upstream to torch 2.4.1,
echo        so transformers will be pinned to 4.x for this install.
echo        Some model operations may not be implemented on DirectML.
echo.
echo [NOTE] If your system has both an iGPU and a dGPU, the client
echo        may pick the wrong DirectML device. To list devices:
echo          .venv\Scripts\python.exe -c "import torch_directml as d; print([d.device_name(i) for i in range(8)])"
echo        Then set CROWDGPT_DML_DEVICE to the desired index before
echo        launching, e.g.:  set CROWDGPT_DML_DEVICE=1
echo.
exit /b 0

REM ============================================================
REM Subroutine: note_amd_rocm
REM ============================================================
:note_amd_rocm
echo [NOTE] AMD ROCm backend selected.
echo        This requires the AMD HIP SDK for Windows to be installed.
if defined HIP_GFX_OVERRIDE (
    echo        HSA_OVERRIDE_GFX_VERSION=!HIP_GFX_OVERRIDE! will be baked into the launcher.
    echo        If you still see "invalid device function", try another value:
    echo          install.bat --backend amd-rocm --hip-override 11.0.1
)
echo.
exit /b 0

REM ============================================================
REM Subroutine: verify_directml_devices
REM ============================================================
:verify_directml_devices
"!APP_PY!" -c "import torch_directml as d; names = [d.device_name(i) for i in range(8) if d.device_name(i)]; [print('  DirectML[%d]: %s' % (i, n)) for i, n in enumerate(names)]" 2>nul
if errorlevel 1 (
    echo [WARN] Could not enumerate DirectML devices.
    exit /b 0
)
"!APP_PY!" -c "import torch_directml as d; assert d.is_available(), 'no DirectML device'; print('  DirectML default: [0]', d.device_name(0))" 2>nul
if errorlevel 1 echo [WARN] torch-directml imported but no DirectML device is available.
exit /b 0

REM ============================================================
REM Subroutine: do_integration
REM ============================================================
:do_integration
echo.
echo ============================================================
echo                Installing desktop integration
echo ============================================================
echo [INFO] Install location: !APP_DIR!

if not exist "!APP_DIR!" mkdir "!APP_DIR!"

echo [INFO] Copying app files (venv excluded; already in place)...
robocopy "!SRC_DIR!" "!APP_DIR!" /E /XD .git .venv __pycache__ checkpoints pending_uploads /NFL /NDL /NJH /NJS /NC /NS /NP >nul
if errorlevel 8 (
    echo [ERROR] Failed to copy files to !APP_DIR!
    exit /b 1
)

if not exist "!APP_PY!" (
    echo [ERROR] Installed venv Python missing at !APP_PY!
    exit /b 1
)

REM ---- CMD shim ----
REM NOTE: the outer if must be wrapped in parens. CMD parses the
REM block eagerly, so `if A if B ( ... !VAR! ... )` trips the
REM delayed-expansion parser even when A is false.
> "!BIN_DIR!\crowdgpt.cmd" echo @echo off
if /I "!BACKEND!"=="amd-rocm" (
    if defined HIP_GFX_OVERRIDE (
        >>"!BIN_DIR!\crowdgpt.cmd" echo set "HSA_OVERRIDE_GFX_VERSION=!HIP_GFX_OVERRIDE!"
    )
)
if "!HIP_LAUNCH_BLOCKING!"=="1" (
    >>"!BIN_DIR!\crowdgpt.cmd" echo set "HIP_LAUNCH_BLOCKING=1"
)
>>"!BIN_DIR!\crowdgpt.cmd" echo "!APP_PY!" "!APP_DIR!\client.py" %%*

REM ---- GUI launcher (no console window) ----
set "GUI_VBS=!BIN_DIR!\crowdgpt-gui.vbs"
> "!GUI_VBS!"  echo Set WshShell = CreateObject("WScript.Shell")
>> "!GUI_VBS!" echo Set WshEnv = WshShell.Environment("Process")
if /I "!BACKEND!"=="amd-rocm" (
    if defined HIP_GFX_OVERRIDE (
        >>"!GUI_VBS!" echo WshEnv("HSA_OVERRIDE_GFX_VERSION") = "!HIP_GFX_OVERRIDE!"
    )
)
if "!HIP_LAUNCH_BLOCKING!"=="1" (
    >>"!GUI_VBS!" echo WshEnv("HIP_LAUNCH_BLOCKING") = "1"
)
>> "!GUI_VBS!" echo WshShell.Run """!APP_PY!"" ""!APP_DIR!\client.py""", 0, False

REM ---- Add to user PATH ----
set "ENV_PS=%TEMP%\crowdgpt-env-%RANDOM%.ps1"
> "!ENV_PS!"  echo $p = [Environment]::GetEnvironmentVariable('Path', 'User')
>> "!ENV_PS!" echo if ($null -eq $p) { $p = '' }
>> "!ENV_PS!" echo if ($p -notlike '*!BIN_DIR!*') {
>> "!ENV_PS!" echo   if ($p) { $new = $p + ';!BIN_DIR!' } else { $new = '!BIN_DIR!' }
>> "!ENV_PS!" echo   [Environment]::SetEnvironmentVariable('Path', $new, 'User')
>> "!ENV_PS!" echo   Write-Host 'Added to user PATH'
>> "!ENV_PS!" echo }
powershell -NoProfile -ExecutionPolicy Bypass -File "!ENV_PS!"
del "!ENV_PS!" >nul 2>nul

REM ---- Icon ----
set "APP_ICON="
if exist "!APP_DIR!\docs\logo-app.ico" set "APP_ICON=!APP_DIR!\docs\logo-app.ico"
if not defined APP_ICON if exist "!APP_DIR!\docs\logo-app.svg" set "APP_ICON=!APP_DIR!\docs\logo-app.svg"

REM ---- Start Menu shortcut ----
set "SC_PS=%TEMP%\crowdgpt-sc-%RANDOM%.ps1"
> "!SC_PS!"  echo $ws = New-Object -ComObject WScript.Shell
>> "!SC_PS!" echo $sc = $ws.CreateShortcut('!START_MENU!\CrowdGPT.lnk')
>> "!SC_PS!" echo $sc.TargetPath = '!GUI_VBS!'
>> "!SC_PS!" echo $sc.WorkingDirectory = '!APP_DIR!'
if defined APP_ICON >>"!SC_PS!" echo $sc.IconLocation = '!APP_ICON!'
>> "!SC_PS!" echo $sc.Description = 'Decentralized AI Training Client'
>> "!SC_PS!" echo $sc.Save()
powershell -NoProfile -ExecutionPolicy Bypass -File "!SC_PS!"
del "!SC_PS!" >nul 2>nul

REM ---- Add/Remove Programs entry ----
set "REG_PS=%TEMP%\crowdgpt-reg-%RANDOM%.ps1"
> "!REG_PS!"  echo $k = 'HKCU:\Software\Microsoft\Windows\CurrentVersion\Uninstall\CrowdGPT'
>> "!REG_PS!" echo New-Item -Path $k -Force ^| Out-Null
>> "!REG_PS!" echo Set-ItemProperty -Path $k -Name 'DisplayName'     -Value 'CrowdGPT'
>> "!REG_PS!" echo Set-ItemProperty -Path $k -Name 'DisplayVersion'  -Value '0.9.6'
>> "!REG_PS!" echo Set-ItemProperty -Path $k -Name 'Publisher'       -Value 'CrowdGPT Project'
>> "!REG_PS!" echo Set-ItemProperty -Path $k -Name 'InstallLocation' -Value '!APP_DIR!'
>> "!REG_PS!" echo Set-ItemProperty -Path $k -Name 'UninstallString' -Value ('"' + '!APP_DIR!\uninstall.bat' + '"')
>> "!REG_PS!" echo Set-ItemProperty -Path $k -Name 'NoModify'        -Value 1 -Type DWord
>> "!REG_PS!" echo Set-ItemProperty -Path $k -Name 'NoRepair'        -Value 1 -Type DWord
powershell -NoProfile -ExecutionPolicy Bypass -File "!REG_PS!"
del "!REG_PS!" >nul 2>nul

REM ---- Uninstaller ----
> "!APP_DIR!\uninstall.bat"  echo @echo off
>> "!APP_DIR!\uninstall.bat" echo setlocal
>> "!APP_DIR!\uninstall.bat" echo echo Uninstalling CrowdGPT...
>> "!APP_DIR!\uninstall.bat" echo del /q "!START_MENU!\CrowdGPT.lnk" 2^>nul
>> "!APP_DIR!\uninstall.bat" echo reg delete "HKCU\Software\Microsoft\Windows\CurrentVersion\Uninstall\CrowdGPT" /f ^>nul 2^>nul
>> "!APP_DIR!\uninstall.bat" echo powershell -NoProfile -Command "$p = [Environment]::GetEnvironmentVariable('Path','User'); if ($p) { $p = ($p -split ';' ^| Where-Object { $_ -ne '!BIN_DIR!' }) -join ';'; [Environment]::SetEnvironmentVariable('Path', $p, 'User') }"
>> "!APP_DIR!\uninstall.bat" echo cd /d "%TEMP%"
>> "!APP_DIR!\uninstall.bat" echo start "" /min cmd /c "timeout /t 2 /nobreak ^>nul ^& rmdir /s /q "!APP_DIR!""
>> "!APP_DIR!\uninstall.bat" echo echo Done. Log out and back in for PATH changes to take effect.
>> "!APP_DIR!\uninstall.bat" echo pause

echo.
echo [OK] Installed command: crowdgpt
echo [OK] Start Menu shortcut: CrowdGPT
echo [OK] Uninstall via: Settings ^> Apps ^> CrowdGPT
echo.
if /I "!BACKEND!"=="amd" (
    echo [NOTE] transformers pinned to 4.x for DirectML compatibility.
)
if /I "!BACKEND!"=="amd-rocm" (
    if defined HIP_GFX_OVERRIDE (
        echo [NOTE] HSA_OVERRIDE_GFX_VERSION=!HIP_GFX_OVERRIDE! baked into launcher.
    )
)
if /I "!BACKEND!"=="cpu" (
    echo [NOTE] CPU-only install. Install the AMD HIP SDK and rerun with
    echo        --backend amd-rocm for GPU, or --backend amd for DirectML.
)
echo [NOTE] A new terminal is required for the "crowdgpt" command to work.
exit /b 0

REM ============================================================
REM Subroutine: apply_override
REM ============================================================
:apply_override
if /I "!BACKEND_OVERRIDE!"=="nvidia" (
    set "BACKEND=nvidia"
    set "TORCH_INDEX=https://download.pytorch.org/whl/cu121"
    set "WHEEL_NOTE=forced: nvidia/cu121"
    exit /b 0
)
if /I "!BACKEND_OVERRIDE!"=="amd" (
    set "BACKEND=amd"
    set "TORCH_INDEX="
    set "WHEEL_NOTE=forced: amd/directml"
    exit /b 0
)
if /I "!BACKEND_OVERRIDE!"=="amd-rocm" (
    set "BACKEND=amd-rocm"
    set "TORCH_INDEX=https://repo.radeon.com/rocm/windows/rocm-rel-6.4.4/"
    set "WHEEL_NOTE=forced: amd/rocm"
    exit /b 0
)
if /I "!BACKEND_OVERRIDE!"=="intel" (
    set "BACKEND=intel"
    set "TORCH_INDEX=https://download.pytorch.org/whl/xpu"
    set "WHEEL_NOTE=forced: intel/xpu"
    exit /b 0
)
if /I "!BACKEND_OVERRIDE!"=="cpu" (
    set "BACKEND=cpu"
    set "TORCH_INDEX=https://download.pytorch.org/whl/cpu"
    set "WHEEL_NOTE=forced: cpu"
    exit /b 0
)
echo [ERROR] --backend must be one of: nvidia, amd, amd-rocm, intel, cpu
exit /b 1

REM ============================================================
REM Subroutine: detect_backend
REM ============================================================
:detect_backend

where nvidia-smi >nul 2>nul
if errorlevel 1 goto :detect_no_nvidia

set "NVIDIA_DRIVER="
for /f "delims=" %%V in ('nvidia-smi --query-gpu^=driver_version --format^=csv,noheader 2^>nul') do (
    if not defined NVIDIA_DRIVER set "NVIDIA_DRIVER=%%V"
)
if defined NVIDIA_DRIVER (
    for /f "tokens=1 delims= " %%A in ("!NVIDIA_DRIVER!") do set "NVIDIA_DRIVER=%%A"
)

if not defined NVIDIA_DRIVER (
    set "BACKEND=nvidia"
    set "TORCH_INDEX=https://download.pytorch.org/whl/cu121"
    set "WHEEL_NOTE=NVIDIA GPU - could not read driver, defaulting to cu121"
    exit /b 0
)

set "DRIVER_MAJOR="
for /f "tokens=1 delims=." %%M in ("!NVIDIA_DRIVER!") do set "DRIVER_MAJOR=%%M"

if not defined DRIVER_MAJOR (
    set "BACKEND=nvidia"
    set "TORCH_INDEX=https://download.pytorch.org/whl/cu121"
    set "WHEEL_NOTE=NVIDIA driver !NVIDIA_DRIVER! - defaulting to cu121"
    exit /b 0
)

if !DRIVER_MAJOR! GEQ 525 (
    set "BACKEND=nvidia"
    set "TORCH_INDEX=https://download.pytorch.org/whl/cu121"
    set "WHEEL_NOTE=NVIDIA driver !NVIDIA_DRIVER! -^> CUDA 12.1 wheel"
    exit /b 0
)
if !DRIVER_MAJOR! GEQ 470 (
    set "BACKEND=nvidia"
    set "TORCH_INDEX=https://download.pytorch.org/whl/cu118"
    set "WHEEL_NOTE=NVIDIA driver !NVIDIA_DRIVER! -^> CUDA 11.8 wheel - older driver"
    exit /b 0
)
echo [ERROR] NVIDIA driver !NVIDIA_DRIVER! is too old for any PyTorch wheel.
echo         Update your GPU driver and rerun.
exit /b 1

:detect_no_nvidia

powershell -NoProfile -Command "try { $g = Get-CimInstance Win32_VideoController | Select-Object -ExpandProperty Name; if ($g -match 'Intel.*Arc') { exit 0 } else { exit 1 } } catch { exit 1 }"
if not errorlevel 1 (
    set "BACKEND=intel"
    set "TORCH_INDEX=https://download.pytorch.org/whl/xpu"
    set "WHEEL_NOTE=Intel Arc detected -^> XPU wheel"
    exit /b 0
)

set "AMD_GPU_NAME="
set "AMD_GPU_TMP=%TEMP%\crowdgpt-amd-%RANDOM%.txt"
powershell -NoProfile -Command "((Get-CimInstance Win32_VideoController | Where-Object { $_.Name -match 'AMD|Radeon' } | Select-Object -ExpandProperty Name) -replace '\s+', ' ') -join ' / '" > "%AMD_GPU_TMP%" 2>nul
if exist "%AMD_GPU_TMP%" (
    for /f "usebackq delims=" %%A in ("%AMD_GPU_TMP%") do (
        if not defined AMD_GPU_NAME set "AMD_GPU_NAME=%%A"
    )
    del "%AMD_GPU_TMP%" >nul 2>nul
)

if not defined AMD_GPU_NAME (
    set "BACKEND=cpu"
    set "TORCH_INDEX=https://download.pytorch.org/whl/cpu"
    set "WHEEL_NOTE=No usable GPU detected -^> CPU wheel - training will be very slow"
    exit /b 0
)

echo !AMD_GPU_NAME! | findstr /I /C:"RX 7900" /C:"RX 7800" /C:"RX 7700" /C:"RX 7650" /C:"RX 7600" /C:"RX 9070" /C:"RX 9060" /C:"RX 9050" /C:"Ryzen AI Max" /C:"Ryzen AI 9" >nul
if not errorlevel 1 (
    set "BACKEND=amd-rocm"
    set "TORCH_INDEX=https://repo.radeon.com/rocm/windows/rocm-rel-6.4.4/"
    set "WHEEL_NOTE=!AMD_GPU_NAME! -^> ROCm 6.4.4 wheel for Windows"
    exit /b 0
)

set "BACKEND=amd"
set "TORCH_INDEX="
set "WHEEL_NOTE=!AMD_GPU_NAME! -^> DirectML (not on AMD's ROCm Windows list)"
exit /b 0

:fatal
echo.
echo ============================================================
echo [ERROR] Installation failed.
echo ============================================================
pause
exit /b 1
