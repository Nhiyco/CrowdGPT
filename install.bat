@echo off
setlocal EnableExtensions EnableDelayedExpansion

REM ============================================================
REM CrowdGPT Universal Installer for Windows  (v9.7)
REM ------------------------------------------------------------
REM v9.7:
REM   - Fixed delayed-expansion bug in requirements filter
REM     (literal `!` inside regex was eaten by batch, breaking
REM      PowerShell parsing -> "Missing closing '}'")
REM   - Replaced regex class [<>=!~] with \W
REM ------------------------------------------------------------
REM v9.6:
REM   - AMD prioritization: ROCm 6.4.4 for RX 7000/9000 & Ryzen AI,
REM     DirectML (torch 2.4.1) for older AMD GPUs
REM   - Hardcoded PyTorch versions per backend
REM   - transformers pinned <4.49 for DirectML
REM   - Fixed !TMP_REQ! delayed expansion bug
REM   - ROCm wheels installed via direct URLs (not --find-links)
REM ============================================================

set "REPO_URL=https://github.com/Vxtzq/CrowdGPT.git"
set "REPO_DIR=CrowdGPT"
set "BACKEND="
set "TORCH_INDEX="
set "TORCH_EXTRA="
set "WHEEL_NOTE="
set "LAUNCH=1"
set "INTEGRATE=1"
set "BACKEND_OVERRIDE="
set "AMD_GPU_NAME="
set "HIP_SDK_FOUND=0"
set "PY_VERSION=3.11"
set "TRANSFORMERS_PIN="
set "ROCM_BASE=https://repo.radeon.com/rocm/windows/rocm-rel-6.4.4"

REM ============================================================
REM Arg parsing
REM ============================================================
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
echo   --backend ^<name^>   Force: nvidia ^| amd ^| amd-rocm ^| intel ^| cpu
echo   --dir ^<path^>       Directory to clone into. Default: .\CrowdGPT
echo   --no-launch        Install only, don't launch client.py
echo   --no-integrate     Skip Start Menu and PATH integration
echo   --help             Show this message
echo.
exit /b 0

:args_done

echo.
echo ============================================================
echo                CrowdGPT Installer  (v9.7)
echo ============================================================
echo.

REM ============================================================
REM 1. uv
REM ============================================================
where uv >nul 2>nul
if errorlevel 1 (
    echo [INFO] Installing uv...
    where winget >nul 2>nul
    if not errorlevel 1 (
        winget install --id=astral-sh.uv -e --source winget --accept-source-agreements --accept-package-agreements
        if errorlevel 1 goto :fatal
    ) else (
        echo [INFO] Installing uv via official installer...
        powershell -NoProfile -ExecutionPolicy Bypass -Command "irm https://astral.sh/uv/install.ps1 | iex"
        if errorlevel 1 goto :fatal
    )
    set "PATH=%USERPROFILE%\.local\bin;%LOCALAPPDATA%\uv;%PATH%"
)

where uv >nul 2>nul
if errorlevel 1 (
    echo [ERROR] uv installed but not on PATH. Restart terminal and rerun.
    goto :fatal
)
echo [OK] uv is ready.

REM ============================================================
REM 2. Git
REM ============================================================
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

REM ============================================================
REM 3. Clone / update repo
REM ============================================================
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

REM ============================================================
REM 4. Install paths
REM ============================================================
set "APP_DIR=%LOCALAPPDATA%\CrowdGPT"
set "BIN_DIR=%APP_DIR%"
set "VENV_DIR=%APP_DIR%\.venv"
set "APP_PY=%VENV_DIR%\Scripts\python.exe"
set "START_MENU=%APPDATA%\Microsoft\Windows\Start Menu\Programs"

REM ============================================================
REM 5. Hardware detection
REM ============================================================
echo.
echo [INFO] Detecting hardware backend...
call :detect_backend
if errorlevel 1 goto :fatal

if not "!BACKEND_OVERRIDE!"=="" (
    if /I "!BACKEND_OVERRIDE!"=="nvidia" (
        set "BACKEND=nvidia"
        set "TORCH_INDEX=https://download.pytorch.org/whl/cu121"
        set "TORCH_EXTRA="
        set "WHEEL_NOTE=forced: nvidia/cu121"
    ) else if /I "!BACKEND_OVERRIDE!"=="amd" (
        set "BACKEND=amd"
        set "TORCH_INDEX="
        set "TORCH_EXTRA="
        set "WHEEL_NOTE=forced: amd/directml"
    ) else if /I "!BACKEND_OVERRIDE!"=="amd-rocm" (
        set "BACKEND=amd-rocm"
        set "TORCH_INDEX="
        set "TORCH_EXTRA="
        set "WHEEL_NOTE=forced: amd/rocm"
    ) else if /I "!BACKEND_OVERRIDE!"=="intel" (
        set "BACKEND=intel"
        set "TORCH_INDEX=https://download.pytorch.org/whl/xpu"
        set "TORCH_EXTRA=https://pypi.org/simple"
        set "WHEEL_NOTE=forced: intel/xpu"
    ) else if /I "!BACKEND_OVERRIDE!"=="cpu" (
        set "BACKEND=cpu"
        set "TORCH_INDEX=https://download.pytorch.org/whl/cpu"
        set "TORCH_EXTRA="
        set "WHEEL_NOTE=forced: cpu"
    ) else (
        echo [ERROR] --backend must be one of: nvidia ^| amd ^| amd-rocm ^| intel ^| cpu
        goto :fatal
    )
)

echo.
echo ============================================================
echo                       Backend
echo ============================================================
echo   Detected:    !BACKEND!
if defined TORCH_INDEX echo   Torch index: !TORCH_INDEX!
if defined TORCH_EXTRA echo   Extra index: !TORCH_EXTRA!
echo   Note:        !WHEEL_NOTE!
echo ============================================================
echo.

REM ============================================================
REM 6. Clean previous venv, ensure app dir
REM ============================================================
if exist "!VENV_DIR!" (
    echo [INFO] Removing previous venv at !VENV_DIR!...
    rmdir /s /q "!VENV_DIR!" >nul 2>nul
)
if not exist "!APP_DIR!" mkdir "!APP_DIR!"

REM ============================================================
REM 7. Sync app files
REM ============================================================
echo [INFO] Syncing app files to !APP_DIR!...
robocopy "%CD%" "!APP_DIR!" /E /XD .git .venv __pycache__ checkpoints pending_uploads /NFL /NDL /NJH /NJS /NC /NS >nul
if errorlevel 8 (
    echo [ERROR] Failed to copy files to !APP_DIR!
    goto :fatal
)

REM ============================================================
REM 8. Create venv
REM ============================================================
if /I "!BACKEND!"=="amd-rocm" (
    set "PY_VERSION=3.12"
    echo [INFO] Creating venv with Python 3.12 ^(ROCm wheels are cp312^)...
) else (
    set "PY_VERSION=3.11"
    echo [INFO] Creating venv with Python 3.11...
)

uv venv --python !PY_VERSION! --seed "!VENV_DIR!"
if errorlevel 1 (
    echo [ERROR] Failed to create venv with uv.
    goto :fatal
)

if not exist "!APP_PY!" (
    echo [ERROR] Venv Python missing at !APP_PY!
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

REM ============================================================
REM 9. Install PyTorch  (per-backend hardcoded versions)
REM ============================================================
echo.
echo ============================================================
echo                   Installing PyTorch
echo ============================================================

"!APP_PY!" -m pip uninstall -y torch torchvision torchaudio torch-directml >nul 2>nul

if /I "!BACKEND!"=="nvidia" goto :pytorch_nvidia
if /I "!BACKEND!"=="amd" goto :pytorch_amd
if /I "!BACKEND!"=="amd-rocm" goto :pytorch_amd_rocm
if /I "!BACKEND!"=="intel" goto :pytorch_intel
goto :pytorch_cpu

REM ------------------------------------------------------------
REM NVIDIA: latest CUDA 12.1, fallback to CUDA 11.8
REM ------------------------------------------------------------
:pytorch_nvidia
echo [INFO] Trying PyTorch CUDA 12.1 cu121...
"!APP_PY!" -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
if not errorlevel 1 goto :pytorch_done
echo.
echo [WARN] cu121 failed. Trying CUDA 11.8 cu118...
"!APP_PY!" -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
if not errorlevel 1 goto :pytorch_done
goto :pytorch_cpu

REM ------------------------------------------------------------
REM AMD DirectML: hardcoded torch 2.4.1 + transformers <4.49
REM ------------------------------------------------------------
:pytorch_amd
echo [INFO] Trying PyTorch DirectML ^(torch 2.4.1^)...
"!APP_PY!" -m pip install torch-directml torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1
if not errorlevel 1 (
    echo [OK] torch-directml + torch 2.4.1 installed.
    echo [INFO] Pinning transformers to ^<4.49 ^(compatible with torch 2.4.x^)...
    set "TRANSFORMERS_PIN=transformers>=4.40,<4.49"
    set "USE_TORCH=1"
    setx USE_TORCH 1 >nul 2>nul
    goto :pytorch_done
)
echo.
echo [WARN] torch-directml install failed. Falling back to CPU.
goto :pytorch_cpu

REM ------------------------------------------------------------
REM AMD ROCm: official ROCm 6.4.4 wheels (cp312 only)
REM ------------------------------------------------------------
:pytorch_amd_rocm
echo [INFO] Installing official AMD ROCm PyTorch for Windows...
echo        Requires Radeon RX 9000 series, RX 7000 series,
echo        or Ryzen AI Max/Max+ Strix Halo.
echo.

call :check_hip_sdk
if "!HIP_SDK_FOUND!"=="1" (
    echo [OK] AMD HIP SDK detected.
) else (
    echo [WARN] AMD HIP SDK for Windows was not detected.
    echo        The ROCm PyTorch wheel will install, but
    echo        torch.cuda.is_available^(^) will return False until
    echo        the HIP SDK is present. GPU acceleration will not work.
    echo.
    echo        HIP SDK download page:
    echo          https://www.amd.com/en/developer/resources/rocm-hub/hip-sdk.html
    echo.
    echo        Do you want to open the download page in your browser?
    echo          [Y] Yes  [N] No, continue anyway
    choice /C YN /N /M " Choose: "
    if errorlevel 2 (
        echo [INFO] Continuing without HIP SDK.
    ) else (
        start "" "https://www.amd.com/en/developer/resources/rocm-hub/hip-sdk.html"
        echo [INFO] Browser opened. Install HIP SDK, reboot, then rerun.
    )
    echo.
)

echo [INFO] Installing torch, torchvision, torchaudio from ROCm 6.4.4...
"!APP_PY!" -m pip install --no-cache-dir "!ROCM_BASE!/torch-2.8.0a0+gitfc14c65-cp312-cp312-win_amd64.whl" "!ROCM_BASE!/torchvision-0.24.0a0+c85f008-cp312-cp312-win_amd64.whl" "!ROCM_BASE!/torchaudio-2.6.0a0+1a8f621-cp312-cp312-win_amd64.whl"
if not errorlevel 1 (
    echo [OK] ROCm PyTorch installed.
    goto :pytorch_done
)

echo.
echo [WARN] ROCm wheel install failed. Falling back to DirectML...
"!APP_PY!" -m pip install torch-directml torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1
if not errorlevel 1 (
    echo [OK] DirectML fallback installed.
    set "TRANSFORMERS_PIN=transformers>=4.40,<4.49"
    set "USE_TORCH=1"
    setx USE_TORCH 1 >nul 2>nul
    goto :pytorch_done
)
goto :pytorch_cpu

REM ------------------------------------------------------------
REM Intel XPU: latest XPU wheel
REM ------------------------------------------------------------
:pytorch_intel
echo [INFO] Trying PyTorch XPU...
"!APP_PY!" -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/xpu --extra-index-url https://pypi.org/simple
if not errorlevel 1 goto :pytorch_done
goto :pytorch_cpu

REM ------------------------------------------------------------
REM CPU fallback: latest CPU wheel
REM ------------------------------------------------------------
:pytorch_cpu
echo.
echo [WARN] Installing CPU-only PyTorch. Training will be very slow.
"!APP_PY!" -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
if errorlevel 1 goto :fatal

:pytorch_done

REM ============================================================
REM 10. Project requirements  (delayed expansion used)
REM ============================================================
if exist "requirements.txt" (
    echo.
    echo ============================================================
    echo               Installing project dependencies
    echo ============================================================
    set "TMP_REQ=%TEMP%\crowdgpt-req-!RANDOM!-!RANDOM!.txt"
    powershell -NoProfile -ExecutionPolicy Bypass -Command ^
        "Get-Content -LiteralPath 'requirements.txt' | Where-Object { $_ -notmatch '^\s*(torch|torchvision|torchaudio|torch-directml)(\W|$)' } | Set-Content -LiteralPath '!TMP_REQ!'"
    if errorlevel 1 goto :fatal

    for %%A in ("!TMP_REQ!") do set "TMP_SIZE=%%~zA"
    if "!TMP_SIZE!"=="0" (
        echo [WARN] requirements.txt contained only torch packages.
    ) else (
        "!APP_PY!" -m pip install -r "!TMP_REQ!"
        if errorlevel 1 goto :fatal
    )
    del /q "!TMP_REQ!" >nul 2>nul
) else (
    echo [WARN] requirements.txt not found; skipping project deps.
)

REM ---- Pin transformers for DirectML (torch 2.4.x) ----
if defined TRANSFORMERS_PIN (
    echo.
    echo [INFO] Pinning transformers for DirectML: !TRANSFORMERS_PIN!
    "!APP_PY!" -m pip install "!TRANSFORMERS_PIN!"
    if errorlevel 1 (
        echo [WARN] Failed to pin transformers. USE_TORCH=1 is the fallback.
    ) else (
        echo [OK] transformers pinned to a DirectML-compatible version.
    )
)

REM ============================================================
REM 11. WebView2 runtime check
REM ============================================================
echo.
echo ============================================================
echo           GUI backend for pywebview - WebView2
echo ============================================================
reg query "HKLM\SOFTWARE\WOW6432Node\Microsoft\EdgeUpdate\Clients\{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}" >nul 2>nul
if errorlevel 1 (
    reg query "HKLM\SOFTWARE\Microsoft\EdgeUpdate\Clients\{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}" >nul 2>nul
)
if errorlevel 1 (
    echo [WARN] Microsoft Edge WebView2 Runtime not detected.
    echo        Install from:
    echo          https://developer.microsoft.com/microsoft-edge/webview2/
) else (
    echo [OK] WebView2 Runtime present.
)

REM ============================================================
REM 12. Verify install
REM ============================================================
echo.
echo ============================================================
echo                    Verifying install
echo ============================================================
"!APP_PY!" -c "import sys, torch; print('  Python:', sys.version.split()[0]); print('  Torch:  ', torch.__version__); print('  CUDA:   ', torch.cuda.is_available())"
if errorlevel 1 echo [WARN] torch import failed.

if /I "!BACKEND!"=="amd-rocm" (
    "!APP_PY!" -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" >nul 2>nul
    if errorlevel 1 (
        echo [WARN] ROCm wheel installed but torch.cuda.is_available^(^) is False.
        echo        The AMD HIP SDK for Windows is likely missing.
        echo        Download: https://www.amd.com/en/developer/resources/rocm-hub/hip-sdk.html
        echo        Install it, reboot, then rerun with --backend amd-rocm.
    ) else (
        echo [OK] ROCm GPU detected.
    )
)

"!APP_PY!" -c "import webview; print('  pywebview: OK')" 2>nul
if errorlevel 1 echo [WARN] pywebview import failed.

REM ============================================================
REM 13. Desktop integration
REM ============================================================
if "!INTEGRATE!"=="1" (
    call :do_integration
    if errorlevel 1 goto :fatal
)

REM ============================================================
REM 14. Launch
REM ============================================================
if "!LAUNCH!"=="1" (
    echo Launching client.py...
    echo.
    "!APP_PY!" "!APP_DIR!\client.py"
    set "EXITCODE=!ERRORLEVEL!"
    echo.
    echo CrowdGPT exited with code !EXITCODE!.
    exit /b !EXITCODE!
) else (
    if "!INTEGRATE!"=="1" (
        echo To launch later:  crowdgpt
    ) else (
        echo To launch later:  "!APP_PY!" "!APP_DIR!\client.py"
    )
    exit /b 0
)

REM ============================================================
REM Subroutine: do_integration
REM ============================================================
:do_integration
echo.
echo ============================================================
echo                Installing desktop integration
echo ============================================================
echo [INFO] Install location: !APP_DIR!

if not exist "!APP_PY!" (
    echo [ERROR] Venv Python missing at !APP_PY!
    exit /b 1
)

> "!BIN_DIR!\crowdgpt.cmd" echo @echo off
>>"!BIN_DIR!\crowdgpt.cmd" echo "!APP_PY!" "!APP_DIR!\client.py" %%*

> "!BIN_DIR!\crowdgpt-gui.vbs" echo Set WshShell = CreateObject("WScript.Shell")
>>"!BIN_DIR!\crowdgpt-gui.vbs" echo WshShell.Run """!APP_PY!"" ""!APP_DIR!\client.py""", 0, False

> "%TEMP%\crowdgpt_env.ps1" echo $p = [Environment]::GetEnvironmentVariable('Path', 'User')
>> "%TEMP%\crowdgpt_env.ps1" echo if ($null -eq $p) { $p = '' }
>> "%TEMP%\crowdgpt_env.ps1" echo if ($p -notlike '*!BIN_DIR!*') {
>> "%TEMP%\crowdgpt_env.ps1" echo   $new = if ($p) { $p + ';!BIN_DIR!' } else { '!BIN_DIR!' }
>> "%TEMP%\crowdgpt_env.ps1" echo   [Environment]::SetEnvironmentVariable('Path', $new, 'User')
>> "%TEMP%\crowdgpt_env.ps1" echo   Write-Host 'Added to user PATH'
>> "%TEMP%\crowdgpt_env.ps1" echo }
powershell -NoProfile -ExecutionPolicy Bypass -File "%TEMP%\crowdgpt_env.ps1"
del "%TEMP%\crowdgpt_env.ps1" >nul 2>nul

if /I "!BACKEND!"=="amd" (
    setx USE_TORCH 1 >nul 2>nul
    echo [OK] USE_TORCH=1 set persistently.
)

set "APP_ICON="
if exist "!APP_DIR!\docs\logo-app.ico" set "APP_ICON=!APP_DIR!\docs\logo-app.ico"
if not defined APP_ICON if exist "!APP_DIR!\docs\logo-app.svg" set "APP_ICON=!APP_DIR!\docs\logo-app.svg"

> "%TEMP%\crowdgpt_shortcut.ps1" echo $ws = New-Object -ComObject WScript.Shell
>> "%TEMP%\crowdgpt_shortcut.ps1" echo $sc = $ws.CreateShortcut('!START_MENU!\CrowdGPT.lnk')
>> "%TEMP%\crowdgpt_shortcut.ps1" echo $sc.TargetPath = '!APP_DIR!\crowdgpt-gui.vbs'
>> "%TEMP%\crowdgpt_shortcut.ps1" echo $sc.WorkingDirectory = '!APP_DIR!'
>> "%TEMP%\crowdgpt_shortcut.ps1" echo if ('!APP_ICON!') { $sc.IconLocation = '!APP_ICON!' }
>> "%TEMP%\crowdgpt_shortcut.ps1" echo $sc.Description = 'Decentralized AI Training Client'
>> "%TEMP%\crowdgpt_shortcut.ps1" echo $sc.Save()
powershell -NoProfile -ExecutionPolicy Bypass -File "%TEMP%\crowdgpt_shortcut.ps1"
del "%TEMP%\crowdgpt_shortcut.ps1" >nul 2>nul

> "%TEMP%\crowdgpt_reg.ps1" echo $k = 'HKCU:\Software\Microsoft\Windows\CurrentVersion\Uninstall\CrowdGPT'
>> "%TEMP%\crowdgpt_reg.ps1" echo New-Item -Path $k -Force ^| Out-Null
>> "%TEMP%\crowdgpt_reg.ps1" echo Set-ItemProperty -Path $k -Name 'DisplayName' -Value 'CrowdGPT'
>> "%TEMP%\crowdgpt_reg.ps1" echo Set-ItemProperty -Path $k -Name 'DisplayVersion' -Value '0.7'
>> "%TEMP%\crowdgpt_reg.ps1" echo Set-ItemProperty -Path $k -Name 'Publisher' -Value 'CrowdGPT Project'
>> "%TEMP%\crowdgpt_reg.ps1" echo Set-ItemProperty -Path $k -Name 'InstallLocation' -Value '!APP_DIR!'
>> "%TEMP%\crowdgpt_reg.ps1" echo Set-ItemProperty -Path $k -Name 'UninstallString' -Value ('\"' + '!APP_DIR!\uninstall.bat' + '\"')
>> "%TEMP%\crowdgpt_reg.ps1" echo Set-ItemProperty -Path $k -Name 'NoModify' -Value 1 -Type DWord
>> "%TEMP%\crowdgpt_reg.ps1" echo Set-ItemProperty -Path $k -Name 'NoRepair' -Value 1 -Type DWord
powershell -NoProfile -ExecutionPolicy Bypass -File "%TEMP%\crowdgpt_reg.ps1"
del "%TEMP%\crowdgpt_reg.ps1" >nul 2>nul

> "!APP_DIR!\uninstall.bat" echo @echo off
>>"!APP_DIR!\uninstall.bat" echo setlocal
>>"!APP_DIR!\uninstall.bat" echo echo Uninstalling CrowdGPT...
>>"!APP_DIR!\uninstall.bat" echo del /q "!START_MENU!\CrowdGPT.lnk" 2^>nul
>>"!APP_DIR!\uninstall.bat" echo reg delete "HKCU\Software\Microsoft\Windows\CurrentVersion\Uninstall\CrowdGPT" /f ^>nul 2^>nul
>>"!APP_DIR!\uninstall.bat" echo powershell -NoProfile -Command "$p = [Environment]::GetEnvironmentVariable('Path','User'); if ($p) { $p = ($p -split ';' ^| Where-Object { $_ -ne '!BIN_DIR!' }) -join ';'; [Environment]::SetEnvironmentVariable('Path', $p, 'User') }"
>>"!APP_DIR!\uninstall.bat" echo reg delete "HKCU\Environment" /v USE_TORCH /f ^>nul 2^>nul
>>"!APP_DIR!\uninstall.bat" echo cd /d "%%TEMP%%"
>>"!APP_DIR!\uninstall.bat" echo rmdir /s /q "!APP_DIR!"
>>"!APP_DIR!\uninstall.bat" echo echo Done. Log out and back in for PATH changes to take effect.
>>"!APP_DIR!\uninstall.bat" echo pause

echo.
echo [OK] Installed command: crowdgpt
echo [OK] Start Menu shortcut: CrowdGPT
echo [OK] Uninstall via: Settings ^> Apps ^> CrowdGPT
echo.
echo [NOTE] A new terminal is required for the "crowdgpt" command to work.
exit /b 0

REM ============================================================
REM Subroutine: check_hip_sdk
REM ============================================================
:check_hip_sdk
set "HIP_SDK_FOUND=0"
if exist "%ProgramFiles%\AMD\ROCm" (
    for /d %%D in ("%ProgramFiles%\AMD\ROCm\*") do (
        if exist "%%D\bin\hipcc.exe" set "HIP_SDK_FOUND=1"
        if exist "%%D\bin\amdhip64.dll" set "HIP_SDK_FOUND=1"
    )
)
if exist "%ProgramFiles%\AMD\HIP" set "HIP_SDK_FOUND=1"
if exist "%HIP_PATH%\bin\hipcc.exe" set "HIP_SDK_FOUND=1"
exit /b 0

REM ============================================================
REM Subroutine: detect_backend
REM ============================================================
:detect_backend

where nvidia-smi >nul 2>nul
if not errorlevel 1 (
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
        set "WHEEL_NOTE=NVIDIA GPU - defaulting to cu121"
        exit /b 0
    )
    set "DRIVER_MAJOR="
    for /f "tokens=1 delims=." %%M in ("!NVIDIA_DRIVER!") do set "DRIVER_MAJOR=%%M"
    if !DRIVER_MAJOR! GEQ 525 (
        set "BACKEND=nvidia"
        set "TORCH_INDEX=https://download.pytorch.org/whl/cu121"
        set "WHEEL_NOTE=NVIDIA driver !NVIDIA_DRIVER! -^> CUDA 12.1"
    ) else if !DRIVER_MAJOR! GEQ 470 (
        set "BACKEND=nvidia"
        set "TORCH_INDEX=https://download.pytorch.org/whl/cu118"
        set "WHEEL_NOTE=NVIDIA driver !NVIDIA_DRIVER! -^> CUDA 11.8"
    ) else (
        echo [ERROR] NVIDIA driver too old.
        exit /b 1
    )
    exit /b 0
)

powershell -NoProfile -Command "try { $g = Get-CimInstance Win32_VideoController | Select-Object -ExpandProperty Name; if ($g -match 'Intel.*Arc') { exit 0 } else { exit 1 } } catch { exit 1 }"
if not errorlevel 1 (
    set "BACKEND=intel"
    set "TORCH_INDEX=https://download.pytorch.org/whl/xpu"
    set "TORCH_EXTRA=https://pypi.org/simple"
    set "WHEEL_NOTE=Intel Arc detected -^> XPU"
    exit /b 0
)

powershell -NoProfile -Command "try { $g = Get-CimInstance Win32_VideoController | Select-Object -ExpandProperty Name; if ($g -match 'AMD|Radeon') { Write-Output ($g -join ' / ') } else { exit 1 } } catch { exit 1 }" > "%TEMP%\amd_gpu.txt" 2>nul
set /p AMD_GPU_NAME=<"%TEMP%\amd_gpu.txt"
del "%TEMP%\amd_gpu.txt" >nul 2>nul

if defined AMD_GPU_NAME (
    REM Prioritize ROCm for Radeon RX 7000/9000 series and Ryzen AI
    echo !AMD_GPU_NAME! | findstr /I "RX 9 9070 9060 7900 7800 7700 7600 Ryzen AI" >nul
    if not errorlevel 1 (
        set "BACKEND=amd-rocm"
        set "TORCH_INDEX=!ROCM_BASE!/"
        set "TORCH_EXTRA="
        set "WHEEL_NOTE=AMD !AMD_GPU_NAME! -^> ROCm 6.4.4 ^(Windows^)"
        exit /b 0
    )
    REM Older AMD GPUs -> DirectML
    set "BACKEND=amd"
    set "TORCH_INDEX="
    set "TORCH_EXTRA="
    set "WHEEL_NOTE=AMD !AMD_GPU_NAME! -^> DirectML ^(torch 2.4.1^)"
    exit /b 0
)

set "BACKEND=cpu"
set "TORCH_INDEX=https://download.pytorch.org/whl/cpu"
set "TORCH_EXTRA="
set "WHEEL_NOTE=No GPU -^> CPU"
exit /b 0

:fatal
echo.
echo ============================================================
echo [ERROR] Installation failed.
echo ============================================================
pause
exit /b 1
