@echo off
setlocal EnableExtensions EnableDelayedExpansion

REM ============================================================
REM CrowdGPT Universal Installer for Windows
REM v9.1
REM  - venv created at FINAL path (no relocation)
REM  - AMD DirectML: transformers pinned to 4.x (torch 2.4.1 ceiling)
REM  - CUDA/ROCm/Intel/CPU: transformers 5.x as normal
REM  - torch <-> transformers compatibility verified BEFORE launch
REM  - uninstaller self-deletes via delayed cmd
REM  - ROCm GPU allow-list matches AMD's real Windows matrix
REM  - v8: fixed unescaped parens in echo inside if() block
REM  - v9: AMD GPU name captured via temp file (v5 approach);
REM        the for/f+inline-PowerShell variant was broken
REM  - v9.1: collapse whitespace in AMD GPU name; drop redundant
REM          "AMD " prefix in WHEEL_NOTE (was showing "AMD AMD")
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
echo   --backend ^<name^>   Force backend: nvidia ^| amd ^| amd-rocm ^| intel ^| cpu
echo   --dir ^<path^>       Directory to clone into. Default: .\CrowdGPT
echo   --no-launch        Install only, don't launch client.py
echo   --no-integrate     Skip Start Menu and PATH integration
echo   --help             Show this message
echo.
exit /b 0

:args_done

echo.
echo ============================================================
echo                 CrowdGPT Installer  (v9.1)
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

echo.
echo ============================================================
echo                       Backend
echo ============================================================
echo   Detected:    !BACKEND!
if defined TORCH_INDEX echo   Torch index: !TORCH_INDEX!
echo   Note:        !WHEEL_NOTE!
echo ============================================================
echo.

if /I "!BACKEND!"=="amd" call :note_amd

REM ------------------------------------------------------------
REM 5. Create venv AT ITS FINAL LOCATION
REM ------------------------------------------------------------
if "!INTEGRATE!"=="1" (
    if not exist "!APP_DIR!" mkdir "!APP_DIR!"
)

echo [INFO] Creating venv with Python 3.11 at !VENV_DIR!...
if exist "!VENV_DIR!" rmdir /s /q "!VENV_DIR!" >nul 2>nul

uv venv --python 3.11 --seed "!VENV_DIR!"
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
echo        Requires Radeon RX 7900 XTX/XT/GRE, RX 9070 XT/9070,
echo        or Ryzen AI Max/Max+ Strix Halo.
"!APP_PY!" -m pip install torch torchvision torchaudio --index-url https://repo.radeon.com/rocm/windows/rocm-rel-6.4.4/
if not errorlevel 1 goto :pytorch_done
echo.
echo [WARN] ROCm PyTorch install failed. Falling back to CPU PyTorch.
echo        Update your AMD GPU driver and rerun for ROCm support.
goto :pytorch_cpu

:pytorch_intel
echo [INFO] Trying PyTorch XPU...
"!APP_PY!" -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/xpu --extra-index-url https://pypi.org/simple
if not errorlevel 1 goto :pytorch_done
goto :pytorch_cpu

:pytorch_cpu
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
REM     torch-directml is upstream-stuck on torch 2.4.1;
REM     transformers 5.x refuses torch < 2.5.
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
> "!BIN_DIR!\crowdgpt.cmd" echo @echo off
>>"!BIN_DIR!\crowdgpt.cmd" echo "!APP_PY!" "!APP_DIR!\client.py" %%*

REM ---- GUI launcher (no console window) ----
set "GUI_VBS=!BIN_DIR!\crowdgpt-gui.vbs"
> "!GUI_VBS!"  echo Set WshShell = CreateObject("WScript.Shell")
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
>> "!REG_PS!" echo Set-ItemProperty -Path $k -Name 'DisplayVersion'  -Value '0.9.1'
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

REM --- AMD: capture GPU name via temp file (v5 approach) ---
REM     Do NOT use for/f with inline PowerShell here: CMD eats the
REM     ^| escapes and the inner cmd.exe sees the pipes, so nothing
REM     is captured. A temp file sidesteps the whole problem.
REM     v9.1: collapse runs of whitespace inside PowerShell so names
REM     like "AMD  Radeon" become "AMD Radeon".
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

REM AMD's real Windows ROCm support matrix (ROCm 6.4):
REM   Radeon RX 7900 XTX / XT / GRE
REM   Radeon RX 9070 XT / 9070
REM   Ryzen AI Max / Max+  (Strix Halo)
echo !AMD_GPU_NAME! | findstr /I /C:"RX 7900" /C:"RX 9070" /C:"Ryzen AI Max" >nul
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
