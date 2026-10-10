#!/usr/bin/env bash
# ============================================================
# CrowdGPT universal installer — Linux + macOS  (v8)
#
#  - venv created at FINAL path (no relocation, no rewrites)
#  - AMD ROCm: torch version detected, transformers pinned
#    to 4.x when torch < 2.5 (rocm6.1 wheels are torch 2.4.1)
#  - NVIDIA/Apple/Intel/CPU: transformers 5.x as normal
#  - verify checks torch <-> transformers linkage before launch
#  - ROCm index chosen from detected ROCm version
#  - Python >= 3.10 enforced
#  - v8: verify step no longer dies from `set -e` before
#        printing the actionable error message
# ============================================================
set -Eeuo pipefail

REPO_URL="https://github.com/Vxtzq/CrowdGPT.git"
REPO_DIR="CrowdGPT"
BACKEND_OVERRIDE=""
LAUNCH=1
INTEGRATE=1

if [ -t 1 ]; then
    BOLD=$'\033[1m'; DIM=$'\033[2m'
    RED=$'\033[31m'; GREEN=$'\033[32m'; YELLOW=$'\033[33m'
    BLUE=$'\033[34m'; CYAN=$'\033[36m'; RESET=$'\033[0m'
else
    BOLD=""; DIM=""; RED=""; GREEN=""; YELLOW=""; BLUE=""; CYAN=""; RESET=""
fi
log()  { printf '\n%s[INFO]%s %s\n' "$BLUE" "$RESET" "$*"; }
ok()   { printf '%s[ OK ]%s %s\n'   "$GREEN" "$RESET" "$*"; }
warn() { printf '%s[WARN]%s %s\n'   "$YELLOW" "$RESET" "$*" >&2; }
die()  { printf '\n%s[FAIL]%s %s\n' "$RED" "$RESET" "$*" >&2; exit 1; }
hdr()  { printf '\n%s%s%s\n' "$BOLD" "$*" "$RESET"; }

trap 'printf "\n%s[FAIL]%s Installer failed near line %s.\n" "$RED" "$RESET" "$LINENO" >&2' ERR

while [[ $# -gt 0 ]]; do
    case "$1" in
        --backend)      BACKEND_OVERRIDE="${2:-}"; shift 2 ;;
        --dir)          REPO_DIR="${2:-}"; shift 2 ;;
        --no-launch)    LAUNCH=0; shift ;;
        --no-integrate) INTEGRATE=0; shift ;;
        --help|-h)
            cat <<EOF
CrowdGPT installer

Options:
  --backend <name>   Force backend: nvidia | amd | intel | apple | cpu
  --dir <path>       Directory to clone into (default: ./CrowdGPT)
  --no-launch        Install only, don't launch client.py
  --no-integrate     Skip desktop/command integration
  --help             Show this message
EOF
            exit 0
            ;;
        *) die "Unknown argument: $1 (see --help)" ;;
    esac
done

echo
echo "============================================================"
echo "                 CrowdGPT Installer  (v8)"
echo "============================================================"
echo

OS="$(uname -s)"
ARCH="$(uname -m)"

run_as_root() {
    if [[ $EUID -eq 0 ]]; then
        "$@"
    elif command -v sudo >/dev/null 2>&1; then
        sudo "$@"
    else
        die "Root privileges required but sudo is unavailable."
    fi
}

install_uv() {
    command -v uv >/dev/null 2>&1 && return
    log "Bootstrapping uv (Python manager)..."
    if command -v curl >/dev/null 2>&1; then
        curl -LsSf https://astral.sh/uv/install.sh | sh
    elif command -v wget >/dev/null 2>&1; then
        wget -qO- https://astral.sh/uv/install.sh | sh
    else
        die "Need curl or wget to bootstrap uv."
    fi
    export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
    command -v uv >/dev/null 2>&1 || die "uv installed but not on PATH."
}

find_python() {
    local candidates=(python3.12 python3.11 python3.10 python3 python)
    for c in "${candidates[@]}"; do
        if command -v "$c" >/dev/null 2>&1; then
            local ver
            ver="$("$c" -c 'import sys; print("%d.%d" % sys.version_info[:2])' 2>/dev/null || echo "0.0")"
            local major="${ver%%.*}" minor="${ver##*.}"
            if [[ "$major" -eq 3 && "$minor" -ge 10 ]]; then
                PYTHON_EXE="$(command -v "$c")"
                return
            fi
        fi
    done
    install_uv
    log "No suitable Python 3.10+ found — installing via uv..."
    uv python install 3.11 --default
    export PATH="$HOME/.local/bin:$PATH"
    PYTHON_EXE="$(uv python find 3.11)"
}

copy_tree() {
    local src="$1" dst="$2"
    mkdir -p "$dst"
    if command -v rsync >/dev/null 2>&1; then
        rsync -a --delete \
            --exclude='.git' \
            --exclude='.venv' \
            --exclude='__pycache__' \
            --exclude='*.pyc' \
            --exclude='checkpoints' \
            --exclude='pending_uploads' \
            "$src"/ "$dst"/
    else
        find "$dst" -mindepth 1 -maxdepth 1 ! -name '.git' -exec rm -rf {} + 2>/dev/null || true
        ( cd "$src" && tar cf - \
            --exclude='.git' \
            --exclude='.venv' \
            --exclude='__pycache__' \
            --exclude='*.pyc' \
            --exclude='checkpoints' \
            --exclude='pending_uploads' \
            . ) | ( cd "$dst" && tar xf - )
    fi
}

# ============================================================
# 1. Python (>= 3.10)
# ============================================================
PYTHON_EXE=""
find_python
ok "Python: $PYTHON_EXE"
"$PYTHON_EXE" --version

# ============================================================
# 2. Git
# ============================================================
if ! command -v git >/dev/null 2>&1; then
    log "Installing Git..."
    if [[ "$OS" == "Linux" ]]; then
        if   command -v apt-get >/dev/null 2>&1; then run_as_root apt-get update && run_as_root apt-get install -y git
        elif command -v dnf     >/dev/null 2>&1; then run_as_root dnf install -y git
        elif command -v pacman  >/dev/null 2>&1; then run_as_root pacman -Sy --noconfirm git
        elif command -v zypper  >/dev/null 2>&1; then run_as_root zypper --non-interactive install git
        else die "Unsupported Linux package manager. Install git manually."
        fi
    elif [[ "$OS" == "Darwin" ]]; then
        if command -v brew >/dev/null 2>&1; then
            brew install git
        else
            xcode-select --install >/dev/null 2>&1 || true
            die "Install Xcode Command Line Tools or Homebrew, then rerun."
        fi
    fi
fi
ok "Git: $(git --version)"

# ============================================================
# 3. Clone / update repo
# ============================================================
if [[ -f "client.py" && -d ".git" ]]; then
    REPO_DIR="."
elif [[ -d "$REPO_DIR/.git" ]]; then
    log "Updating existing checkout..."
    git -C "$REPO_DIR" pull --ff-only || warn "git pull failed; continuing."
else
    [[ -e "$REPO_DIR" ]] && die "$REPO_DIR exists but is not a git repo."
    log "Cloning CrowdGPT..."
    git clone "$REPO_URL" "$REPO_DIR"
fi
cd "$REPO_DIR"
[[ -f client.py ]] || die "client.py not found after checkout."
SRC_DIR="$PWD"

# ============================================================
# 4. Hardware detection
# ============================================================
detect_backend() {
    BACKEND=""
    TORCH_INDEX=""
    TORCH_EXTRA=""
    WHEEL_NOTE=""

    if [[ "$OS" == "Darwin" ]]; then
        if [[ "$ARCH" == "arm64" ]]; then
            BACKEND="apple"
            TORCH_INDEX="https://pypi.org/simple"
            WHEEL_NOTE="Apple Silicon (MPS) — bundled with default PyTorch wheel"
        else
            BACKEND="cpu"
            TORCH_INDEX="https://download.pytorch.org/whl/cpu"
            WHEEL_NOTE="Intel Mac — CPU only"
        fi
        return
    fi

    if command -v nvidia-smi >/dev/null 2>&1; then
        local DRIVER MAJOR
        DRIVER="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -n1 | tr -d '[:space:]' || true)"

        if [[ "$ARCH" == "aarch64" ]]; then
            BACKEND="cpu"
            TORCH_INDEX="https://download.pytorch.org/whl/cpu"
            WHEEL_NOTE="Jetson (aarch64) — generic CUDA wheels are incompatible. Using CPU for now."
            warn "Jetson detected. For GPU support, install NVIDIA's Jetson wheels manually:"
            warn "  https://developer.download.nvidia.com/compute/redist/jp/"
            warn "Continuing with CPU PyTorch."
            return
        fi

        MAJOR="${DRIVER%%.*}"
        if [[ -z "$DRIVER" ]]; then
            BACKEND="nvidia"
            TORCH_INDEX="https://download.pytorch.org/whl/cu124"
            WHEEL_NOTE="NVIDIA GPU — could not read driver version, defaulting to cu124"
        elif [[ "$MAJOR" =~ ^[0-9]+$ ]] && (( MAJOR >= 525 )); then
            BACKEND="nvidia"
            TORCH_INDEX="https://download.pytorch.org/whl/cu124"
            WHEEL_NOTE="NVIDIA driver $DRIVER -> CUDA 12.4 wheel"
        elif [[ "$MAJOR" =~ ^[0-9]+$ ]] && (( MAJOR >= 470 )); then
            BACKEND="nvidia"
            TORCH_INDEX="https://download.pytorch.org/whl/cu118"
            WHEEL_NOTE="NVIDIA driver $DRIVER -> CUDA 11.8 wheel (older driver)"
        else
            die "NVIDIA driver '$DRIVER' is too old. Upgrade to >= 470."
        fi
        return
    fi

    local ROCM_VER=""
    if [[ -f /opt/rocm/.info/version ]]; then
        ROCM_VER="$(cut -d. -f1,2 < /opt/rocm/.info/version 2>/dev/null || true)"
    fi
    if [[ -z "$ROCM_VER" ]] && command -v rocminfo >/dev/null 2>&1; then
        ROCM_VER="$(rocminfo 2>/dev/null | grep -oE 'ROCm[[:space:]]+[0-9]+\.[0-9]+' | head -n1 | awk '{print $2}' || true)"
    fi
    if [[ -n "$ROCM_VER" ]] || [[ -d /opt/rocm ]] || command -v rocm-smi >/dev/null 2>&1; then
        BACKEND="amd"
        case "$ROCM_VER" in
            6.3*|6.4*) TORCH_INDEX="https://download.pytorch.org/whl/rocm6.3" ;;
            6.2*)      TORCH_INDEX="https://download.pytorch.org/whl/rocm6.2" ;;
            6.1*)      TORCH_INDEX="https://download.pytorch.org/whl/rocm6.1" ;;
            6.*)       TORCH_INDEX="https://download.pytorch.org/whl/rocm6.1" ;;
            5.*)       TORCH_INDEX="https://download.pytorch.org/whl/rocm5.7" ;;
            *)         TORCH_INDEX="https://download.pytorch.org/whl/rocm6.1" ;;
        esac
        WHEEL_NOTE="AMD ROCm ${ROCM_VER:-unknown} -> ${TORCH_INDEX##*/}"
        if command -v rocminfo >/dev/null 2>&1; then
            local GFX
            GFX="$(rocminfo 2>/dev/null | grep -oE 'gfx[0-9a-f]+' | sort -u | head -n1 || true)"
            case "$GFX" in
                gfx103*) warn "RX 6000 (Navi 2x) — you may need: export HSA_OVERRIDE_GFX_VERSION=10.3.0" ;;
                gfx110*) warn "RX 7000 (Navi 3x) — you may need: export HSA_OVERRIDE_GFX_VERSION=11.0.0" ;;
                gfx120*) : ;;
                "")      warn "Could not read GPU arch. If torch can't see the GPU, try HSA_OVERRIDE_GFX_VERSION matching your arch." ;;
                *)       warn "GPU arch: $GFX — check AMD's ROCm compatibility matrix." ;;
            esac
        fi
        return
    fi

    if command -v sycl-ls >/dev/null 2>&1 || [[ -d /opt/intel/oneapi ]]; then
        BACKEND="intel"
        TORCH_INDEX="https://download.pytorch.org/whl/xpu"
        TORCH_EXTRA="https://pypi.org/simple"
        WHEEL_NOTE="Intel Arc / Data Center GPU -> XPU wheel"
        return
    fi

    BACKEND="cpu"
    TORCH_INDEX="https://download.pytorch.org/whl/cpu"
    WHEEL_NOTE="No GPU detected — CPU wheel (training will be very slow)"
}

if [[ -n "$BACKEND_OVERRIDE" ]]; then
    case "$BACKEND_OVERRIDE" in
        nvidia) BACKEND="nvidia"; TORCH_INDEX="https://download.pytorch.org/whl/cu124"; WHEEL_NOTE="forced: nvidia/cu124" ;;
        amd)    BACKEND="amd";    TORCH_INDEX="https://download.pytorch.org/whl/rocm6.1"; WHEEL_NOTE="forced: amd/rocm6.1" ;;
        intel)  BACKEND="intel";  TORCH_INDEX="https://download.pytorch.org/whl/xpu";    WHEEL_NOTE="forced: intel/xpu" ;;
        apple)  BACKEND="apple";  TORCH_INDEX="https://pypi.org/simple";                WHEEL_NOTE="forced: apple/mps" ;;
        cpu)    BACKEND="cpu";    TORCH_INDEX="https://download.pytorch.org/whl/cpu";   WHEEL_NOTE="forced: cpu" ;;
        *) die "--backend must be one of: nvidia | amd | intel | apple | cpu" ;;
    esac
else
    detect_backend
fi

hdr "Backend"
echo "  Detected:    $BACKEND"
echo "  Torch index: $TORCH_INDEX"
echo "  Note:        $WHEEL_NOTE"

# ============================================================
# 5. Install paths
# ============================================================
if [[ "$OS" == "Darwin" ]]; then
    APP_DIR="$HOME/Library/Application Support/CrowdGPT"
    BIN_DIR="/usr/local/bin"
    APPS_DIR="$HOME/Applications"
    APP_BUNDLE="$APPS_DIR/CrowdGPT.app"
else
    APP_DIR="$HOME/.local/share/crowdgpt"
    BIN_DIR="$HOME/.local/bin"
    DESKTOP_DIR="$HOME/.local/share/applications"
    ICON_DIR="$HOME/.local/share/icons/hicolor/256x256/apps"
fi

if [[ "$INTEGRATE" -eq 1 ]]; then
    VENV_DIR="$APP_DIR/.venv"
    mkdir -p "$APP_DIR"
else
    VENV_DIR="$SRC_DIR/.venv"
fi
APP_PY="$VENV_DIR/bin/python"

# ============================================================
# 6. Create venv AT ITS FINAL LOCATION
# ============================================================
log "Creating venv (with system site-packages for GTK access) at $VENV_DIR..."
rm -rf "$VENV_DIR"
if ! "$PYTHON_EXE" -m venv "$VENV_DIR" --system-site-packages 2>/dev/null; then
    install_uv
    uv venv --system-site-packages --python 3.11 "$VENV_DIR"
fi

[[ -x "$APP_PY" ]] || die "venv Python missing at $APP_PY"

log "Upgrading pip..."
"$APP_PY" -m pip install --upgrade pip >/dev/null

# ============================================================
# 7. Install PyTorch
# ============================================================
hdr "Installing PyTorch"

"$APP_PY" -m pip uninstall -y torch torchvision torchaudio >/dev/null 2>&1 || true

PIP_TORCH_ARGS=(
    install
    torch torchvision torchaudio
    --index-url "$TORCH_INDEX"
)
if [[ -n "$TORCH_EXTRA" ]]; then
    PIP_TORCH_ARGS+=( --extra-index-url "$TORCH_EXTRA" )
fi

log "pip ${PIP_TORCH_ARGS[*]}"
if ! "$APP_PY" -m pip "${PIP_TORCH_ARGS[@]}"; then
    die "PyTorch install failed from $TORCH_INDEX"
fi

TORCH_VER="$("$APP_PY" -c 'import torch; print(torch.__version__)' 2>/dev/null || echo "0")"
TORCH_MAJOR_MINOR="$(printf '%s' "$TORCH_VER" | grep -oE '^[0-9]+\.[0-9]+' || echo "0.0")"
log "Installed torch version: $TORCH_VER"

# ============================================================
# 8. Install project requirements (torch lines stripped)
# ============================================================
if [[ -f requirements.txt ]]; then
    hdr "Installing project dependencies"
    TMP_REQ="$(mktemp -t crowdgpt-req.XXXXXX.txt)"
    grep -viE '^[[:space:]]*(torch|torchvision|torchaudio)([[:space:]]*[<>=!~].*)?$' \
        requirements.txt > "$TMP_REQ" || true
    if [[ -s "$TMP_REQ" ]]; then
        "$APP_PY" -m pip install -r "$TMP_REQ"
    else
        warn "requirements.txt contained only torch packages; nothing to install."
    fi
    rm -f "$TMP_REQ"
else
    warn "requirements.txt not found; skipping project deps."
fi

# ============================================================
# 8b. Pin transformers when torch < 2.5
# ============================================================
NEED_OLD_TRANSFORMERS=0
case "$TORCH_MAJOR_MINOR" in
    2.0|2.1|2.2|2.3|2.4) NEED_OLD_TRANSFORMERS=1 ;;
esac

if [[ "$NEED_OLD_TRANSFORMERS" -eq 1 ]]; then
    log "torch $TORCH_VER < 2.5 — pinning transformers to 4.x..."
    if ! "$APP_PY" -m pip install --upgrade "transformers>=4.45,<5.0"; then
        die "Failed to pin transformers to 4.x for torch $TORCH_VER."
    fi
    ok "transformers pinned to 4.x"
fi

# ============================================================
# 9. GUI backend for pywebview
# ============================================================
hdr "GUI backend for pywebview"

GTK_WORKS=0
if "$APP_PY" -c "import gi" 2>/dev/null; then
    GTK_WORKS=1
    ok "GTK bindings available (system PyGObject)"
fi

if [[ "$GTK_WORKS" -eq 0 ]]; then
    log "GTK not available inside venv — installing PyQt6 fallback..."
    if ! "$APP_PY" -m pip install PyQt6 PyQt6-WebEngine; then
        warn "PyQt6 install failed. Trying PyQt5 as last resort..."
        if ! "$APP_PY" -m pip install PyQt5 PyQtWebEngine; then
            die "Could not install any Qt backend. pywebview needs GTK or Qt."
        fi
    fi
    ok "Qt backend installed"
fi

if ! "$APP_PY" - <<'PYEOF'
try:
    import webview  # noqa
    print("  pywebview: OK")
except Exception as e:
    print(f"  pywebview import error: {e}")
    raise SystemExit(1)
PYEOF
then
    warn "pywebview preflight failed — the client may not open a window"
fi

# ============================================================
# 10. Verify
# ============================================================
hdr "Verifying install"

"$APP_PY" - <<'PYEOF' || warn "torch verification print failed (continuing)"
import sys, torch
print("  Python:", sys.version.split()[0])
print("  Torch:  ", torch.__version__)
print("  CUDA:   ", torch.cuda.is_available(), end="")
if torch.cuda.is_available():
    print(f"  ({torch.cuda.get_device_name(0)})")
else:
    print()
try:
    if hasattr(torch.backends, "mps"):
        print("  MPS:    ", torch.backends.mps.is_available())
except Exception:
    pass
try:
    import torch.version
    if torch.version.hip:
        print("  ROCm:   ", torch.version.hip)
except Exception:
    pass
PYEOF

# >>> FIX v8: use `if ! ...` so `set -e` doesn't exit before the
#             actionable error message is printed.
if ! "$APP_PY" - <<'PYEOF'
import transformers, torch
assert transformers.utils.is_torch_available(), (
    "transformers refuses to use installed torch — version mismatch "
    f"(transformers {transformers.__version__} vs torch {torch.__version__})"
)
print(f"  transformers: {transformers.__version__} -> torch {torch.__version__} linked OK")
PYEOF
then
    die "transformers cannot use the installed torch. Model loading WILL fail at runtime.
    Fix: \"$APP_PY\" -m pip install 'transformers>=4.45,<5.0'
    (or upgrade torch to >=2.5), then rerun this installer."
fi

# ============================================================
# 11. Desktop + command integration
# ============================================================
if [[ "$INTEGRATE" -eq 1 ]]; then
    hdr "Installing desktop integration"
    log "Install location: $APP_DIR"

    if [[ "$OS" == "Darwin" ]]; then
        mkdir -p "$APPS_DIR"
    else
        mkdir -p "$BIN_DIR" "$DESKTOP_DIR" "$ICON_DIR"
    fi

    log "Copying app files (venv excluded; already in place)..."
    copy_tree "$SRC_DIR" "$APP_DIR"

    [[ -x "$APP_PY" ]] || die "Installed venv Python missing at $APP_PY"

    # ---------- Linux ----------
    if [[ "$OS" != "Darwin" ]]; then
        cat > "$BIN_DIR/crowdgpt" <<EOF
#!/usr/bin/env bash
exec "$APP_PY" "$APP_DIR/client.py" "\$@"
EOF
        chmod +x "$BIN_DIR/crowdgpt"

        ICON_SRC=""
        for c in "$APP_DIR/docs/logo-app.svg" "$APP_DIR/docs/logo-black.svg" "$APP_DIR/docs/logo-white.svg"; do
            [[ -f "$c" ]] && ICON_SRC="$c" && break
        done

        ICON_NAME="crowdgpt"
        if [[ -n "$ICON_SRC" ]]; then
            if command -v rsvg-convert >/dev/null 2>&1; then
                rsvg-convert -w 256 -h 256 "$ICON_SRC" -o "$ICON_DIR/crowdgpt.png" 2>/dev/null || true
            elif "$APP_PY" -c "import cairosvg" 2>/dev/null; then
                "$APP_PY" -c "import cairosvg; cairosvg.svg2png(url='$ICON_SRC', write_to='$ICON_DIR/crowdgpt.png', output_width=256, output_height=256)" 2>/dev/null || true
            fi
            if [[ ! -f "$ICON_DIR/crowdgpt.png" ]]; then
                cp "$ICON_SRC" "$ICON_DIR/crowdgpt.svg" 2>/dev/null || true
            fi
        fi

        cat > "$DESKTOP_DIR/crowdgpt.desktop" <<EOF
[Desktop Entry]
Version=1.0
Type=Application
Name=CrowdGPT
GenericName=Decentralized AI Training Client
Comment=Contribute GPU cycles to the CrowdGPT network
Exec="$BIN_DIR/crowdgpt"
Icon=$ICON_NAME
Terminal=false
Categories=Science;Network;Utility;
Keywords=AI;ML;Training;Distributed;LLM;
StartupNotify=true
StartupWMClass=CrowdGPT
EOF
        chmod +x "$DESKTOP_DIR/crowdgpt.desktop"

        command -v update-desktop-database >/dev/null 2>&1 && \
            update-desktop-database "$DESKTOP_DIR" >/dev/null 2>&1 || true
        command -v gtk-update-icon-cache >/dev/null 2>&1 && \
            gtk-update-icon-cache -f -t "$HOME/.local/share/icons/hicolor" >/dev/null 2>&1 || true

        cat > "$BIN_DIR/crowdgpt-uninstall" <<EOF
#!/usr/bin/env bash
set -e
echo "Uninstalling CrowdGPT..."
rm -rf "$APP_DIR"
rm -f "$BIN_DIR/crowdgpt" "$BIN_DIR/crowdgpt-uninstall"
rm -f "$DESKTOP_DIR/crowdgpt.desktop"
rm -f "$ICON_DIR/crowdgpt.png" "$ICON_DIR/crowdgpt.svg"
command -v update-desktop-database >/dev/null 2>&1 && update-desktop-database "$DESKTOP_DIR" >/dev/null 2>&1 || true
echo "Done."
EOF
        chmod +x "$BIN_DIR/crowdgpt-uninstall"

        ok "Command:   $BIN_DIR/crowdgpt"
        ok "App entry: $DESKTOP_DIR/crowdgpt.desktop"
        ok "Uninstall: crowdgpt-uninstall"

        if ! echo "$PATH" | tr ':' '\n' | grep -qx "$BIN_DIR"; then
            warn "$BIN_DIR is not on your PATH."
            warn "Add this to ~/.bashrc or ~/.zshrc:"
            warn "    export PATH=\"\$HOME/.local/bin:\$PATH\""
        fi
    fi

    # ---------- macOS ----------
    if [[ "$OS" == "Darwin" ]]; then
        if [[ -w "$BIN_DIR" ]]; then
            cat > "$BIN_DIR/crowdgpt" <<EOF
#!/usr/bin/env bash
exec "$APP_PY" "$APP_DIR/client.py" "\$@"
EOF
            chmod +x "$BIN_DIR/crowdgpt"
        else
            log "Creating $BIN_DIR/crowdgpt (requires sudo)..."
            run_as_root tee "$BIN_DIR/crowdgpt" >/dev/null <<EOF
#!/usr/bin/env bash
exec "$APP_PY" "$APP_DIR/client.py" "\$@"
EOF
            run_as_root chmod +x "$BIN_DIR/crowdgpt"
        fi

        rm -rf "$APP_BUNDLE"
        mkdir -p "$APP_BUNDLE/Contents/MacOS" "$APP_BUNDLE/Contents/Resources"

        cat > "$APP_BUNDLE/Contents/Info.plist" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleName</key><string>CrowdGPT</string>
    <key>CFBundleDisplayName</key><string>CrowdGPT</string>
    <key>CFBundleIdentifier</key><string>net.crowdgpt.client</string>
    <key>CFBundleVersion</key><string>0.8.0</string>
    <key>CFBundleShortVersionString</key><string>0.8</string>
    <key>CFBundlePackageType</key><string>APPL</string>
    <key>CFBundleExecutable</key><string>launcher</string>
    <key>CFBundleIconFile</key><string>icon.icns</string>
    <key>LSMinimumSystemVersion</key><string>11.0</string>
    <key>NSHighResolutionCapable</key><true/>
    <key>LSUIElement</key><false/>
</dict>
</plist>
EOF

        cat > "$APP_BUNDLE/Contents/MacOS/launcher" <<EOF
#!/usr/bin/env bash
exec "$APP_PY" "$APP_DIR/client.py"
EOF
        chmod +x "$APP_BUNDLE/Contents/MacOS/launcher"

        ICON_SRC=""
        for c in "$APP_DIR/docs/logo-app.svg" "$APP_DIR/docs/logo-black.svg"; do
            [[ -f "$c" ]] && ICON_SRC="$c" && break
        done
        if [[ -n "$ICON_SRC" ]] && command -v rsvg-convert >/dev/null 2>&1; then
            TMP_ICONSET="$(mktemp -d)/crowdgpt.iconset"
            mkdir -p "$TMP_ICONSET"
            for size in 16 32 64 128 256 512; do
                rsvg-convert -w $size -h $size "$ICON_SRC" -o "$TMP_ICONSET/icon_${size}x${size}.png" 2>/dev/null || true
                rsvg-convert -w $((size*2)) -h $((size*2)) "$ICON_SRC" -o "$TMP_ICONSET/icon_${size}x${size}@2x.png" 2>/dev/null || true
            done
            iconutil -c icns "$TMP_ICONSET" -o "$APP_BUNDLE/Contents/Resources/icon.icns" 2>/dev/null || true
            rm -rf "$TMP_ICONSET"
        fi

        if [[ -w "$BIN_DIR" ]]; then
            cat > "$BIN_DIR/crowdgpt-uninstall" <<EOF
#!/usr/bin/env bash
set -e
echo "Uninstalling CrowdGPT..."
rm -f "$BIN_DIR/crowdgpt" "$BIN_DIR/crowdgpt-uninstall"
rm -rf "$APP_DIR" "$APP_BUNDLE"
echo "Done."
EOF
            chmod +x "$BIN_DIR/crowdgpt-uninstall"
        else
            run_as_root tee "$BIN_DIR/crowdgpt-uninstall" >/dev/null <<EOF
#!/usr/bin/env bash
set -e
echo "Uninstalling CrowdGPT..."
rm -f "$BIN_DIR/crowdgpt" "$BIN_DIR/crowdgpt-uninstall"
rm -rf "$APP_DIR" "$APP_BUNDLE"
echo "Done."
EOF
            run_as_root chmod +x "$BIN_DIR/crowdgpt-uninstall"
        fi

        /System/Library/Frameworks/CoreServices.framework/Frameworks/LaunchServices.framework/Support/lsregister -f "$APP_BUNDLE" 2>/dev/null || true

        ok "Command:    crowdgpt"
        ok "App bundle: $APP_BUNDLE"
        ok "Uninstall:  crowdgpt-uninstall"

        if [[ ! -w "$BIN_DIR" ]]; then
            warn "/usr/local/bin was not writable; uninstall requires sudo."
        fi
    fi

    echo
    echo "============================================================"
    echo "             Installation complete"
    echo "============================================================"
    echo "  Backend: $BACKEND"
    echo "  Torch:   $TORCH_VER"
    echo
    echo "  Launch from terminal:  crowdgpt"
    if [[ "$OS" == "Darwin" ]]; then
        echo "  Launch from GUI:       Spotlight -> CrowdGPT"
    else
        echo "  Launch from GUI:       Activities / app menu -> CrowdGPT"
    fi
    echo "  Uninstall:             crowdgpt-uninstall"
    echo
else
    hdr "Skipping desktop integration (--no-integrate)"
    echo "  Build venv: $VENV_DIR"
fi

# ============================================================
# 12. Launch
# ============================================================
if [[ "$LAUNCH" -eq 1 ]]; then
    echo "Launching client.py..."
    echo
    if [[ "$INTEGRATE" -eq 1 ]]; then
        exec "$APP_PY" "$APP_DIR/client.py"
    else
        exec "$APP_PY" "$SRC_DIR/client.py"
    fi
else
    if [[ "$INTEGRATE" -eq 1 ]]; then
        echo "To launch later:  crowdgpt"
    else
        echo "To launch later:  $APP_PY $SRC_DIR/client.py"
    fi
fi
