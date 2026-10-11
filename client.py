#!/usr/bin/env python3
"""
CrowdGPT GUI Client — Clean rewrite.

- HF-direct weight downloads (bypasses Cloudflare tunnel size limits)
- Coordinator inline fallback with hard size check
- Chunked delta upload (64 MB chunks, per-chunk SHA-256, retries)
- Auto-update from GitHub via docs/version.txt
- MOTD from docs/motd.txt
- Multi-language (en/fr/es/de/tr)
- Real-time inference preview during training
"""

import os, sys, io, json, time, struct, math, gc, threading, base64, logging, tempfile, atexit, subprocess, hashlib, shutil
from pathlib import Path
import webview
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import requests
from torch.utils.checkpoint import checkpoint
import http.client
from requests.exceptions import ChunkedEncodingError
import gzip

logging.getLogger("pywebview").setLevel(logging.CRITICAL)

os.environ['PYTORCH_ALLOC_CONF'] = 'expandable_segments:True'
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', datefmt='%H:%M:%S')
log = logging.getLogger(__name__)


# ============================================================
# AUTO-UPDATE
# ============================================================
CLIENT_VERSION = "0.6"
VERSION_URL = "https://raw.githubusercontent.com/Vxtzq/CrowdGPT/main/docs/version.txt"
CLIENT_URL  = "https://raw.githubusercontent.com/Vxtzq/CrowdGPT/main/client.py"
UPDATE_TIMEOUT_VERSION = 5
UPDATE_TIMEOUT_CLIENT  = 60
UPDATE_CHECK_BETWEEN_ROUNDS = True


def _parse_version(v):
    try:
        parts = str(v).strip().split("-")[0].split(".")
        return tuple(int(x) for x in parts[:4])
    except Exception:
        return (0,)


def _fetch_remote_version():
    try:
        r = requests.get(VERSION_URL, timeout=UPDATE_TIMEOUT_VERSION)
        r.raise_for_status()
        line = r.text.strip().splitlines()[0].strip()
        return line or None
    except Exception as e:
        log.info(f"Version check skipped: {e}")
        return None


def _download_new_client(expected_version):
    try:
        r = requests.get(CLIENT_URL, timeout=UPDATE_TIMEOUT_CLIENT)
        r.raise_for_status()
        code = r.text
    except Exception as e:
        log.warning(f"client.py download failed: {e}")
        return None
    try:
        compile(code, "client.py", "exec")
    except SyntaxError as e:
        log.warning(f"Downloaded client.py failed syntax check: {e}")
        return None
    markers = [
        f'CLIENT_VERSION = "{expected_version}"',
        f"CLIENT_VERSION = '{expected_version}'",
    ]
    if not any(m in code for m in markers):
        log.warning(f"Downloaded client.py does not declare version {expected_version}")
        return None
    return code


def _apply_update(new_code):
    try:
        current = Path(__file__).resolve()
    except Exception:
        return False
    backup = current.with_name(current.name + ".bak")
    tmp = current.with_name(current.name + ".new")
    try:
        try:
            shutil.copy2(str(current), str(backup))
        except Exception:
            pass
        tmp.write_text(new_code, encoding="utf-8")
        try:
            st = current.stat()
            os.chmod(str(tmp), st.st_mode)
        except Exception:
            pass
        os.replace(str(tmp), str(current))
        return True
    except Exception as e:
        log.warning(f"Failed to apply update: {e}")
        try:
            if tmp.exists():
                tmp.unlink()
        except Exception:
            pass
        return False


def check_for_update():
    remote = _fetch_remote_version()
    if not remote:
        return False
    if _parse_version(remote) <= _parse_version(CLIENT_VERSION):
        return False
    log.info(f"Update available: {CLIENT_VERSION} -> {remote}")
    new_code = _download_new_client(remote)
    if not new_code:
        return False
    if not _apply_update(new_code):
        return False
    log.info(f"client.py updated to {remote}")
    return True


def _restart_self():
    python = sys.executable
    script = str(Path(__file__).resolve())
    args = [python, script] + sys.argv[1:]
    if os.name == "nt":
        subprocess.Popen(args, close_fds=True, creationflags=0x00000008)
        sys.exit(0)
    else:
        os.execv(python, args)


# ============================================================
# CONFIG
# ============================================================
CHECKPOINT_DIR = Path("checkpoints")
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_SERVER = "https://server.crowdgpt.net"
DATASET_REPO_ID = "Vxtzq/CrowdGPT"
TOKENIZER_REPO = "Qwen/Qwen3-0.6B"
HF_MODEL_REPO_ID = "Vxtzq/Crowd-v1"
USE_HF_WEIGHTS = True

UPLOAD_CHUNK_BYTES = 64 * 1024 * 1024
MAX_DIRECT_UPLOAD_BYTES = 80 * 1024 * 1024
MAX_CHUNK_RETRY = 4

MODEL_CONFIG = {
    "vocabSize": 151669, "dim": 1536, "nLayers": 24, "nHeads": 16, "nKvHeads": 4,
    "headDim": 96, "maxSeqLen": 2048, "mlpHidden": 2560, "weightTying": True,
}
VOCAB_SIZE = MODEL_CONFIG["vocabSize"]
DIM = MODEL_CONFIG["dim"]
N_LAYERS = MODEL_CONFIG["nLayers"]
N_HEADS = MODEL_CONFIG["nHeads"]
N_KV_HEADS = MODEL_CONFIG["nKvHeads"]
HEAD_DIM = MODEL_CONFIG["headDim"]
MAX_SEQ_LEN = MODEL_CONFIG["maxSeqLen"]
MLP_HIDDEN = MODEL_CONFIG["mlpHidden"]
ENG_NUM_BUCKETS = 227865
LOSS_CHUNK = 256
UPLOAD_BUFFER_MIN = 25
ALLOWED_BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64, 128]


def _calc_model_size():
    size = VOCAB_SIZE * DIM
    for _ in range(N_LAYERS):
        size += DIM * 2 + DIM * (N_HEADS * HEAD_DIM) + DIM * (N_KV_HEADS * HEAD_DIM) * 2 + DIM * DIM + DIM * 2
        size += DIM * MLP_HIDDEN * 2 + MLP_HIDDEN * DIM
    size += DIM * 2
    return size


EXPECTED_MODEL_SIZE = _calc_model_size()
EXPECTED_ENGRAM_SIZE = ENG_NUM_BUCKETS * DIM
EXPECTED_MODEL_BYTES = EXPECTED_MODEL_SIZE * 2
EXPECTED_ENGRAM_BYTES = EXPECTED_ENGRAM_SIZE * 2
EXPECTED_WEIGHT_BYTES = EXPECTED_MODEL_BYTES + EXPECTED_ENGRAM_BYTES
memory_config = {"ram_gb": 12, "is_auto_detected": False}
train_device, train_backend = None, None


def safe_float(v, default=0.0):
    try:
        if v is None or math.isnan(v) or math.isinf(v):
            return default
        return float(v)
    except Exception:
        return default


def sanitize_for_json(obj):
    if isinstance(obj, dict):
        return {str(k): sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize_for_json(v) for v in obj]
    if isinstance(obj, (int, float, str, bool, type(None))):
        if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
            return 0.0
        return obj
    try:
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            v = float(obj)
            return 0.0 if math.isnan(v) or math.isinf(v) else v
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
    except ImportError:
        pass
    return str(obj)


# ============================================================
# TOKENIZER
# ============================================================
TOKENIZER = None
_tokenizer_started = False
PREVIEW_EVERY = 10


def start_tokenizer_loader():
    global _tokenizer_started
    if _tokenizer_started:
        return
    _tokenizer_started = True
    threading.Thread(target=_load_tokenizer_worker, daemon=True).start()


def _load_tokenizer_worker():
    global TOKENIZER
    try:
        from transformers import AutoTokenizer
        candidates = [TOKENIZER_REPO, "Qwen/Qwen2.5-1.5B", "Qwen/Qwen2.5-0.5B", "Vxtzq/Crowd-v1"]
        for repo in candidates:
            try:
                tok = AutoTokenizer.from_pretrained(repo, trust_remote_code=True)
                if len(tok) >= 151000:
                    TOKENIZER = tok
                    log.info(f"Tokenizer loaded: {repo} ({len(tok)} tokens)")
                    return
            except Exception:
                continue
        log.warning("No compatible tokenizer loaded.")
    except ImportError:
        log.warning("transformers not installed.")
    TOKENIZER = None


def decode_ids(ids, max_chars=220):
    try:
        ids = [int(i) for i in ids]
        if TOKENIZER is not None:
            s = TOKENIZER.decode(ids, skip_special_tokens=True)
            s = ' '.join(s.split())
            if len(s) > max_chars:
                s = s[:max_chars] + "…"
            return s or "…"
    except Exception:
        pass
    s = ' '.join(f"#{int(i)}" for i in ids)
    if len(s) > max_chars:
        s = s[:max_chars] + "…"
    return s


def decode_one(tok_id):
    s = decode_ids([tok_id], max_chars=32)
    return s if s.strip() else "␠"


# ============================================================
# LOGOS / ICON
# ============================================================
LOGO_URI_LIGHT = ""
LOGO_URI_DARK = ""
_lp_light = Path(__file__).parent / "docs/logo-black.svg"
_lp_app = Path(__file__).parent / "docs/logo-app.svg"
_lp_dark = Path(__file__).parent / "docs/logo-white.svg"
if _lp_light.exists():
    try:
        LOGO_URI_LIGHT = "data:image/svg+xml;base64," + base64.b64encode(_lp_light.read_bytes()).decode()
    except Exception:
        pass
if _lp_dark.exists():
    try:
        LOGO_URI_DARK = "data:image/svg+xml;base64," + base64.b64encode(_lp_dark.read_bytes()).decode()
    except Exception:
        pass
if not LOGO_URI_DARK:
    LOGO_URI_DARK = LOGO_URI_LIGHT

ICON_PATH = None
_cleanup_icon = []
if _lp_light.exists():
    try:
        import cairosvg
        png_bytes = cairosvg.svg2png(url=str(_lp_app), output_width=256, output_height=256)
        tmp = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
        tmp.write(png_bytes)
        tmp.close()
        ICON_PATH = os.path.abspath(tmp.name)
        _cleanup_icon.append(ICON_PATH)

        def _clean():
            for p in _cleanup_icon:
                try:
                    os.unlink(p)
                except Exception:
                    pass
        atexit.register(_clean)
    except ImportError:
        ICON_PATH = os.path.abspath(str(_lp_app))
    except Exception:
        ICON_PATH = os.path.abspath(str(_lp_app))


# ============================================================
# HARDWARE
# ============================================================
def get_available_backends():
    backends = {}
    backends['cuda'] = torch.cuda.is_available() and not (hasattr(torch.version, 'hip') and torch.version.hip)
    backends['rocm'] = hasattr(torch.version, 'hip') and torch.version.hip and torch.cuda.is_available()
    backends['mps'] = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
    try:
        import intel_extension_for_pytorch
        backends['xpu'] = hasattr(torch, 'xpu') and torch.xpu.is_available()
    except ImportError:
        backends['xpu'] = False
    try:
        import torch_directml
        backends['directml'] = torch_directml.is_available()
    except ImportError:
        backends['directml'] = False
    backends['cpu'] = True
    return backends


def get_best_default_backend():
    avail = get_available_backends()
    for bk in ['cuda', 'rocm', 'mps', 'xpu', 'directml', 'cpu']:
        if avail.get(bk):
            return bk
    return 'cpu'


def detect_training_backend(force=None):
    if force and force != "auto":
        n = force.lower()
        if n == "cpu":
            return torch.device('cpu'), "CPU"
        if n == "cuda" and torch.cuda.is_available():
            return torch.device('cuda'), "CUDA"
        if n == "rocm" and hasattr(torch.version, 'hip') and torch.version.hip and torch.cuda.is_available():
            return torch.device('cuda'), "ROCM"
        if n == "mps" and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.device('mps'), "MPS"
        if n == "xpu":
            try:
                import intel_extension_for_pytorch
                if hasattr(torch, 'xpu') and torch.xpu.is_available():
                    return torch.device('xpu'), "XPU"
            except ImportError:
                pass
        if n == "directml":
            try:
                import torch_directml
                if torch_directml.is_available():
                    return torch_directml.device(0), "DIRECTML"
            except ImportError:
                pass
    if torch.cuda.is_available() and not (hasattr(torch.version, 'hip') and torch.version.hip):
        return torch.device('cuda'), "CUDA"
    if hasattr(torch.version, 'hip') and torch.version.hip and torch.cuda.is_available():
        return torch.device('cuda'), "ROCM"
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps'), "MPS"
    try:
        import intel_extension_for_pytorch
        if hasattr(torch, 'xpu') and torch.xpu.is_available():
            return torch.device('xpu'), "XPU"
    except ImportError:
        pass
    try:
        import torch_directml
        if torch_directml.is_available():
            return torch_directml.device(0), "DIRECTML"
    except ImportError:
        pass
    return torch.device('cpu'), "CPU"


def auto_detect_vram_budget():
    global train_backend
    if train_backend in ("CUDA", "ROCM") and torch.cuda.is_available():
        try:
            fb, tb = torch.cuda.mem_get_info(0)
            memory_config.update({"ram_gb": round(max(1.0, min(fb / 1024 ** 3, tb / 1024 ** 3 - 2.0) - 1.0), 2), "is_auto_detected": True})
            return
        except Exception:
            pass
    if train_backend == "MPS":
        try:
            out = subprocess.check_output(["sysctl", "-n", "hw.memsize"]).decode().strip()
            total_gb = int(out) / 1024 ** 3
            memory_config.update({"ram_gb": round(max(1.0, total_gb * 0.75 - 2.0), 2), "is_auto_detected": True})
            return
        except Exception:
            pass
    try:
        import psutil
        memory_config.update({"ram_gb": round(max(1.0, psutil.virtual_memory().available / 1024 ** 3 - 2.0), 2), "is_auto_detected": True})
    except Exception:
        pass


def has_bitsandbytes():
    try:
        import bitsandbytes
        return True
    except Exception:
        return False


def estimate_vram_bytes(bs, sl, u8):
    mb = EXPECTED_MODEL_SIZE * 2
    ob = EXPECTED_MODEL_SIZE * (4 if u8 else 12)
    gb = EXPECTED_MODEL_SIZE * 4
    ab = N_LAYERS * (bs * sl * DIM * 2 + bs * N_HEADS * 64 * sl * 4)
    lb = bs * LOSS_CHUNK * VOCAB_SIZE * 4
    eb = bs * sl * DIM * 2
    return int(mb + ob + gb + ab + lb + eb + 0.8 * 1024 ** 3)


def recommend_batch_size(sl=2048):
    u8 = has_bitsandbytes()
    budget = int(memory_config["ram_gb"] * 1024 ** 3)
    best = 1
    for bs in ALLOWED_BATCH_SIZES:
        if estimate_vram_bytes(bs, sl, u8) <= budget:
            best = bs
        else:
            break
    return best


# ============================================================
# MODEL
# ============================================================
def precompute_freqs(dim, sl, dev):
    inv = 1.0 / (10000.0 ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    f = torch.einsum("i,j->ij", torch.arange(sl, dtype=torch.float32), inv)
    e = torch.cat((f, f), -1)
    return e.cos()[None, None, :, :].to(dev), e.sin()[None, None, :, :].to(dev)


def rotate_half(x):
    return torch.cat((-x[..., x.shape[-1] // 2:], x[..., :x.shape[-1] // 2]), -1)


class GroupedQueryAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.nh = N_HEADS
        self.nkv = N_KV_HEADS
        self.nrep = N_HEADS // N_KV_HEADS
        self.wq = nn.Linear(DIM, N_HEADS * HEAD_DIM, bias=False)
        self.wk = nn.Linear(DIM, N_KV_HEADS * HEAD_DIM, bias=False)
        self.wv = nn.Linear(DIM, N_KV_HEADS * HEAD_DIM, bias=False)
        self.wo = nn.Linear(DIM, DIM, bias=False)

    def forward(self, x, cos, sin, use_chunked=True):
        B, T, C = x.size()
        q = self.wq(x).view(B, T, self.nh, HEAD_DIM).transpose(1, 2)
        k = self.wk(x).view(B, T, self.nkv, HEAD_DIM).transpose(1, 2)
        v = self.wv(x).view(B, T, self.nkv, HEAD_DIM).transpose(1, 2)
        ct, st = cos[:, :, :T, :], sin[:, :, :T, :]
        q = q * ct + rotate_half(q) * st
        k = k * ct + rotate_half(k) * st
        k = k.unsqueeze(2).expand(B, self.nkv, self.nrep, T, HEAD_DIM).reshape(B, self.nh, T, HEAD_DIM)
        v = v.unsqueeze(2).expand(B, self.nkv, self.nrep, T, HEAD_DIM).reshape(B, self.nh, T, HEAD_DIM)
        if use_chunked:
            return self._chunked(q, k, v, B, T, C)
        a = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(HEAD_DIM))
        a = a.masked_fill(torch.tril(torch.ones(T, T, device=x.device)).view(1, 1, T, T) == 0, float('-inf'))
        a = F.softmax(a, dim=-1, dtype=torch.float32).to(q.dtype)
        return self.wo((a @ v).transpose(1, 2).contiguous().view(B, T, C))

    def _chunked(self, q, k, v, B, T, C, cs=64):
        chunks = []
        for i in range(0, T, cs):
            e = min(i + cs, T)
            aw = (q[:, :, i:e, :] @ k[:, :, :e, :].transpose(-2, -1)) * (1.0 / math.sqrt(HEAD_DIM))
            ri = torch.arange(i, e, device=q.device).unsqueeze(1)
            ci = torch.arange(e, device=q.device).unsqueeze(0)
            aw = aw.masked_fill(~(ci <= ri).unsqueeze(0).unsqueeze(0), float('-inf'))
            chunks.append(F.softmax(aw, dim=-1, dtype=torch.float32).to(q.dtype) @ v[:, :, :e, :])
        return self.wo(torch.cat(chunks, dim=2).transpose(1, 2).contiguous().view(B, T, C))


class SwiGLU(nn.Module):
    def __init__(self):
        super().__init__()
        self.w1 = nn.Linear(DIM, MLP_HIDDEN, bias=False)
        self.w2 = nn.Linear(DIM, MLP_HIDDEN, bias=False)
        self.w3 = nn.Linear(MLP_HIDDEN, DIM, bias=False)

    def forward(self, x):
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln_1 = nn.LayerNorm(DIM)
        self.attn = GroupedQueryAttention()
        self.ln_2 = nn.LayerNorm(DIM)
        self.mlp = SwiGLU()

    def forward(self, x, cos, sin, use_chunked=True):
        x = x + self.attn(self.ln_1(x), cos, sin, use_chunked)
        return x + self.mlp(self.ln_2(x))


class EngramMemory(nn.Module):
    def __init__(self, bn, device):
        super().__init__()
        self.device = device
        self.table = nn.Embedding(ENG_NUM_BUCKETS, DIM, sparse=True).to('cpu')
        nn.init.normal_(self.table.weight, mean=0.0, std=0.02)
        self.use_async = device.type in ('cuda',) and torch.cuda.is_available()
        self.ts = torch.cuda.Stream(device=device) if self.use_async else None

    def forward(self, idx):
        px = torch.cat([torch.zeros_like(idx[:, :1]), idx[:, :-1]], dim=1)
        hi = (px * 1000003 + idx) % ENG_NUM_BUCKETS
        uq, inv = torch.unique(hi.flatten(), return_inverse=True)
        cr = self.table(uq.cpu())
        if self.use_async:
            with torch.cuda.stream(self.ts):
                cg = cr.to(self.device, non_blocking=True)
            torch.cuda.current_stream(self.device).wait_stream(self.ts)
        else:
            cg = cr.to(self.device)
        return cg[inv].view(idx.shape[0], idx.shape[1], DIM)


class SotaGPT(nn.Module):
    def __init__(self, bn, device):
        super().__init__()
        self.wte = nn.Embedding(VOCAB_SIZE, DIM)
        self.engram = EngramMemory(bn, device)
        self.blocks = nn.ModuleList([Block() for _ in range(N_LAYERS)])
        self.ln_f = nn.LayerNorm(DIM)
        self.lm_head = nn.Linear(DIM, VOCAB_SIZE, bias=False)
        if MODEL_CONFIG["weightTying"]:
            self.wte.weight = self.lm_head.weight
        cm, sm = precompute_freqs(HEAD_DIM, MAX_SEQ_LEN, device)
        self.register_buffer("freqs_cos", cm)
        self.register_buffer("freqs_sin", sm)

    def get_base_weights(self):
        return np.concatenate([
            p.detach().float().flatten().cpu().numpy()
            for n, p in self.named_parameters() if not n.startswith('engram.')
        ])

    def load_base_weights(self, fw):
        ft = torch.from_numpy(fw) if not isinstance(fw, torch.Tensor) else fw
        o = 0
        for n, p in self.named_parameters():
            if not n.startswith('engram.'):
                s = p.numel()
                p.data.copy_(ft[o:o + s].view(p.shape).to(p.device))
                o += s


# ============================================================
# DATASET
# ============================================================
def _fetch_with_retry(url, headers=None, timeout=60, retries=5):
    for a in range(retries):
        try:
            r = requests.get(url, headers=headers, timeout=timeout, allow_redirects=True)
            r.raise_for_status()
            return r
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout,
                ChunkedEncodingError, http.client.IncompleteRead):
            if a < retries - 1:
                time.sleep(2 ** (a + 1))
            else:
                raise


class StreamingShardDataset:
    STEPS_PER_SUBCHUNK = 500

    def __init__(self, repo_id, ci, ss=10 * 1024 * 1024, tps=65, slot=0, at=None):
        self.repo_id = repo_id
        self.ci = ci
        self.sub_size = ss
        self.tps = tps
        self.auth_token = at
        self._fmt = "chunk_{:04d}.bin"
        self.chunk_size = self._discover(ci)
        self.off = (slot * ss) % self.chunk_size
        self.data = None
        self.n = 0
        self.steps_used = 0
        self._pft = None
        self._pfr = None
        self._lock = threading.Lock()
        self._load(self.off)
        self._prefetch(self.off + ss)

    def _url(self):
        return f"https://huggingface.co/datasets/{self.repo_id}/resolve/main/chunks/" + self._fmt.format(self.ci)

    def _discover(self, idx):
        for fmt in ("chunk_{:04d}.bin", "chunk_{:d}.bin"):
            url = f"https://huggingface.co/datasets/{self.repo_id}/resolve/main/chunks/" + fmt.format(idx)
            for _ in range(2):
                try:
                    r = requests.head(url, allow_redirects=True, timeout=30)
                    if r.status_code == 200:
                        sz = int(r.headers.get('content-length', 0))
                        if sz > 0:
                            self._fmt = fmt
                            return sz
                except Exception:
                    time.sleep(1)
        raise RuntimeError(f"No chunk {idx}")

    def _slice(self, off):
        end = min(off + self.sub_size, self.chunk_size) - 1
        return _fetch_with_retry(self._url(), headers={'Range': f'bytes={off}-{end}'}, timeout=120).content

    def _set(self, raw):
        tk = np.frombuffer(raw, dtype=np.uint32)
        self.data = tk
        self.n = len(tk) // self.tps
        self.steps_used = 0

    def _load(self, off):
        self._set(self._slice(off))
        self.off = off

    def _prefetch(self, off):
        if off >= self.chunk_size:
            return
        with self._lock:
            self._pfr = None

        def w():
            try:
                r = self._slice(off)
                with self._lock:
                    self._pfr = (off, r)
            except Exception:
                pass
        self._pft = threading.Thread(target=w, daemon=True)
        self._pft.start()

    def needs_new_subchunk(self):
        return self.n == 0 or self.steps_used >= self.STEPS_PER_SUBCHUNK

    def advance(self, srv=None, fmt="bf16"):
        no = self.off + self.sub_size
        if no < self.chunk_size:
            if self._pft:
                self._pft.join(timeout=120)
                self._pft = None
            with self._lock:
                r = self._pfr
                self._pfr = None
            if r and r[0] == no:
                self.off = no
                self._set(r[1])
            else:
                self._load(no)
            self._prefetch(no + self.sub_size)
            return True
        self._req(srv, fmt)
        self.chunk_size = self._discover(self.ci)
        self.off = 0
        self._load(0)
        self._prefetch(self.sub_size)
        return True

    def _req(self, srv, fmt="bf16"):
        if not srv:
            return False
        try:
            h = {"Authorization": f"Bearer {self.auth_token}"} if self.auth_token else {}
            r = requests.get(f"{srv}/fl/task?format={fmt}&skip_weights=true", headers=h, timeout=30)
            if r.status_code != 200:
                return False
            ct = (r.headers.get("content-type") or "").lower()
            if "json" in ct:
                md = r.json()
            else:
                raw = r.content
                if len(raw) < 4:
                    return False
                ml = struct.unpack('<I', raw[:4])[0]
                if len(raw) < 4 + ml:
                    return False
                md = json.loads(raw[4:4 + ml].decode())
            ni = md.get("datasetConfig", {}).get("chunkIdx", self.ci)
            if ni != self.ci:
                self.ci = ni
                return True
        except Exception:
            pass
        return False

    def get_batch(self, bs, seed=None):
        if self.data is None or self.n == 0:
            self.advance()
        self.steps_used += 1
        rng = np.random.RandomState(seed)
        s = rng.randint(0, self.n, size=bs) * self.tps
        inp = np.stack([self.data[x:x + self.tps - 1] for x in s])
        tgt = np.stack([self.data[x + 1:x + self.tps] for x in s])
        return torch.tensor(inp, dtype=torch.long), torch.tensor(tgt, dtype=torch.long)


def decompress_weights(raw, fmt="bf16"):
    if fmt == "fp16":
        return np.frombuffer(raw, dtype=np.uint16).view(np.float16).astype(np.float32)
    return torch.from_numpy(np.frombuffer(raw, dtype=np.uint16).copy()).view(torch.bfloat16).to(torch.float32).numpy()


# ============================================================
# AUTH
# ============================================================
def do_login(srv, u, p):
    try:
        r = requests.post(f"{srv}/auth/login", json={"username": u, "password": p}, timeout=30)
        if r.status_code == 200:
            return r.json().get("token"), None
        try:
            msg = r.json().get("detail", "")
        except Exception:
            msg = r.text
        if "invalid" in msg.lower() or "credentials" in msg.lower():
            return None, "invalid_credentials"
        return None, "login_failed"
    except requests.exceptions.ConnectionError:
        return None, "network"
    except Exception:
        return None, "network"


def do_register(srv, u, email, p):
    try:
        r = requests.post(f"{srv}/auth/register", json={"username": u, "password": p, "email": email}, timeout=30)
        if r.status_code == 200:
            return r.json().get("token"), None
        try:
            msg = r.json().get("detail", "")
        except Exception:
            msg = r.text
        msg_l = msg.lower()
        if "username" in msg_l and "taken" in msg_l:
            return None, "username_taken"
        if "email" in msg_l and "registered" in msg_l:
            return None, "email_taken"
        if "password" in msg_l:
            return None, "weak_password"
        if "invalid" in msg_l:
            return None, "invalid_input"
        return None, "register_failed"
    except requests.exceptions.ConnectionError:
        return None, "network"
    except Exception:
        return None, "network"


# ============================================================
# HEARTBEAT / ROUND
# ============================================================
class HeartbeatManager:
    def __init__(self, srv, h, cr, se, emit):
        self.srv = srv
        self.h = h
        self.cr = cr
        self.stop_training = threading.Event()
        self.shutdown = threading.Event()
        self.se = se
        self.emit = emit
        self._t = threading.Thread(target=self._r, daemon=True)

    def start(self):
        self._t.start()

    def stop(self):
        self.shutdown.set()

    def should_stop(self):
        return self.stop_training.is_set()

    def _r(self):
        while not self.shutdown.is_set() and not self.se.is_set():
            try:
                requests.get(f"{self.srv}/fl/heartbeat", headers=self.h, timeout=10)
                r = requests.get(f"{self.srv}/fl/round_status", headers=self.h, timeout=10)
                if r.status_code == 200:
                    st = r.json()
                    if st.get("current_round", self.cr) != self.cr:
                        self.emit('log', "Next round. Stopping.")
                        self.stop_training.set()
                    rm = (st.get("max_round_hours", 2) - st.get("round_elapsed_hours", 0)) * 60
                    if rm <= 3:
                        self.emit('log', f"ULTIMATUM: {rm:.0f} min left!")
                        self.stop_training.set()
            except Exception:
                pass
            self.shutdown.wait(timeout=15)


def wait_for_round(srv, h, se, emit):
    overlay_on = False

    def show_wait():
        nonlocal overlay_on
        if not overlay_on:
            emit('overlay_show', {'title_key': 'wait_coord', 'subtitle_key': 'wait_coord_sub', 'indeterminate': True})
            overlay_on = True

    def hide():
        nonlocal overlay_on
        if overlay_on:
            emit('overlay_hide', {})
            overlay_on = False

    while not se.is_set():
        try:
            r = requests.get(f"{srv}/fl/round_status", headers=h, timeout=30)
            if r.status_code == 200:
                st = r.json()
                if st.get("is_aggregating"):
                    emit('status', 'waiting')
                    show_wait()
                    for _ in range(30):
                        if se.is_set():
                            hide()
                            return None
                        time.sleep(1)
                    continue
                if not st.get("in_cooldown", False):
                    hide()
                    return st
                emit('status', 'waiting')
                show_wait()
                for _ in range(30):
                    if se.is_set():
                        hide()
                        return None
                    time.sleep(1)
            else:
                time.sleep(10)
        except Exception:
            time.sleep(10)
    hide()
    return None


# ============================================================
# TASK METADATA / HF WEIGHTS
# ============================================================
def fetch_task_metadata(srv, h, se, emit):
    while not se.is_set():
        try:
            r = requests.get(f"{srv}/fl/task?format=bf16&skip_weights=true", headers=h, timeout=30)

            if r.headers.get("X-Status") == "wait":
                for _ in range(10):
                    if se.is_set():
                        return None
                    time.sleep(1)
                continue

            if r.status_code == 200:
                ct = (r.headers.get("content-type") or "").lower()
                if "json" in ct:
                    md = r.json()
                    if md.get("status") == "wait":
                        for _ in range(10):
                            if se.is_set():
                                return None
                            time.sleep(1)
                        continue
                    return md
                raw = r.content
                if len(raw) >= 4:
                    ml = struct.unpack('<I', raw[:4])[0]
                    if len(raw) >= 4 + ml:
                        return json.loads(raw[4:4 + ml].decode())
                raise RuntimeError("Invalid task metadata response")

            if r.status_code in (503, 504):
                time.sleep(10)
                continue
            time.sleep(5)

        except Exception as e:
            if se.is_set():
                return None
            emit('log', f"Task metadata fetch failed: {e}")
            time.sleep(10)
    return None


def _weight_cache_path(revision, filename):
    rev = str(revision or "main").strip().replace("/", "_").replace("\\", "_")
    d = CHECKPOINT_DIR / "hf_weights" / rev
    d.mkdir(parents=True, exist_ok=True)
    return d / filename


def _download_file_with_progress(url, dest, emit, stop_evt, expected_size=None, label="file"):
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists():
        sz = dest.stat().st_size
        if expected_size is not None and sz == int(expected_size):
            log.info(f"Using cached weight file: {dest} ({sz} bytes)")
            return dest
        if expected_size is None and sz > 10_000_000:
            log.info(f"Using cached weight file: {dest} ({sz} bytes, unverified)")
            return dest
        log.info(f"Cached file {dest} is {sz} bytes, expected {expected_size}; re-downloading")
        try:
            dest.unlink()
        except Exception:
            pass

    last_err = None
    for attempt in range(3):
        if stop_evt.is_set():
            raise KeyboardInterrupt("Stop requested during weight download")
        tmp = dest.with_suffix(dest.suffix + ".part")
        try:
            if tmp.exists():
                tmp.unlink()
        except Exception:
            pass
        try:
            log.info(f"Downloading {label}: {url}")
            r = requests.get(url, stream=True, timeout=(15, 3600), allow_redirects=True)
            r.raise_for_status()
            content_length = int(r.headers.get("content-length", 0) or 0)
            total = int(expected_size) if expected_size else content_length
            done = 0
            last_emit = 0.0
            with open(tmp, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    if stop_evt.is_set():
                        raise KeyboardInterrupt("Stop requested during weight download")
                    if not chunk:
                        continue
                    f.write(chunk)
                    done += len(chunk)
                    now = time.time()
                    if now - last_emit > 0.25:
                        emit("overlay_progress", {"done": done, "total": total})
                        last_emit = now
            if expected_size and done != int(expected_size):
                raise RuntimeError(f"Downloaded size mismatch for {label}: got {done}, expected {expected_size}")
            os.replace(tmp, dest)
            log.info(f"Finished downloading {label}: {dest}")
            return dest
        except KeyboardInterrupt:
            raise
        except Exception as e:
            last_err = e
            try:
                if tmp.exists():
                    tmp.unlink()
            except Exception:
                pass
            if attempt < 2:
                emit("log", f"Download retry {attempt + 1} for {label}: {e}")
                time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"Failed to download {label}: {last_err}")


def _load_safetensors_flat(path, expected_numel, label="weights"):
    try:
        from safetensors.torch import load_file
    except ImportError as e:
        raise RuntimeError("Missing dependency: safetensors. Add 'safetensors' to requirements.") from e

    tensors = load_file(str(path))

    if "weights" in tensors:
        t = tensors["weights"]
    elif "flat" in tensors:
        t = tensors["flat"]
    elif len(tensors) == 1:
        t = next(iter(tensors.values()))
    else:
        raise RuntimeError(
            f"{label}: could not identify expected tensor key. "
            f"Expected 'weights' or 'flat'. Found keys: {list(tensors.keys())[:20]}"
        )

    # Force fully-owned, contiguous float32 copy — no aliasing to mmap.
    arr = t.to(torch.float32).contiguous().clone().cpu().numpy()
    del tensors, t
    gc.collect()

    if arr.size != int(expected_numel):
        raise RuntimeError(f"{label} element count mismatch: got {arr.size}, expected {expected_numel}.")

    # Canary read to catch freed-storage aliasing.
    _ = float(arr[0]) + float(arr[-1])
    return arr


def fetch_weights_from_hf(md, se, emit):
    wcfg = md.get("weights") or {}
    if wcfg.get("source") != "huggingface":
        raise RuntimeError("Task metadata does not advertise Hugging Face weights.")

    revision = wcfg.get("revision") or "main"
    files = wcfg.get("files") or {}
    model_info = files.get("model") or {}
    engram_info = files.get("engram") or {}

    model_url = model_info.get("url") or (
        f"https://huggingface.co/{HF_MODEL_REPO_ID}/resolve/{revision}/model_bf16.safetensors"
    )
    engram_url = engram_info.get("url") or (
        f"https://huggingface.co/{HF_MODEL_REPO_ID}/resolve/{revision}/engram_bf16.safetensors"
    )
    model_size = model_info.get("size")
    engram_size = engram_info.get("size")

    if not model_size:
        try:
            hr = requests.head(model_url, allow_redirects=True, timeout=30)
            model_size = int(hr.headers.get("content-length", 0) or 0) or None
        except Exception:
            model_size = None
    if not engram_size:
        try:
            hr = requests.head(engram_url, allow_redirects=True, timeout=30)
            engram_size = int(hr.headers.get("content-length", 0) or 0) or None
        except Exception:
            engram_size = None

    model_path = _weight_cache_path(revision, "model_bf16.safetensors")
    engram_path = _weight_cache_path(revision, "engram_bf16.safetensors")

    emit("status", "downloading")
    emit("overlay_show", {"title_key": "dl_weights", "indeterminate": False})

    _download_file_with_progress(model_url, model_path, emit, se, expected_size=model_size, label="model_bf16.safetensors")
    _download_file_with_progress(engram_url, engram_path, emit, se, expected_size=engram_size, label="engram_bf16.safetensors")

    emit("overlay_hide", {})
    emit("status", "preparing")
    emit("log", "Loading weights from safetensors...")

    iw = _load_safetensors_flat(model_path, EXPECTED_MODEL_SIZE, "model weights")
    ie = _load_safetensors_flat(engram_path, EXPECTED_ENGRAM_SIZE, "engram weights")

    return iw, ie


def fetch_task(srv, h, prec, se, emit):
    """Inline fallback: download weights from coordinator. Hard size check."""
    while not se.is_set():
        try:
            emit('status', 'downloading')
            emit('overlay_show', {'title_key': 'dl_weights', 'indeterminate': False})
            r = requests.get(f"{srv}/fl/task?format={prec}", headers=h, timeout=(15, 3600), stream=True)

            if r.status_code != 200:
                r.close()
                emit('overlay_hide', {})
                emit('log', f"/fl/task HTTP {r.status_code}: {r.text[:120]}")
                time.sleep(10)
                continue

            if r.headers.get("X-Status") == "wait":
                r.close()
                emit('overlay_hide', {})
                for _ in range(10):
                    if se.is_set():
                        return None, None
                    time.sleep(1)
                continue

            cl_total = int(r.headers.get('content-length', 0) or 0)
            total = cl_total if cl_total > 0 else EXPECTED_WEIGHT_BYTES
            chunks = []
            done = 0
            last = 0.0
            for chunk in r.iter_content(chunk_size=512 * 1024):
                if not chunk:
                    continue
                chunks.append(chunk)
                done += len(chunk)
                now = time.time()
                if now - last > 0.25:
                    emit('overlay_progress', {'done': done, 'total': total})
                    last = now
            r.close()
            emit('overlay_progress', {'done': done, 'total': total})
            emit('overlay_hide', {})
            emit('status', 'preparing')

            raw = b"".join(chunks)
            del chunks

            if len(raw) < 4:
                emit('log', f"/fl/task returned only {len(raw)} bytes; retrying")
                time.sleep(5)
                continue

            ml = struct.unpack('<I', raw[:4])[0]
            if 4 + ml > len(raw):
                emit('log', f"Truncated header (ml={ml}, len={len(raw)}); retrying")
                time.sleep(5)
                continue

            md = json.loads(raw[4:4 + ml].decode())
            wb = raw[4 + ml:]

            emit('log', f"/fl/task inline: metadata OK, weights={len(wb):,} bytes")

            if len(wb) < EXPECTED_WEIGHT_BYTES:
                emit('log', f"⚠️ Inline stream truncated at {len(wb):,} (need {EXPECTED_WEIGHT_BYTES:,}); retrying")
                time.sleep(10)
                continue

            return md, wb

        except Exception as e:
            emit('overlay_hide', {})
            if se.is_set():
                return None, None
            emit('log', f"Task fetch failed: {e}")
            time.sleep(15)


# ============================================================
# FORWARD / LOSS
# ============================================================
def _chunk_ce(m, xs, ys):
    return F.cross_entropy(m.lm_head(xs).reshape(-1, VOCAB_SIZE), ys.reshape(-1))


def _fwl(m, x, y, sl, ua, ad, ls=1.0, preview=False):
    def fwd():
        xe = m.wte(x)
        if ua:
            with torch.autocast(device_type='cuda', enabled=False):
                eo = m.engram(x)
        else:
            eo = m.engram(x)
        xe = xe + eo.to(xe.dtype)
        for b in m.blocks:
            xe = checkpoint(b, xe, m.freqs_cos, m.freqs_sin, True, use_reentrant=False)
        return m.ln_f(xe)

    nc = max(1, math.ceil(sl / LOSS_CHUNK))
    if ua:
        with torch.autocast(device_type='cuda', dtype=ad):
            xe = fwd()
            lo = sum(
                checkpoint(_chunk_ce, m, xe[:, i:i + LOSS_CHUNK, :], y[:, i:i + LOSS_CHUNK], use_reentrant=False)
                for i in range(0, sl, LOSS_CHUNK)
            ) / nc
    else:
        xe = fwd()
        lo = sum(
            checkpoint(_chunk_ce, m, xe[:, i:i + LOSS_CHUNK, :], y[:, i:i + LOSS_CHUNK], use_reentrant=False)
            for i in range(0, sl, LOSS_CHUNK)
        ) / nc

    cards = []
    if preview:
        try:
            with torch.no_grad():
                B, T, C = xe.shape
                items = []
                for b in range(min(B, 3)):
                    items.append((b, T - 1))
                for offset in (24, 48, 72):
                    p = T - offset - 1
                    if p >= 9:
                        items.append((0, p))
                items = items[:6]

                for b, p in items:
                    if p < 9 or p >= T:
                        continue
                    start = max(0, p - 9)
                    ctx_ids = x[b, start:p + 1].tolist()
                    target_id = int(y[b, p].item())
                    hidden = xe[b, p, :].unsqueeze(0).to(m.lm_head.weight.dtype)
                    logits = m.lm_head(hidden).float()
                    probs = torch.softmax(logits[0], dim=-1)
                    prob, pred = torch.max(probs, dim=0)
                    pred_id = int(pred.item())
                    cards.append({
                        'context': decode_ids(ctx_ids),
                        'target': decode_one(target_id),
                        'pred': decode_one(pred_id),
                        'match': bool(pred_id == target_id),
                        'prob': safe_float(prob.item()),
                        'target_id': int(target_id),
                        'pred_id': int(pred_id),
                    })
        except Exception as e:
            log.warning(f"Preview extraction failed: {e}")
            cards = []

    return lo * ls, cards


# ============================================================
# CHUNKED UPLOAD
# ============================================================
def _sha256_hex(data):
    return hashlib.sha256(data).hexdigest()


def _post_json_with_retry(url, payload, h, emit, se, timeout=60, retries=4):
    last_err = None
    for attempt in range(retries):
        if se.is_set():
            raise KeyboardInterrupt("Stop requested")
        try:
            r = requests.post(url, json=payload, headers=h, timeout=timeout)
            if r.status_code == 503:
                emit("log", "Server busy/aggregating, waiting before retry...")
                last_err = RuntimeError("server busy 503")
                time.sleep(10)
                continue
            r.raise_for_status()
            return r
        except KeyboardInterrupt:
            raise
        except Exception as e:
            last_err = e
            if attempt < retries - 1:
                emit("log", f"POST retry {attempt + 1}: {e}")
                time.sleep(min(20, 2 ** (attempt + 1)))
    raise RuntimeError(f"POST {url} failed: {last_err}")


def _chunked_submit_once(srv, h, compressed_data, meta, emit, se, chunk_size):
    total = len(compressed_data)
    chunks = max(1, math.ceil(total / chunk_size))
    checksum = _sha256_hex(compressed_data)

    init_payload = {
        **meta,
        "totalBytes": total,
        "chunkSize": chunk_size,
        "chunks": chunks,
        "checksum": checksum,
    }

    emit("log", f"Initializing chunked upload: {chunks} chunks, {total / (1024 * 1024):.1f} MB total")

    r = _post_json_with_retry(f"{srv}/fl/submit/init", init_payload, h, emit, se, timeout=60, retries=4)
    j = r.json()
    upload_id = j.get("uploadId")
    if not upload_id:
        raise RuntimeError("Server did not return uploadId")

    for idx in range(chunks):
        if se.is_set():
            raise KeyboardInterrupt("Stop requested during upload")
        start = idx * chunk_size
        end = min(total, start + chunk_size)
        part = compressed_data[start:end]
        part_sha = _sha256_hex(part)

        attempt = 0
        while True:
            try:
                rr = requests.post(
                    f"{srv}/fl/submit/chunk?uploadId={upload_id}&index={idx}",
                    data=part,
                    headers={**h, "Content-Type": "application/octet-stream", "X-Chunk-Sha256": part_sha},
                    timeout=300,
                )
                if rr.status_code == 413:
                    raise RuntimeError("CHUNK_TOO_LARGE")
                rr.raise_for_status()
                break
            except KeyboardInterrupt:
                raise
            except Exception as e:
                if "CHUNK_TOO_LARGE" in str(e):
                    raise
                attempt += 1
                if se.is_set() or attempt >= MAX_CHUNK_RETRY:
                    raise
                emit("log", f"Chunk {idx + 1}/{chunks} failed, retry {attempt}: {e}")
                time.sleep(min(20, 2 ** attempt))

        emit("overlay_progress", {"done": end, "total": total})
        emit("log", f"  ✓ chunk {idx + 1}/{chunks} ({end / (1024 * 1024):.1f}/{total / (1024 * 1024):.1f} MB)")

    emit("log", "Finalizing chunked upload...")
    return _post_json_with_retry(f"{srv}/fl/submit/finalize", {"uploadId": upload_id}, h, emit, se, timeout=900, retries=2)


def chunked_submit_delta(srv, h, compressed_data, meta, emit, se):
    chunk_size = UPLOAD_CHUNK_BYTES
    while True:
        try:
            return _chunked_submit_once(srv, h, compressed_data, meta, emit, se, chunk_size)
        except Exception as e:
            msg = str(e)
            if ("CHUNK_TOO_LARGE" in msg or "413" in msg) and chunk_size > 8 * 1024 * 1024:
                chunk_size = max(8 * 1024 * 1024, chunk_size // 2)
                emit("log", f"Chunk too large. Reducing chunk size to {chunk_size / (1024 * 1024):.0f} MB and retrying.")
                continue
            raise


def stream_with_progress(data, emit_cb, total_size, stop_evt):
    chunk_size = 128 * 1024
    uploaded = 0
    last_emit = 0.0
    for i in range(0, len(data), chunk_size):
        if stop_evt.is_set():
            raise KeyboardInterrupt("Stop requested")
        chunk = data[i:i + chunk_size]
        uploaded += len(chunk)
        now = time.time()
        if now - last_emit > 0.25:
            emit_cb("overlay_progress", {"done": uploaded, "total": total_size})
            last_emit = now
        yield chunk
    emit_cb("overlay_progress", {"done": total_size, "total": total_size})


# ============================================================
# TRAIN ROUND
# ============================================================
def run_single_round_wrapper(srv, at, se, emit):
    global train_device, train_backend
    h = {"Authorization": f"Bearer {at}"} if at else {}
    emit('status', 'waiting')
    rs = wait_for_round(srv, h, se, emit)
    if not rs:
        return

    cr = rs["current_round"]
    hb = HeartbeatManager(srv, h, cr, se, emit)
    hb.start()

    model = ob = oe = x = y = lo = iw = ie = iew = ds = None
    micro_step = 0
    lval = 10.0
    tt = 0
    tid = ""

    try:
        emit('log', "Fetching task metadata...")
        md = fetch_task_metadata(srv, h, se, emit)
        if md is None:
            return

        tid = md.get("taskId")
        gs = md.get("globalStep")
        sl = MAX_SEQ_LEN

        if not tid:
            emit("log", "Task metadata missing taskId.")
            return

        iw = ie = None

        if USE_HF_WEIGHTS:
            wcfg = md.get("weights") or {}
            if wcfg.get("source") == "huggingface":
                try:
                    emit('log', "Downloading weights from Hugging Face...")
                    iw, ie = fetch_weights_from_hf(md, se, emit)
                except KeyboardInterrupt:
                    raise
                except Exception as e:
                    import traceback as _tb
                    emit('log', f"Hugging Face weight download failed: {e}")
                    print(f"[HF-FAIL-TRACEBACK]\n{_tb.format_exc()}\n", flush=True)
                    emit('log', "Falling back to coordinator weight stream...")
                    iw = ie = None

        if iw is None or ie is None:
            emit('log', "Downloading weights from coordinator...")
            md2, wb = fetch_task(srv, h, "bf16", se, emit)
            if md2 is None:
                return
            md = md2
            tid = md.get("taskId")
            gs = md.get("globalStep")
            if not tid:
                emit("log", "Coordinator task metadata missing taskId.")
                return
            bbl = EXPECTED_MODEL_SIZE * 2
            ebl = EXPECTED_ENGRAM_SIZE * 2
            needed = bbl + ebl
            if len(wb) < needed:
                raise RuntimeError(f"Incomplete coordinator weight payload: got {len(wb)}, expected {needed}")
            iw = decompress_weights(wb[:bbl], "bf16")
            ie = decompress_weights(wb[bbl:bbl + ebl], "bf16")
            del wb
            gc.collect()

        emit('status', 'preparing')
        emit('log', "Preparing model...")

        dc = md.get("datasetConfig", {})
        sc = md.get("shardConfig", {})
        ds = StreamingShardDataset(
            dc.get("repoId", DATASET_REPO_ID),
            dc.get("chunkIdx", 0),
            dc.get("subChunkSize", 10 * 1024 * 1024),
            dc.get("tokensPerSample", sl + 1),
            sc.get("slot", 0),
            at,
        )

        bs = recommend_batch_size(sl)

        while bs >= 1:
            try:
                if train_backend in ("CUDA", "ROCM"):
                    try:
                        torch.cuda.empty_cache()
                    except Exception:
                        pass
                gc.collect()

                model = SotaGPT(train_backend, train_device).to(train_device)
                model.engram.table.to('cpu')

                # Debug probe
                try:
                    print(f"[IW-PROBE] iw: type={type(iw).__name__} "
                          f"shape={getattr(iw, 'shape', None)} "
                          f"size={getattr(iw, 'size', None)} "
                          f"nbytes={getattr(iw, 'nbytes', None)}", flush=True)
                    print(f"[IW-PROBE] ie: type={type(ie).__name__} "
                          f"shape={getattr(ie, 'shape', None)} "
                          f"size={getattr(ie, 'size', None)}", flush=True)
                except Exception as _pe:
                    print(f"[IW-PROBE] probe failed: {_pe}", flush=True)

                model.load_base_weights(iw)
                model.engram.table.weight.data.copy_(torch.from_numpy(ie).view(model.engram.table.weight.shape))
                model.train()
                break
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    if model:
                        del model
                        model = None
                    if train_backend in ("CUDA", "ROCM"):
                        try:
                            torch.cuda.empty_cache()
                        except Exception:
                            pass
                    gc.collect()
                    bs = max(1, bs // 2)
                else:
                    raise

        if model is None:
            emit('log', "Failed to initialize model.")
            return

        mbt = bs * sl
        as_ = max(1, round(131072 / mbt))
        ls = 1.0 / as_

        def ha():
            nonlocal as_, ls
            as_ = max(1, as_ // 2)
            ls = 1.0 / as_
            ob.zero_grad(set_to_none=True)
            gc.collect()
            if train_backend in ("CUDA", "ROCM"):
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
            emit('log', f"OOM -> accum now {as_}x")

        bp = [p for n, p in model.named_parameters() if not n.startswith('engram.')]
        ep = list(model.engram.parameters())

        try:
            import bitsandbytes as bnb
            ob = bnb.optim.AdamW8bit(bp, lr=1e-4, betas=(0.9, 0.95), weight_decay=0.01)
        except Exception:
            ob = torch.optim.AdamW(bp, lr=1e-4, betas=(0.9, 0.95), weight_decay=0.01)

        try:
            oe = torch.optim.SparseAdam(ep, lr=1e-4)
        except Exception:
            oe = torch.optim.AdamW(ep, lr=1e-4, weight_decay=0.01)

        iew = model.engram.table.weight.data.cpu().clone()
        ua = train_backend in ("CUDA", "ROCM") and torch.cuda.is_bf16_supported()
        ad = torch.bfloat16 if ua else None

        emit('log', "Calibrating speed...")
        cs_ = time.time()
        cal_global_steps = 3
        target_micro_steps = 100
        emit('status', 'calibrating')
        cal_done = 0

        while cal_done < cal_global_steps:
            if hb.should_stop() or se.is_set():
                return
            ob.zero_grad(set_to_none=True)
            try:
                for mi in range(as_):
                    if hb.should_stop() or se.is_set():
                        break
                    if ds.needs_new_subchunk():
                        ds.advance(srv, "bf16")
                    sd = (abs(hash("cal")) % 10000) + cal_done * 1000 + mi
                    x, y = ds.get_batch(bs, seed=sd)
                    x, y = x.to(train_device), y.to(train_device)

                    want_preview = (micro_step == 0) or ((micro_step + 1) % PREVIEW_EVERY == 0)
                    lo, cards = _fwl(model, x, y, sl, ua, ad, ls, preview=want_preview)
                    lval = float(lo.item())
                    micro_step += 1

                    if cards:
                        emit('model_preview', {'step': micro_step, 'cards': cards})

                    if cal_done >= 1 and mi == 0:
                        elapsed = time.time() - cs_
                        sps_micro = elapsed / micro_step
                        rh = max(0.1, rs.get("max_round_hours", 2.0) - rs.get("round_elapsed_hours", 0))
                        remaining = max(60, (rh * 3600) - elapsed - (UPLOAD_BUFFER_MIN * 60))
                        target_micro_steps = micro_step + int(remaining / sps_micro)

                    elapsed_cal = time.time() - cs_
                    current_tps = tt / max(elapsed_cal, 1.0)

                    time_left_mins = 0
                    if cal_done >= 1 and micro_step > 0:
                        elapsed = time.time() - cs_
                        sps_micro = elapsed / micro_step
                        rh = max(0.1, rs.get("max_round_hours", 2.0) - rs.get("round_elapsed_hours", 0))
                        remaining = max(60, (rh * 3600) - elapsed - (UPLOAD_BUFFER_MIN * 60))
                        time_left_mins = remaining / 60.0

                    emit('cal_stats', {
                        'step': micro_step,
                        'total': target_micro_steps,
                        'loss': safe_float(lval * as_, 10.0),
                        'microbatch': micro_step,
                        'tps': safe_float(current_tps),
                        'time_left': safe_float(time_left_mins),
                    })

                    lo.backward()
                    oe.step()
                    oe.zero_grad(set_to_none=True)
                    tt += x.numel()
                    del lo, x, y

                if hb.should_stop() or se.is_set():
                    return

            except RuntimeError as e:
                if "out of memory" not in str(e).lower():
                    raise
                ha()
                continue

            torch.nn.utils.clip_grad_norm_(bp, 1.0)
            ob.step()
            ob.zero_grad(set_to_none=True)
            gc.collect()
            cal_done += 1

        if se.is_set():
            return

        rh = max(0.1, rs.get("max_round_hours", 2.0) - rs.get("round_elapsed_hours", 0))
        remaining = max(60, (rh * 3600) - (time.time() - cs_) - (UPLOAD_BUFFER_MIN * 60))
        dl = time.time() + remaining

        if micro_step > 0:
            sps_micro = (time.time() - cs_) / micro_step
            target_micro_steps = micro_step + int(remaining / sps_micro)
        else:
            target_micro_steps = 1000

        emit('status', 'training')
        emit('log', f"Target: ~{target_micro_steps} micro-steps")

        last_stats_emit = 0.0

        while time.time() < dl:
            if hb.should_stop() or se.is_set():
                break
            ob.zero_grad(set_to_none=True)
            als = 0.0
            cc = 0
            try:
                for mi in range(as_):
                    if hb.should_stop() or se.is_set() or time.time() >= dl:
                        break
                    if ds.needs_new_subchunk():
                        ds.advance(srv, "bf16")
                    sd = (abs(hash("t")) % 10000) + micro_step * 1000 + mi
                    x, y = ds.get_batch(bs, seed=sd)
                    x, y = x.to(train_device), y.to(train_device)

                    want_preview = (micro_step == 0) or ((micro_step + 1) % PREVIEW_EVERY == 0)
                    lo, cards = _fwl(model, x, y, sl, ua, ad, ls, preview=want_preview)
                    lval = float(lo.item()) * as_
                    lo.backward()
                    oe.step()
                    oe.zero_grad(set_to_none=True)

                    als += lval
                    tt += x.numel()
                    cc += 1
                    micro_step += 1

                    if cards:
                        emit('model_preview', {'step': micro_step, 'cards': cards})

                    rem_time = max(1, dl - time.time())
                    elapsed_total = time.time() - cs_
                    if micro_step > 5:
                        sps_micro = elapsed_total / micro_step
                        future_steps = int(rem_time / sps_micro)
                        current_target = micro_step + future_steps
                        if current_target > target_micro_steps:
                            target_micro_steps = current_target

                    if time.time() - last_stats_emit > 0.25 or time.time() >= dl:
                        last_stats_emit = time.time()
                        lv = als / cc if cc > 0 else 10.0
                        ctps = tt / max(time.time() - cs_, 1)
                        emit('stats', {
                            'step': micro_step,
                            'target': target_micro_steps,
                            'loss': safe_float(lv, 10.0),
                            'tps': safe_float(ctps),
                            'global_step': gs,
                            'round': cr,
                            'time_left': safe_float(max(0, (dl - time.time()) / 60)),
                        })

                    del lo, x, y

                if hb.should_stop() or se.is_set():
                    break

            except RuntimeError as e:
                if "out of memory" not in str(e).lower():
                    raise
                ha()
                continue

            if cc == 0:
                break
            torch.nn.utils.clip_grad_norm_(bp, 1.0)
            ob.step()
            ob.zero_grad(set_to_none=True)

        if se.is_set():
            emit('log', "Stopped by user. Skipping upload.")
            return

        fl = float(lval) if lval else 10.0
        emit('log', f"Done: {micro_step} micro-steps, loss {fl:.4f}")
        emit('status', 'uploading')

        dbf = torch.from_numpy(model.get_base_weights() - iw).to(torch.bfloat16).view(torch.uint16).numpy()
        ed = model.engram.table.weight.data.cpu() - iew
        rn = ed.abs().sum(dim=1)
        k = max(1, int(len(rn) * 0.10))
        tv, ti = torch.topk(rn, k)
        ai = ti[tv > 1e-8]
        si = ai.cpu().numpy().astype(np.uint32) if len(ai) else np.array([], dtype=np.uint32)
        sv = ed[ai].to(torch.bfloat16).view(torch.uint16).numpy() if len(ai) else np.array([], dtype=np.uint16)

        has_engram = bool(len(si) > 0)
        engram_sparse_count = int(len(si))

        pl = json.dumps({
            "taskId": tid,
            "loss": fl,
            "localSteps": micro_step,
            "tokensProcessed": tt,
            "loraRank": 0,
            "isDelta": True,
            "weightFormat": "bf16",
            "hasEngram": has_engram,
            "engramSparseCount": engram_sparse_count,
        }).encode()

        bi = (
            struct.pack('<I', len(pl)) + pl
            + np.ascontiguousarray(dbf).tobytes()
            + np.ascontiguousarray(si).tobytes()
            + np.ascontiguousarray(sv).tobytes()
        )

        compressed_data = gzip.compress(bi, compresslevel=6)
        upload_size_mb = len(compressed_data) / (1024 * 1024)

        del dbf, ed, si, sv, pl, bi
        gc.collect()

        meta = {
            "taskId": tid,
            "loss": fl,
            "localSteps": micro_step,
            "tokensProcessed": tt,
            "loraRank": 0,
            "isDelta": True,
            "weightFormat": "bf16",
            "hasEngram": has_engram,
            "engramSparseCount": engram_sparse_count,
        }

        emit('overlay_show', {'title_key': 'uploading', 'indeterminate': False})
        emit('log', f"Uploading {upload_size_mb:.1f} MB delta to server...")

        r = None
        try:
            if len(compressed_data) <= MAX_DIRECT_UPLOAD_BYTES:
                try:
                    r = requests.post(
                        f"{srv}/fl/submit",
                        headers={"Content-Type": "application/octet-stream", "Content-Encoding": "gzip", **h},
                        data=stream_with_progress(compressed_data, emit, len(compressed_data), se),
                        timeout=600,
                    )
                    if r.status_code == 413:
                        emit("log", "Direct upload rejected by edge/proxy. Switching to chunked upload...")
                        r = chunked_submit_delta(srv, h, compressed_data, meta, emit, se)
                except Exception as e:
                    if len(compressed_data) > 8 * 1024 * 1024:
                        emit("log", f"Direct upload failed ({e}). Trying chunked upload...")
                        r = chunked_submit_delta(srv, h, compressed_data, meta, emit, se)
                    else:
                        raise
            else:
                r = chunked_submit_delta(srv, h, compressed_data, meta, emit, se)

            if r is not None and r.status_code == 200:
                emit("log", "Submitted successfully!")
            elif r is not None:
                emit("log", f"Submit failed: {r.text[:200]}")

        except KeyboardInterrupt:
            emit("log", "Upload aborted by user.")
        except Exception as e:
            emit("log", f"Upload failed: {e}")
        finally:
            emit("overlay_hide", {})
            try:
                del compressed_data
            except Exception:
                pass
            gc.collect()

    finally:
        hb.stop()
        model = ob = oe = x = y = lo = iw = ie = iew = ds = None
        gc.collect()
        if train_backend in ("CUDA", "ROCM"):
            try:
                torch.cuda.empty_cache()
                gc.collect()
                torch.cuda.empty_cache()
            except Exception:
                pass


# ============================================================
# HTML / CSS / JS  — keep this block EXACTLY as in the file
# you were pasting. It is functional and I don't want to
# introduce new bugs by editing it.
# ============================================================
# (Paste the same HTML = """...""" block you already have, unchanged.)


# ============================================================
# API
# ============================================================
class Api:
    def __init__(self):
        self.window = None
        self.stop_event = threading.Event()
        self.thread = None
        self.auth_token = None
        self.username = None
        self.server_url = DEFAULT_SERVER
        self._lp = 0.0
        self.selected_backend = get_best_default_backend()

    def emit(self, ev, data):
        if not self.window:
            return
        if ev in ('log', 'status'):
            try:
                print(f"[EMIT {ev}] {data}", flush=True)
            except Exception:
                pass
        if ev == 'preview_loss':
            n = time.time()
            if n - self._lp < 0.4:
                return
            self._lp = n
        try:
            clean_data = sanitize_for_json(data)
            self.window.evaluate_js(f"window.handleEvent({json.dumps(ev)}, {json.dumps(clean_data)});")
        except Exception as e:
            log.warning(f"Emit failed for {ev}: {e}")

    def get_backends(self):
        available = get_available_backends()
        self.emit('backends', {'available': available, 'current': self.selected_backend})

    def set_backend(self, backend_name):
        self.selected_backend = backend_name
        if self.thread and self.thread.is_alive():
            self.stop_event.set()

    def login(self, s, u, p):
        self.server_url = s or self.server_url
        threading.Thread(target=self._do_login, args=(u, p), daemon=True).start()

    def _do_login(self, u, p):
        tok, err = do_login(self.server_url, u, p)
        if tok:
            self.auth_token = tok
            self.username = u
            self.emit('login_success', {'username': u})
        else:
            self.emit('login_error', {'code': err or 'login_failed'})

    def register(self, s, u, email, p):
        self.server_url = s or self.server_url
        threading.Thread(target=self._do_register, args=(u, email, p), daemon=True).start()

    def _do_register(self, u, email, p):
        tok, err = do_register(self.server_url, u, email, p)
        if tok:
            self.auth_token = tok
            self.username = u
            self.emit('login_success', {'username': u})
        else:
            self.emit('login_error', {'code': err or 'register_failed'})

    def login_anon(self):
        self.auth_token = None
        self.username = "anonymous"
        self.emit('login_success', {'username': 'anonymous'})

    def start(self):
        if self.thread and self.thread.is_alive():
            return
        self.stop_event.clear()
        self.thread = threading.Thread(target=self._rs, daemon=True)
        self.thread.start()

    def stop(self):
        self.stop_event.set()

    def _check_updates_between_rounds(self):
        try:
            if check_for_update():
                self.emit('log', "🔄 Updated to a new version — restarting...")
                self.emit('status', 'idle')
                time.sleep(2)
                _restart_self()
        except Exception as e:
            log.warning(f"Update check failed: {e}")

    def _rs(self):
        global train_device, train_backend
        try:
            train_device, train_backend = detect_training_backend(self.selected_backend)
            self.emit('meta', {'backend': train_backend})
            self.emit('log', f"Backend: {train_backend} ({train_device})")
            auto_detect_vram_budget()
            self.emit('log', f"VRAM: {memory_config['ram_gb']:.1f} GB")
        except Exception as e:
            self.emit('log', f"Error: {e}")
            self.emit('status', 'idle')
            return

        round_count = 0
        while not self.stop_event.is_set():
            round_count += 1
            self.emit('log', f"{'='*40}")
            self.emit('log', f"Starting Round Cycle #{round_count}")
            self.emit('log', f"{'='*40}")

            try:
                run_single_round_wrapper(self.server_url, self.auth_token, self.stop_event, self.emit)
            except KeyboardInterrupt:
                self.stop_event.set()
                break
            except Exception as e:
                import traceback
                tb = traceback.format_exc()
                print(f"\n[ROUND-FAIL-TRACEBACK]\n{tb}\n", flush=True)
                for ln in tb.splitlines()[-14:]:
                    self.emit('log', ln)
                self.emit('log', f"Round failed: {e}")

            if self.stop_event.is_set():
                break

            if UPDATE_CHECK_BETWEEN_ROUNDS:
                self._check_updates_between_rounds()

            self.emit('status', 'waiting')
            self.emit('log', "✅ Round complete. Auto-relaunching next round in 10s...")
            for _ in range(10):
                if self.stop_event.is_set():
                    break
                time.sleep(1)

        self.emit('status', 'idle')
        self.emit('log', "👋 Training stopped by user.")


def startup():
    try:
        scr = webview.screens[0]
        aw, ah = scr.width, scr.height
        w = max(480, min(1040, int(aw * 0.80)))
        h = max(360, min(800, int(ah * 0.80)))
        w = min(w, aw - 16)
        h = min(h, ah - 48)
        w = max(320, w)
        h = max(240, h)
        window.resize(w, h)
        window.move(max(0, (aw - w) // 2), max(0, (ah - h) // 2))
    except Exception:
        pass

    def set_window_icon():
        try:
            import gi
            gi.require_version('Gtk', '3.0')
            from gi.repository import Gtk, GdkPixbuf
            icon_path = None
            if ICON_PATH and os.path.exists(ICON_PATH):
                icon_path = ICON_PATH
            elif _lp_light.exists():
                icon_path = str(_lp_light.absolute())
            if not icon_path:
                return
            for w in Gtk.Window.list_toplevels():
                if w.get_title() == "CrowdGPT":
                    pixbuf = GdkPixbuf.Pixbuf.new_from_file_at_scale(icon_path, 64, 64, True)
                    w.set_icon(pixbuf)
                    break
        except Exception:
            pass

    threading.Timer(0.5, set_window_icon).start()


if __name__ == "__main__":
    print(f"CrowdGPT client v{CLIENT_VERSION}")
    try:
        if check_for_update():
            print("Update applied. Restarting...")
            _restart_self()
    except Exception as e:
        log.warning(f"Auto-update check error: {e}")

    api = Api()
    start_tokenizer_loader()

    kwargs = {
        "js_api": api,
        "width": 1000,
        "height": 720,
        "min_size": (320, 240),
    }

    html_out = (HTML
                .replace("__LOGO__", LOGO_URI_LIGHT)
                .replace("__LOGO_DARK__", LOGO_URI_DARK)
                .replace("__VERSION__", CLIENT_VERSION))

    try:
        if ICON_PATH and os.path.exists(ICON_PATH):
            window = webview.create_window("CrowdGPT", html=html_out, icon=ICON_PATH, **kwargs)
        else:
            window = webview.create_window("CrowdGPT", html=html_out, **kwargs)
    except TypeError:
        window = webview.create_window("CrowdGPT", html=html_out, **kwargs)

    api.window = window
    webview.start(startup, debug=False)
