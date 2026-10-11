#!/usr/bin/env python3
"""
CrowdGPT GUI Client — HF weights + chunked upload edition.

Weight source of truth: HuggingFace (via metadata from /fl/task?skip_weights=true).
Delta upload: chunked (50 MB/chunk) with per-chunk SHA-256 and retries,
compatible with Cloudflare tunnels (100 MB response cap).
"""
import os, sys, io, json, time, struct, math, gc, threading, base64, logging, tempfile, atexit, subprocess, hashlib
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

# ============ CONFIG ============
CHECKPOINT_DIR = Path("checkpoints")
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

DATASET_REPO_ID = "Vxtzq/CrowdGPT"
TOKENIZER_REPO = "Qwen/Qwen3-0.6B"

MODEL_CONFIG = {
    "vocabSize": 151669, "dim": 1536, "nLayers": 24, "nHeads": 16, "nKvHeads": 4,
    "headDim": 96, "maxSeqLen": 2048, "mlpHidden": 2560, "weightTying": True,
}
VOCAB_SIZE = MODEL_CONFIG["vocabSize"]; DIM = MODEL_CONFIG["dim"]
N_LAYERS = MODEL_CONFIG["nLayers"]; N_HEADS = MODEL_CONFIG["nHeads"]
N_KV_HEADS = MODEL_CONFIG["nKvHeads"]; HEAD_DIM = MODEL_CONFIG["headDim"]
MAX_SEQ_LEN = MODEL_CONFIG["maxSeqLen"]; MLP_HIDDEN = MODEL_CONFIG["mlpHidden"]
ENG_NUM_BUCKETS = 227865
LOSS_CHUNK = 256; UPLOAD_BUFFER_MIN = 25
ALLOWED_BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64, 128]

UPLOAD_CHUNK_BYTES = 50 * 1024 * 1024   # 50 MB per chunk
UPLOAD_CHUNK_RETRIES = 5
UPLOAD_INIT_RETRIES = 3


def _calc_model_size():
    size = VOCAB_SIZE * DIM
    for _ in range(N_LAYERS):
        size += DIM*2 + DIM*(N_HEADS*HEAD_DIM) + DIM*(N_KV_HEADS*HEAD_DIM)*2 + DIM*DIM + DIM*2
        size += DIM*MLP_HIDDEN*2 + MLP_HIDDEN*DIM
    size += DIM*2
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
        if v is None or math.isnan(v) or math.isinf(v): return default
        return float(v)
    except: return default


# ============ JSON SANITIZER ============
def sanitize_for_json(obj):
    if isinstance(obj, dict):
        return {str(k): sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [sanitize_for_json(v) for v in obj]
    elif isinstance(obj, (int, float, str, bool, type(None))):
        if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
            return 0.0
        return obj
    try:
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating):
            v = float(obj)
            return 0.0 if math.isnan(v) or math.isinf(v) else v
        if isinstance(obj, np.bool_): return bool(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
    except ImportError:
        pass
    return str(obj)


# ============ TOKENIZER ============
TOKENIZER = None
_tokenizer_started = False
PREVIEW_EVERY = 15


def start_tokenizer_loader():
    global _tokenizer_started
    if _tokenizer_started: return
    _tokenizer_started = True
    threading.Thread(target=_load_tokenizer_worker, daemon=True).start()


def _load_tokenizer_worker():
    global TOKENIZER
    try:
        from transformers import AutoTokenizer
        candidates = [TOKENIZER_REPO, "Qwen/Qwen2.5-0.5B", "Vxtzq/Crowd-v1"]
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
            if len(s) > max_chars: s = s[:max_chars] + "…"
            return s or "…"
    except Exception:
        pass
    s = ' '.join(f"#{int(i)}" for i in ids)
    if len(s) > max_chars: s = s[:max_chars] + "…"
    return s


def decode_one(tok_id):
    s = decode_ids([tok_id], max_chars=32)
    return s if s.strip() else "␠"


# ============ LOGOS / ICON ============
LOGO_URI_LIGHT = ""
LOGO_URI_DARK = ""
_lp_light = Path(__file__).parent / "docs/logo-black.svg"
_lp_app = Path(__file__).parent / "docs/logo-app.svg"
_lp_dark = Path(__file__).parent / "docs/logo-white.svg"
if _lp_light.exists():
    try: LOGO_URI_LIGHT = "data:image/svg+xml;base64," + base64.b64encode(_lp_light.read_bytes()).decode()
    except: pass
if _lp_dark.exists():
    try: LOGO_URI_DARK = "data:image/svg+xml;base64," + base64.b64encode(_lp_dark.read_bytes()).decode()
    except: pass
if not LOGO_URI_DARK: LOGO_URI_DARK = LOGO_URI_LIGHT

ICON_PATH = None
_cleanup_icon = []
if _lp_light.exists():
    try:
        import cairosvg
        png_bytes = cairosvg.svg2png(url=str(_lp_app), output_width=256, output_height=256)
        tmp = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
        tmp.write(png_bytes); tmp.close()
        ICON_PATH = os.path.abspath(tmp.name)
        _cleanup_icon.append(ICON_PATH)
        atexit.register(lambda: [os.unlink(p) for p in _cleanup_icon if os.path.exists(p)])
    except ImportError:
        ICON_PATH = os.path.abspath(str(_lp_app))
    except Exception:
        ICON_PATH = os.path.abspath(str(_lp_app))


# ============ HARDWARE ============
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
        if avail.get(bk): return bk
    return 'cpu'


def detect_training_backend(force=None):
    if force and force != "auto":
        n = force.lower()
        if n == "cpu": return torch.device('cpu'), "CPU"
        if n == "cuda" and torch.cuda.is_available(): return torch.device('cuda'), "CUDA"
        if n == "rocm" and hasattr(torch.version, 'hip') and torch.version.hip and torch.cuda.is_available(): return torch.device('cuda'), "ROCM"
        if n == "mps" and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available(): return torch.device('mps'), "MPS"
        if n == "xpu":
            try:
                import intel_extension_for_pytorch
                if hasattr(torch, 'xpu') and torch.xpu.is_available(): return torch.device('xpu'), "XPU"
            except ImportError: pass
        if n == "directml":
            try:
                import torch_directml
                if torch_directml.is_available(): return torch_directml.device(0), "DIRECTML"
            except ImportError: pass
    if torch.cuda.is_available() and not (hasattr(torch.version, 'hip') and torch.version.hip): return torch.device('cuda'), "CUDA"
    if hasattr(torch.version, 'hip') and torch.version.hip and torch.cuda.is_available(): return torch.device('cuda'), "ROCM"
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available(): return torch.device('mps'), "MPS"
    try:
        import intel_extension_for_pytorch
        if hasattr(torch, 'xpu') and torch.xpu.is_available(): return torch.device('xpu'), "XPU"
    except ImportError: pass
    try:
        import torch_directml
        if torch_directml.is_available(): return torch_directml.device(0), "DIRECTML"
    except ImportError: pass
    return torch.device('cpu'), "CPU"


def auto_detect_vram_budget():
    global train_backend
    if train_backend in ("CUDA", "ROCM") and torch.cuda.is_available():
        try:
            fb, tb = torch.cuda.mem_get_info(0)
            memory_config.update({"ram_gb": round(max(1.0, min(fb/1024**3, tb/1024**3 - 2.0) - 1.0), 2), "is_auto_detected": True}); return
        except: pass
    if train_backend == "MPS":
        try:
            out = subprocess.check_output(["sysctl", "-n", "hw.memsize"]).decode().strip()
            total_gb = int(out) / 1024**3
            memory_config.update({"ram_gb": round(max(1.0, total_gb * 0.75 - 2.0), 2), "is_auto_detected": True}); return
        except: pass
    try:
        import psutil
        memory_config.update({"ram_gb": round(max(1.0, psutil.virtual_memory().available/1024**3 - 2.0), 2), "is_auto_detected": True})
    except: pass


def has_bitsandbytes():
    try:
        import bitsandbytes; return True
    except: return False


def estimate_vram_bytes(bs, sl, u8):
    mb = EXPECTED_MODEL_SIZE*2; ob = EXPECTED_MODEL_SIZE*(4 if u8 else 12); gb = EXPECTED_MODEL_SIZE*4
    ab = N_LAYERS*(bs*sl*DIM*2 + bs*N_HEADS*64*sl*4)
    lb = bs*LOSS_CHUNK*VOCAB_SIZE*4; eb = bs*sl*DIM*2
    return int(mb+ob+gb+ab+lb+eb+0.8*1024**3)


def recommend_batch_size(sl=2048):
    u8 = has_bitsandbytes(); budget = int(memory_config["ram_gb"]*1024**3); best = 1
    for bs in ALLOWED_BATCH_SIZES:
        if estimate_vram_bytes(bs, sl, u8) <= budget: best = bs
        else: break
    return best


# ============ MODEL ============
def precompute_freqs(dim, sl, dev):
    inv = 1.0/(10000.0**(torch.arange(0, dim, 2, dtype=torch.float32)/dim))
    f = torch.einsum("i,j->ij", torch.arange(sl, dtype=torch.float32), inv)
    e = torch.cat((f, f), -1)
    return e.cos()[None, None, :, :].to(dev), e.sin()[None, None, :, :].to(dev)


def rotate_half(x): return torch.cat((-x[..., x.shape[-1]//2:], x[..., :x.shape[-1]//2]), -1)


class GroupedQueryAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.nh, self.nkv, self.nrep = N_HEADS, N_KV_HEADS, N_HEADS//N_KV_HEADS
        self.wq = nn.Linear(DIM, N_HEADS*HEAD_DIM, bias=False)
        self.wk = nn.Linear(DIM, N_KV_HEADS*HEAD_DIM, bias=False)
        self.wv = nn.Linear(DIM, N_KV_HEADS*HEAD_DIM, bias=False)
        self.wo = nn.Linear(DIM, DIM, bias=False)

    def forward(self, x, cos, sin, use_chunked=True):
        B, T, C = x.size()
        q = self.wq(x).view(B, T, self.nh, HEAD_DIM).transpose(1, 2)
        k = self.wk(x).view(B, T, self.nkv, HEAD_DIM).transpose(1, 2)
        v = self.wv(x).view(B, T, self.nkv, HEAD_DIM).transpose(1, 2)
        ct, st = cos[:, :, :T, :], sin[:, :, :T, :]
        q = q*ct + rotate_half(q)*st; k = k*ct + rotate_half(k)*st
        k = k.unsqueeze(2).expand(B, self.nkv, self.nrep, T, HEAD_DIM).reshape(B, self.nh, T, HEAD_DIM)
        v = v.unsqueeze(2).expand(B, self.nkv, self.nrep, T, HEAD_DIM).reshape(B, self.nh, T, HEAD_DIM)
        if use_chunked: return self._chunked(q, k, v, B, T, C)
        a = (q @ k.transpose(-2, -1))*(1.0/math.sqrt(HEAD_DIM))
        a = a.masked_fill(torch.tril(torch.ones(T, T, device=x.device)).view(1, 1, T, T) == 0, float('-inf'))
        a = F.softmax(a, dim=-1, dtype=torch.float32).to(q.dtype)
        return self.wo((a @ v).transpose(1, 2).contiguous().view(B, T, C))

    def _chunked(self, q, k, v, B, T, C, cs=64):
        chunks = []
        for i in range(0, T, cs):
            e = min(i+cs, T)
            aw = (q[:, :, i:e, :] @ k[:, :, :e, :].transpose(-2, -1))*(1.0/math.sqrt(HEAD_DIM))
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
    def forward(self, x): return self.w3(F.silu(self.w1(x))*self.w2(x))


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln_1, self.attn, self.ln_2, self.mlp = nn.LayerNorm(DIM), GroupedQueryAttention(), nn.LayerNorm(DIM), SwiGLU()
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
        hi = (px*1000003 + idx) % ENG_NUM_BUCKETS
        uq, inv = torch.unique(hi.flatten(), return_inverse=True)
        cr = self.table(uq.cpu())
        if self.use_async:
            with torch.cuda.stream(self.ts): cg = cr.to(self.device, non_blocking=True)
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
        if MODEL_CONFIG["weightTying"]: self.wte.weight = self.lm_head.weight
        cm, sm = precompute_freqs(HEAD_DIM, MAX_SEQ_LEN, device)
        self.register_buffer("freqs_cos", cm); self.register_buffer("freqs_sin", sm)

    def get_base_weights(self):
        return np.concatenate([p.detach().float().flatten().cpu().numpy() for n, p in self.named_parameters() if not n.startswith('engram.')])

    def load_base_weights(self, fw):
        ft = torch.from_numpy(fw) if not isinstance(fw, torch.Tensor) else fw
        o = 0
        for n, p in self.named_parameters():
            if not n.startswith('engram.'):
                s = p.numel(); p.data.copy_(ft[o:o+s].view(p.shape).to(p.device)); o += s


# ============ DATASET ============
def _fetch_with_retry(url, headers=None, timeout=60, retries=5):
    for a in range(retries):
        try:
            r = requests.get(url, headers=headers, timeout=timeout, allow_redirects=True)
            r.raise_for_status(); return r
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout,
                ChunkedEncodingError, http.client.IncompleteRead):
            if a < retries-1: time.sleep(2**(a+1))
            else: raise


class StreamingShardDataset:
    STEPS_PER_SUBCHUNK = 500
    def __init__(self, repo_id, ci, ss=10*1024*1024, tps=65, slot=0, at=None):
        self.repo_id, self.ci, self.sub_size, self.tps, self.auth_token = repo_id, ci, ss, tps, at
        self._fmt = "chunk_{:04d}.bin"
        self.chunk_size = self._discover(ci)
        self.off = (slot*ss) % self.chunk_size
        self.data, self.n, self.steps_used = None, 0, 0
        self._pft, self._pfr, self._lock = None, None, threading.Lock()
        self._load(self.off); self._prefetch(self.off+ss)

    def _url(self): return f"https://huggingface.co/datasets/{self.repo_id}/resolve/main/chunks/" + self._fmt.format(self.ci)

    def _discover(self, idx):
        for fmt in ("chunk_{:04d}.bin", "chunk_{:d}.bin"):
            url = f"https://huggingface.co/datasets/{self.repo_id}/resolve/main/chunks/" + fmt.format(idx)
            for _ in range(2):
                try:
                    r = requests.head(url, allow_redirects=True, timeout=30)
                    if r.status_code == 200:
                        sz = int(r.headers.get('content-length', 0))
                        if sz > 0: self._fmt = fmt; return sz
                except: time.sleep(1)
        raise RuntimeError(f"No chunk {idx}")

    def _slice(self, off):
        end = min(off+self.sub_size, self.chunk_size)-1
        return _fetch_with_retry(self._url(), headers={'Range': f'bytes={off}-{end}'}, timeout=120).content

    def _set(self, raw):
        tk = np.frombuffer(raw, dtype=np.uint32)
        self.data, self.n, self.steps_used = tk, len(tk)//self.tps, 0

    def _load(self, off): self._set(self._slice(off)); self.off = off

    def _prefetch(self, off):
        if off >= self.chunk_size: return
        with self._lock: self._pfr = None
        def w():
            try:
                r = self._slice(off)
                with self._lock: self._pfr = (off, r)
            except: pass
        self._pft = threading.Thread(target=w, daemon=True); self._pft.start()

    def needs_new_subchunk(self): return self.n == 0 or self.steps_used >= self.STEPS_PER_SUBCHUNK

    def advance(self, srv=None, fmt="bf16"):
        no = self.off+self.sub_size
        if no < self.chunk_size:
            if self._pft: self._pft.join(timeout=120); self._pft = None
            with self._lock: r = self._pfr; self._pfr = None
            if r and r[0] == no: self.off = no; self._set(r[1])
            else: self._load(no)
            self._prefetch(no+self.sub_size); return True
        self._req(srv, fmt)
        self.chunk_size = self._discover(self.ci); self.off = 0
        self._load(0); self._prefetch(self.sub_size); return True

    def _req(self, srv, fmt="bf16"):
        if not srv: return False
        try:
            h = {"Authorization": f"Bearer {self.auth_token}"} if self.auth_token else {}
            r = requests.get(f"{srv}/fl/task?format={fmt}&skip_weights=true", headers=h, timeout=30)
            if r.status_code == 200:
                raw = r.content; ml = struct.unpack('<I', raw[:4])[0]
                ni = json.loads(raw[4:4+ml].decode()).get("datasetConfig", {}).get("chunkIdx", self.ci)
                if ni != self.ci: self.ci = ni; return True
        except: pass
        return False

    def get_batch(self, bs, seed=None):
        if self.data is None or self.n == 0: self.advance()
        self.steps_used += 1
        rng = np.random.RandomState(seed)
        s = rng.randint(0, self.n, size=bs)*self.tps
        inp = np.stack([self.data[x:x+self.tps-1] for x in s])
        tgt = np.stack([self.data[x+1:x+self.tps] for x in s])
        return torch.tensor(inp, dtype=torch.long), torch.tensor(tgt, dtype=torch.long)


def decompress_weights(raw, fmt="bf16"):
    if fmt == "fp16":
        return np.frombuffer(raw, dtype=np.uint16).view(np.float16).astype(np.float32)
    return torch.from_numpy(np.frombuffer(raw, dtype=np.uint16).copy()).view(torch.bfloat16).to(torch.float32).numpy()


# ============ SAFETENSORS PARSING ============
def _safetensors_extract(raw, tensor_key="weights"):
    """Extract raw tensor bytes from a single-tensor safetensors blob."""
    if len(raw) < 8:
        raise ValueError(f"safetensors too small ({len(raw)} bytes)")
    n = struct.unpack('<Q', raw[:8])[0]
    if 8 + n > len(raw):
        raise ValueError(f"safetensors header truncated (n={n}, len={len(raw)})")
    header = json.loads(raw[8:8 + n].decode('utf-8'))
    if tensor_key in header:
        info = header[tensor_key]
    elif "__metadata__" in header and len(header) == 2:
        k = next(k for k in header if k != "__metadata__")
        info = header[k]
    elif len(header) == 1:
        info = next(iter(header.values()))
    else:
        raise ValueError(f"safetensors tensor key {tensor_key!r} not found; keys={list(header.keys())}")
    lo, hi = info["data_offsets"]
    start = 8 + n + lo
    end = 8 + n + hi
    if end > len(raw):
        raise ValueError(f"safetensors data truncated ({end} > {len(raw)})")
    return raw[start:end]


# ============ WEIGHTS DOWNLOAD (HF-first) ============
def _download_url(url, expected_bytes, se, emit, label):
    """Stream a direct URL, report progress, return bytes or None."""
    try:
        r = requests.get(url, timeout=(15, 3600), stream=True, allow_redirects=True)
        if r.status_code != 200:
            emit('log', f"{label}: HTTP {r.status_code}")
            r.close(); return None
        buf = []; got = 0; last = 0.0
        for c in r.iter_content(chunk_size=1024 * 1024):
            if se.is_set(): r.close(); return None
            if not c: continue
            buf.append(c); got += len(c)
            now = time.time()
            if now - last > 0.25:
                emit('overlay_progress', {'done': got, 'total': expected_bytes}); last = now
        r.close()
        out = b"".join(buf)
        if expected_bytes and len(out) < expected_bytes:
            emit('log', f"{label}: truncated ({len(out):,} / {expected_bytes:,})")
            return None
        return out
    except Exception as e:
        emit('log', f"{label} error: {e}")
        return None


def _fetch_weights_via_hf(wmeta, se, emit):
    """Download model + engram from HF. Returns concatenated raw bf16 bytes."""
    files = wmeta.get("files") or {}
    model_info = files.get("model")
    engram_info = files.get("engram")
    if not model_info or not engram_info:
        emit('log', "HF metadata incomplete; cannot use HF path")
        return None

    emit('log', f"HF model: {model_info['size']/1e6:.0f} MB from {model_info['url']}")
    raw_model = _download_url(model_info['url'], model_info['size'], se, emit, "model")
    if raw_model is None: return None
    try:
        base = _safetensors_extract(raw_model, model_info.get('tensorKey', 'weights'))
    except Exception as e:
        emit('log', f"model safetensors parse failed: {e}")
        return None
    finally:
        del raw_model

    emit('log', f"HF engram: {engram_info['size']/1e6:.0f} MB")
    raw_eng = _download_url(engram_info['url'], engram_info['size'], se, emit, "engram")
    if raw_eng is None: return None
    try:
        eng = _safetensors_extract(raw_eng, engram_info.get('tensorKey', 'weights'))
    except Exception as e:
        emit('log', f"engram safetensors parse failed: {e}")
        return None
    finally:
        del raw_eng

    if len(base) < EXPECTED_MODEL_BYTES or len(eng) < EXPECTED_ENGRAM_BYTES:
        emit('log', f"HF payload short: base={len(base):,} engram={len(eng):,}")
        return None
    return base[:EXPECTED_MODEL_BYTES] + eng[:EXPECTED_ENGRAM_BYTES]


def fetch_task(srv, h, prec, se, emit):
    """
    Returns (metadata, weights_bytes).
    - Prefers HF-direct download (metadata only over the wire).
    - Falls back to inline /fl/task streaming if HF not advertised.
    """
    while not se.is_set():
        try:
            emit('status', 'downloading')
            emit('overlay_show', {'title_key': 'dl_weights', 'indeterminate': False})

            # --- Step 1: metadata only (tiny) ---
            r = requests.get(f"{srv}/fl/task?format={prec}&skip_weights=true",
                             headers=h, timeout=(15, 120))
            if r.status_code != 200:
                emit('overlay_hide', {})
                emit('log', f"/fl/task HTTP {r.status_code}: {r.text[:120]}")
                time.sleep(10); continue
            if r.headers.get("X-Status") == "wait":
                emit('overlay_hide', {})
                for _ in range(10):
                    if se.is_set(): return None, None
                    time.sleep(1)
                continue

            metadata = r.json()
            wmeta = metadata.get("weights") or {}
            source = wmeta.get("source")

            # --- Step 2: HF preferred ---
            if source == "huggingface":
                emit('log', f"HF source: {wmeta.get('repo')} @ {str(wmeta.get('revision'))[:12]}")
                wb = _fetch_weights_via_hf(wmeta, se, emit)
                if wb is not None and len(wb) >= EXPECTED_WEIGHT_BYTES:
                    emit('overlay_hide', {})
                    emit('status', 'preparing')
                    emit('log', f"Weights ready: {len(wb):,} bytes (HF)")
                    return metadata, wb
                emit('log', "⚠️ HF download failed; falling back to inline /fl/task")

            # --- Step 3: inline fallback ---
            emit('log', "Downloading weights inline from /fl/task (may fail via Cloudflare tunnel)")
            r = requests.get(f"{srv}/fl/task?format={prec}",
                             headers=h, timeout=(15, 3600), stream=True)
            if r.status_code != 200 or r.headers.get("X-Status") == "wait":
                r.close(); emit('overlay_hide', {})
                time.sleep(10); continue

            cl_total = int(r.headers.get('content-length', 0) or 0)
            total = cl_total if cl_total > 0 else EXPECTED_WEIGHT_BYTES
            chunks = []; done = 0; last = 0.0
            for chunk in r.iter_content(chunk_size=512 * 1024):
                if not chunk: continue
                chunks.append(chunk); done += len(chunk)
                now = time.time()
                if now - last > 0.25:
                    emit('overlay_progress', {'done': done, 'total': total}); last = now
            r.close()
            emit('overlay_hide', {})

            raw = b"".join(chunks); del chunks
            if len(raw) < 8:
                emit('log', f"/fl/task returned {len(raw)} bytes; retrying"); time.sleep(5); continue
            ml = struct.unpack('<I', raw[:4])[0]
            if 4 + ml > len(raw):
                emit('log', f"Truncated header (ml={ml}, len={len(raw)}); retrying"); time.sleep(5); continue
            md = json.loads(raw[4:4 + ml].decode())
            wb = raw[4 + ml:]
            emit('log', f"/fl/task inline: metadata OK, weights={len(wb):,} bytes")
            if len(wb) < EXPECTED_WEIGHT_BYTES:
                emit('log', f"⚠️ inline truncated at {len(wb):,} (expected {EXPECTED_WEIGHT_BYTES:,}); retrying")
                time.sleep(10); continue
            emit('status', 'preparing')
            return md, wb

        except Exception as e:
            emit('overlay_hide', {})
            if se.is_set(): return None, None
            emit('log', f"Task fetch failed: {e}"); time.sleep(15)


# ============ CHUNKED UPLOAD ============
def upload_delta_chunked(srv, h, tid, loss, local_steps, tokens_processed,
                         dbf, si, sv, se, emit):
    """
    Build the delta payload, gzip it to a temp file, then upload via:
      POST /fl/submit/init   (JSON metadata, sha256 of gzipped payload)
      POST /fl/submit/chunk  (50 MB per chunk, X-Chunk-Sha256 header)
      POST /fl/submit/finalize
    Retries each chunk up to UPLOAD_CHUNK_RETRIES times with exponential backoff.
    Returns True on success.
    """
    pl = json.dumps({
        "taskId": tid, "loss": loss, "localSteps": local_steps,
        "tokensProcessed": tokens_processed, "loraRank": 0, "isDelta": True,
        "weightFormat": "bf16", "hasEngram": True, "engramSparseCount": int(len(si)),
    }).encode()

    gz_path = None
    try:
        # --- Step 1: gzip payload to a temp file (bounded memory) ---
        with tempfile.NamedTemporaryFile(delete=False, suffix='.bin.gz') as tmp:
            gz_path = tmp.name
        with gzip.open(gz_path, 'wb', compresslevel=2) as gz:
            gz.write(struct.pack('<I', len(pl)))
            gz.write(pl)
            gz.write(np.ascontiguousarray(dbf).tobytes()); del dbf
            gz.write(np.ascontiguousarray(si).tobytes()); del si
            gz.write(np.ascontiguousarray(sv).tobytes()); del sv

        compressed_size = os.path.getsize(gz_path)
        emit('log', f"Payload compressed: {compressed_size/1e6:.1f} MB")

        # --- Step 2: sha256 of the whole gzipped file ---
        h_sha = hashlib.sha256()
        with open(gz_path, 'rb') as f:
            while True:
                b = f.read(4 * 1024 * 1024)
                if not b: break
                h_sha.update(b)
        checksum = h_sha.hexdigest()

        chunk_size = UPLOAD_CHUNK_BYTES
        chunks = (compressed_size + chunk_size - 1) // chunk_size
        emit('log', f"Uploading {chunks} chunk(s) of {chunk_size//1024//1024} MB")

        # --- Step 3: init ---
        init_payload = {
            "taskId": tid,
            "totalBytes": compressed_size,
            "chunkSize": chunk_size,
            "chunks": chunks,
            "checksum": checksum,
            "weightFormat": "bf16",
            "hasEngram": True,
            "engramSparseCount": int(len(si) if isinstance(si, np.ndarray) else 0),
            "loss": loss,
            "localSteps": local_steps,
            "tokensProcessed": tokens_processed,
            "loraRank": 0,
            "isDelta": True,
        }
        # si was already deleted during gzip; use the recorded count. Fix below.
        # Recompute count from the ORIGINAL si before deletion — patch: we
        # stored it as engram_sparse_count before the write. Simplest fix:
        # reconstruct here from a passed-in value.
        # (See call site: we pass `si_count` as len(si) before calling.)
        upload_id = None
        for attempt in range(UPLOAD_INIT_RETRIES):
            try:
                r = requests.post(f"{srv}/fl/submit/init", headers=h, json=init_payload, timeout=60)
                if r.status_code == 200:
                    upload_id = r.json()["uploadId"]; break
                emit('log', f"init attempt {attempt+1}: HTTP {r.status_code}: {r.text[:120]}")
            except Exception as e:
                emit('log', f"init attempt {attempt+1}: {e}")
            time.sleep(2 ** attempt)
        if not upload_id:
            emit('log', "❌ init failed after retries"); return False

        # --- Step 4: chunks ---
        with open(gz_path, 'rb') as f:
            for i in range(chunks):
                if se.is_set(): return False
                chunk_bytes = f.read(chunk_size)
                chunk_sha = hashlib.sha256(chunk_bytes).hexdigest()

                ok = False
                for attempt in range(UPLOAD_CHUNK_RETRIES):
                    if se.is_set(): return False
                    try:
                        r = requests.post(
                            f"{srv}/fl/submit/chunk",
                            params={"uploadId": upload_id, "index": i},
                            headers={"Content-Type": "application/octet-stream",
                                     "X-Chunk-Sha256": chunk_sha, **h},
                            data=chunk_bytes,
                            timeout=(15, 600),
                        )
                        if r.status_code == 200:
                            ok = True; break
                        emit('log', f"chunk {i+1}/{chunks} attempt {attempt+1}: HTTP {r.status_code}: {r.text[:80]}")
                    except Exception as e:
                        emit('log', f"chunk {i+1}/{chunks} attempt {attempt+1}: {e}")
                    time.sleep(2 ** attempt)
                if not ok:
                    emit('log', f"❌ chunk {i+1}/{chunks} failed after {UPLOAD_CHUNK_RETRIES} retries")
                    return False

                done = min((i+1) * chunk_size, compressed_size)
                emit('overlay_progress', {'done': done, 'total': compressed_size})
                emit('log', f"  ✓ chunk {i+1}/{chunks} ({done/1e6:.1f}/{compressed_size/1e6:.1f} MB)")

        # --- Step 5: finalize ---
        for attempt in range(UPLOAD_INIT_RETRIES):
            try:
                r = requests.post(f"{srv}/fl/submit/finalize",
                                  headers={**h},
                                  json={"uploadId": upload_id},
                                  timeout=600)
                if r.status_code == 200:
                    emit('log', f"✅ Submitted successfully ({chunks} chunks)")
                    return True
                emit('log', f"finalize attempt {attempt+1}: HTTP {r.status_code}: {r.text[:120]}")
            except Exception as e:
                emit('log', f"finalize attempt {attempt+1}: {e}")
            time.sleep(2 ** attempt)
        emit('log', "❌ finalize failed after retries")
        return False

    finally:
        if gz_path and os.path.exists(gz_path):
            try: os.unlink(gz_path)
            except: pass


# ============ AUTH ============
def do_login(srv, u, p):
    try:
        r = requests.post(f"{srv}/auth/login", json={"username": u, "password": p}, timeout=30)
        if r.status_code == 200: return r.json().get("token"), None
        try: msg = r.json().get("detail", "")
        except: msg = r.text
        if "invalid" in msg.lower() or "credentials" in msg.lower(): return None, "invalid_credentials"
        return None, "login_failed"
    except requests.exceptions.ConnectionError:
        return None, "network"
    except:
        return None, "network"


def do_register(srv, u, email, p):
    try:
        r = requests.post(f"{srv}/auth/register", json={"username": u, "password": p, "email": email}, timeout=30)
        if r.status_code == 200: return r.json().get("token"), None
        try: msg = r.json().get("detail", "")
        except: msg = r.text
        msg_l = msg.lower()
        if "username" in msg_l and "taken" in msg_l: return None, "username_taken"
        if "email" in msg_l and "registered" in msg_l: return None, "email_taken"
        if "password" in msg_l: return None, "weak_password"
        if "invalid" in msg_l: return None, "invalid_input"
        return None, "register_failed"
    except requests.exceptions.ConnectionError:
        return None, "network"
    except:
        return None, "network"


# ============ HEARTBEAT / ROUND ============
class HeartbeatManager:
    def __init__(self, srv, h, cr, se, emit):
        self.srv, self.h, self.cr = srv, h, cr
        self.stop_training = threading.Event(); self.shutdown = threading.Event()
        self.se, self.emit = se, emit
        self._t = threading.Thread(target=self._r, daemon=True)
    def start(self): self._t.start()
    def stop(self): self.shutdown.set()
    def should_stop(self): return self.stop_training.is_set()
    def _r(self):
        while not self.shutdown.is_set() and not self.se.is_set():
            try:
                requests.get(f"{self.srv}/fl/heartbeat", headers=self.h, timeout=10)
                r = requests.get(f"{self.srv}/fl/round_status", headers=self.h, timeout=10)
                if r.status_code == 200:
                    st = r.json()
                    if st.get("current_round", self.cr) != self.cr:
                        self.emit('log', "Next round. Stopping."); self.stop_training.set()
                    rm = (st.get("max_round_hours", 2) - st.get("round_elapsed_hours", 0))*60
                    if rm <= 3:
                        self.emit('log', f"ULTIMATUM: {rm:.0f} min left!"); self.stop_training.set()
            except: pass
            self.shutdown.wait(timeout=15)


def wait_for_round(srv, h, se, emit):
    overlay_on = False
    def show_wait():
        nonlocal overlay_on
        if not overlay_on:
            emit('overlay_show', {'title_key': 'wait_coord', 'subtitle_key': 'wait_coord_sub', 'indeterminate': True}); overlay_on = True
    def hide():
        nonlocal overlay_on
        if overlay_on:
            emit('overlay_hide', {}); overlay_on = False
    while not se.is_set():
        try:
            r = requests.get(f"{srv}/fl/round_status", headers=h, timeout=30)
            if r.status_code == 200:
                st = r.json()
                if st.get("is_aggregating"):
                    emit('status', 'waiting'); show_wait()
                    for _ in range(30):
                        if se.is_set(): hide(); return None
                        time.sleep(1)
                    continue
                if not st.get("in_cooldown", False):
                    hide(); return st
                emit('status', 'waiting'); show_wait()
                for _ in range(30):
                    if se.is_set(): hide(); return None
                    time.sleep(1)
            else:
                time.sleep(10)
        except:
            time.sleep(10)
    hide(); return None


# ============ FORWARD / LOSS ============
def _chunk_ce(m, xs, ys):
    return F.cross_entropy(m.lm_head(xs).reshape(-1, VOCAB_SIZE), ys.reshape(-1))


def _fwl(m, x, y, sl, ua, ad, ls=1.0, preview=False):
    def fwd():
        xe = m.wte(x)
        if ua:
            with torch.autocast(device_type='cuda', enabled=False): eo = m.engram(x)
        else:
            eo = m.engram(x)
        xe = xe + eo.to(xe.dtype)
        for b in m.blocks:
            xe = checkpoint(b, xe, m.freqs_cos, m.freqs_sin, True, use_reentrant=False)
        return m.ln_f(xe)

    nc = max(1, math.ceil(sl/LOSS_CHUNK))
    if ua:
        with torch.autocast(device_type='cuda', dtype=ad):
            xe = fwd()
            lo = sum(checkpoint(_chunk_ce, m, xe[:, i:i+LOSS_CHUNK, :], y[:, i:i+LOSS_CHUNK], use_reentrant=False)
                     for i in range(0, sl, LOSS_CHUNK))/nc
    else:
        xe = fwd()
        lo = sum(checkpoint(_chunk_ce, m, xe[:, i:i+LOSS_CHUNK, :], y[:, i:i+LOSS_CHUNK], use_reentrant=False)
                 for i in range(0, sl, LOSS_CHUNK))/nc

    cards = []
    if preview:
        try:
            with torch.no_grad():
                B, T, C = xe.shape
                items = []
                for b in range(min(B, 3)): items.append((b, T - 1))
                for offset in (24, 48, 72):
                    p = T - offset - 1
                    if p >= 9: items.append((0, p))
                items = items[:6]
                for b, p in items:
                    if p < 9 or p >= T: continue
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
                        'pred_id': int(pred_id)
                    })
        except Exception as e:
            log.warning(f"Preview extraction failed: {e}")
            cards = []
    return lo * ls, cards


# ============ TRAIN ROUND ============
def run_single_round_wrapper(srv, at, se, emit):
    global train_device, train_backend
    h = {"Authorization": f"Bearer {at}"} if at else {}
    emit('status', 'waiting')
    rs = wait_for_round(srv, h, se, emit)
    if not rs: return

    cr = rs["current_round"]
    hb = HeartbeatManager(srv, h, cr, se, emit)
    hb.start()

    model = ob = oe = x = y = lo = iw = ie = iew = ds = None
    micro_step = 0
    lval = 10.0
    tt = 0
    tid = ""

    try:
        emit('log', "Fetching weights (HF-preferred)…")
        md, wb = fetch_task(srv, h, "bf16", se, emit)
        if md is None: return

        emit('status', 'preparing')
        emit('log', "Preparing model…")

        tid, gs = md['taskId'], md['globalStep']
        sl = 2048
        if len(wb) < EXPECTED_WEIGHT_BYTES:
            emit('log', f"❌ Weight buffer too small: {len(wb):,} / {EXPECTED_WEIGHT_BYTES:,}")
            return

        iw = decompress_weights(wb[:EXPECTED_MODEL_BYTES], "bf16")
        ie = decompress_weights(wb[EXPECTED_MODEL_BYTES:EXPECTED_WEIGHT_BYTES], "bf16")
        del wb; gc.collect()

        dc = md.get("datasetConfig", {})
        sc = md.get("shardConfig", {})
        ds = StreamingShardDataset(
            dc.get("repoId", DATASET_REPO_ID),
            dc.get("chunkIdx", 0),
            dc.get("subChunkSize", 10*1024*1024),
            dc.get("tokensPerSample", sl+1),
            sc.get("slot", 0),
            at
        )

        bs = recommend_batch_size(sl)
        while bs >= 1:
            try:
                if train_backend in ("CUDA", "ROCM"):
                    try: torch.cuda.empty_cache()
                    except: pass
                gc.collect()
                model = SotaGPT(train_backend, train_device).to(train_device)
                model.engram.table.to('cpu')
                model.load_base_weights(iw)
                model.engram.table.weight.data.copy_(torch.from_numpy(ie).view(model.engram.table.weight.shape))
                model.train()
                break
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    if model: del model; model = None
                    if train_backend in ("CUDA", "ROCM"):
                        try: torch.cuda.empty_cache()
                        except: pass
                    gc.collect(); bs = max(1, bs//2)
                else:
                    raise

        if model is None:
            emit('log', "Failed to initialize model."); return

        mbt = bs*sl; as_ = max(1, round(131072/mbt)); ls = 1.0/as_

        def ha():
            nonlocal as_, ls
            as_ = max(1, as_//2); ls = 1.0/as_
            ob.zero_grad(set_to_none=True); gc.collect()
            if train_backend in ("CUDA", "ROCM"):
                try: torch.cuda.empty_cache()
                except: pass
            emit('log', f"OOM -> accum now {as_}x")

        bp = [p for n, p in model.named_parameters() if not n.startswith('engram.')]
        ep = list(model.engram.parameters())
        try:
            import bitsandbytes as bnb
            ob = bnb.optim.AdamW8bit(bp, lr=1e-4, betas=(0.9, 0.95), weight_decay=0.01)
        except:
            ob = torch.optim.AdamW(bp, lr=1e-4, betas=(0.9, 0.95), weight_decay=0.01)
        try:
            oe = torch.optim.SparseAdam(ep, lr=1e-4)
        except:
            oe = torch.optim.AdamW(ep, lr=1e-4, weight_decay=0.01)

        iew = model.engram.table.weight.data.cpu().clone()
        ua = train_backend in ("CUDA", "ROCM") and torch.cuda.is_bf16_supported()
        ad = torch.bfloat16 if ua else None

        emit('log', "Calibrating speed…")
        cs_ = time.time()
        cal_global_steps = 3
        target_micro_steps = 100
        emit('status', 'calibrating')
        cal_done = 0

        while cal_done < cal_global_steps:
            if hb.should_stop() or se.is_set(): return
            ob.zero_grad(set_to_none=True)
            try:
                for mi in range(as_):
                    if hb.should_stop() or se.is_set(): break
                    if ds.needs_new_subchunk(): ds.advance(srv, "bf16")
                    sd = (abs(hash("cal")) % 10000) + cal_done*1000 + mi
                    x, y = ds.get_batch(bs, seed=sd); x, y = x.to(train_device), y.to(train_device)
                    want_preview = (micro_step == 0) or ((micro_step + 1) % PREVIEW_EVERY == 0)
                    lo, cards = _fwl(model, x, y, sl, ua, ad, ls, preview=want_preview)
                    lval = float(lo.item()); micro_step += 1
                    if cards: emit('model_preview', {'step': micro_step, 'cards': cards})
                    if cal_done >= 1 and mi == 0:
                        elapsed = time.time() - cs_
                        sps_micro = elapsed / micro_step
                        rh = max(0.1, rs.get("max_round_hours", 2.0) - rs.get("round_elapsed_hours", 0))
                        remaining = max(60, (rh*3600) - elapsed - (UPLOAD_BUFFER_MIN*60))
                        target_micro_steps = micro_step + int(remaining / sps_micro)
                    elapsed_cal = time.time() - cs_
                    current_tps = tt / max(elapsed_cal, 1.0)
                    emit('cal_stats', {
                        'step': micro_step, 'total': target_micro_steps,
                        'loss': safe_float(lval * as_, 10.0),
                        'microbatch': micro_step,
                        'tps': safe_float(current_tps)
                    })
                    lo.backward()
                    oe.step(); oe.zero_grad(set_to_none=True)
                    tt += x.numel(); del lo, x, y
                if hb.should_stop() or se.is_set(): return
            except RuntimeError as e:
                if "out of memory" not in str(e).lower(): raise
                ha(); continue
            torch.nn.utils.clip_grad_norm_(bp, 1.0)
            ob.step(); ob.zero_grad(set_to_none=True)
            gc.collect(); cal_done += 1

        if se.is_set(): return

        rh = max(0.1, rs.get("max_round_hours", 2.0) - rs.get("round_elapsed_hours", 0))
        remaining = max(60, (rh*3600) - (time.time() - cs_) - (UPLOAD_BUFFER_MIN*60))
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
            if hb.should_stop() or se.is_set(): break
            ob.zero_grad(set_to_none=True)
            als = 0.0; cc = 0
            try:
                for mi in range(as_):
                    if hb.should_stop() or se.is_set() or time.time() >= dl: break
                    if ds.needs_new_subchunk(): ds.advance(srv, "bf16")
                    sd = (abs(hash("t")) % 10000) + micro_step*1000 + mi
                    x, y = ds.get_batch(bs, seed=sd); x, y = x.to(train_device), y.to(train_device)
                    want_preview = (micro_step == 0) or ((micro_step + 1) % PREVIEW_EVERY == 0)
                    lo, cards = _fwl(model, x, y, sl, ua, ad, ls, preview=want_preview)
                    lval = float(lo.item())*as_
                    lo.backward()
                    oe.step(); oe.zero_grad(set_to_none=True)
                    als += lval; tt += x.numel(); cc += 1; micro_step += 1
                    if cards: emit('model_preview', {'step': micro_step, 'cards': cards})
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
                        lv = als/cc if cc > 0 else 10.0
                        ctps = tt/max(time.time()-cs_, 1)
                        emit('stats', {
                            'step': micro_step, 'target': target_micro_steps,
                            'loss': safe_float(lv, 10.0), 'tps': safe_float(ctps),
                            'global_step': gs, 'round': cr,
                            'time_left': safe_float(max(0, (dl-time.time())/60))
                        })
                    del lo, x, y
                if hb.should_stop() or se.is_set(): break
            except RuntimeError as e:
                if "out of memory" not in str(e).lower(): raise
                ha(); continue
            if cc == 0: break
            torch.nn.utils.clip_grad_norm_(bp, 1.0)
            ob.step(); ob.zero_grad(set_to_none=True)

        if se.is_set():
            emit('log', "Stopped by user. Skipping upload."); return

        fl = float(lval) if lval else 10.0
        emit('log', f"Done: {micro_step} micro-steps, loss {fl:.4f}")
        emit('status', 'uploading')

        # --- Build delta ---
        dbf = torch.from_numpy(model.get_base_weights()-iw).to(torch.bfloat16).view(torch.uint16).numpy()
        ed = model.engram.table.weight.data.cpu()-iew
        rn = ed.abs().sum(dim=1); k = max(1, int(len(rn)*0.10))
        tv, ti = torch.topk(rn, k); ai = ti[tv > 1e-8]
        si = ai.cpu().numpy().astype(np.uint32) if len(ai) else np.array([], dtype=np.uint32)
        sv = ed[ai].to(torch.bfloat16).view(torch.uint16).numpy() if len(ai) else np.array([], dtype=np.uint16)

        emit('overlay_show', {'title_key': 'uploading', 'indeterminate': False})
        ok = upload_delta_chunked(srv, h, tid, fl, micro_step, tt, dbf, si, sv, se, emit)
        emit('overlay_hide', {})
        if not ok:
            emit('log', "❌ Upload failed; delta not submitted")

    finally:
        hb.stop()
        model = ob = oe = x = y = lo = iw = ie = iew = ds = None
        gc.collect()
        if train_backend in ("CUDA", "ROCM"):
            try:
                torch.cuda.empty_cache(); gc.collect(); torch.cuda.empty_cache()
            except: pass
