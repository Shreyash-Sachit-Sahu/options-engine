"""
GPU Utilities for Options Engine
=================================
Auto-detects the best available device (CUDA → MPS → CPU) and wires it
into stable-baselines3 and any raw PyTorch code in the project.

Usage
-----
    from agent.gpu_utils import get_device, device_banner, patch_sb3_device

    device = get_device()          # e.g. "cuda:0", "mps", or "cpu"
    patch_sb3_device(device)       # sets SB3's global default

    # Pass device= to SAC and all torch tensors
    model = SAC("MlpPolicy", env, device=device, ...)

Diagnostics
-----------
    python agent/gpu_utils.py          # print full device report
    python agent/gpu_utils.py --bench  # run a quick matmul benchmark
"""

import os
import sys
import time
import argparse

import torch


# ─────────────────────────────────────────────────────────────────────────────
# Core: device detection
# ─────────────────────────────────────────────────────────────────────────────

def get_device(prefer: str = "auto", verbose: bool = True) -> str:
    """
    Return the best available torch device string.

    Priority order (when prefer="auto"):
        1. CUDA  — NVIDIA GPU via CUDA
        2. MPS   — Apple Silicon GPU (M1/M2/M3)
        3. CPU   — fallback

    Args:
        prefer  : "auto" | "cuda" | "mps" | "cpu"
                  Force a specific backend. Falls back to CPU if unavailable.
        verbose : Print selection rationale to stdout.

    Returns:
        Device string suitable for torch.device() or SB3's device= param.
        e.g. "cuda:0", "cuda:1", "mps", "cpu"
    """
    if prefer not in ("auto", "cuda", "mps", "cpu"):
        raise ValueError(f"prefer must be one of: auto, cuda, mps, cpu — got '{prefer}'")

    def _pick():
        if prefer == "cpu":
            return "cpu", "forced by --device cpu"

        # ── CUDA ──────────────────────────────────────────────────────────
        if prefer in ("auto", "cuda"):
            if torch.cuda.is_available():
                idx = torch.cuda.current_device()
                name = torch.cuda.get_device_name(idx)
                return f"cuda:{idx}", f"CUDA GPU detected: {name}"
            elif prefer == "cuda":
                print("[GPU] WARNING: CUDA requested but not available — falling back to CPU")
                return "cpu", "CUDA unavailable"

        # ── MPS (Apple Silicon) ───────────────────────────────────────────
        if prefer in ("auto", "mps"):
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps", "Apple MPS GPU detected"
            elif prefer == "mps":
                print("[GPU] WARNING: MPS requested but not available — falling back to CPU")
                return "cpu", "MPS unavailable"

        return "cpu", "No GPU found — using CPU"

    device, reason = _pick()
    if verbose:
        icon = "🟢" if device != "cpu" else "⚪"
        print(f"[GPU] {icon} Device selected: {device!r}  ({reason})")
    return device


# ─────────────────────────────────────────────────────────────────────────────
# Diagnostics banner
# ─────────────────────────────────────────────────────────────────────────────

def device_banner() -> None:
    """Print a detailed GPU / CPU diagnostics banner."""
    sep = "─" * 60
    print(sep)
    print("  PyTorch Device Report")
    print(sep)
    print(f"  PyTorch version : {torch.__version__}")
    print(f"  Python version  : {sys.version.split()[0]}")

    # CUDA
    if torch.cuda.is_available():
        n = torch.cuda.device_count()
        print(f"\n  CUDA available  : YES  ({n} device{'s' if n > 1 else ''})")
        for i in range(n):
            props = torch.cuda.get_device_properties(i)
            mem_gb = props.total_memory / 1024 ** 3
            print(f"    cuda:{i}  {props.name}  {mem_gb:.1f} GB  "
                  f"sm_{props.major}{props.minor}")
        # cuDNN
        if torch.backends.cudnn.is_available():
            print(f"  cuDNN           : {torch.backends.cudnn.version()}  "
                  f"(benchmark={'ON' if torch.backends.cudnn.benchmark else 'OFF'})")
    else:
        print("\n  CUDA available  : NO")
        _print_cuda_hints()

    # MPS
    has_mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    print(f"\n  MPS available   : {'YES' if has_mps else 'NO'}")

    # Recommended device
    device = get_device(verbose=False)
    print(f"\n  Recommended     : {device!r}")
    print(sep)


def _print_cuda_hints() -> None:
    """Print actionable CUDA troubleshooting hints."""
    hints = []

    # Check if torch was built without CUDA
    if "+cpu" in torch.__version__ or not hasattr(torch.version, "cuda") or torch.version.cuda is None:
        hints.append(
            "  Your PyTorch is the CPU-only build.\n"
            "  Install a CUDA-enabled build:\n"
            "    pip install torch --index-url https://download.pytorch.org/whl/cu121"
        )
    else:
        cuda_ver = torch.version.cuda
        hints.append(
            f"  PyTorch was built for CUDA {cuda_ver} but no GPU driver was detected.\n"
            "  Possible causes:\n"
            "    • No NVIDIA GPU in this machine\n"
            "    • NVIDIA driver not installed  →  nvidia-smi to verify\n"
            "    • Running inside a container without --gpus flag\n"
            "    • WSL2: GPU passthrough requires CUDA for WSL2 driver"
        )

    for h in hints:
        print(f"\n  HINT: {h}")


# ─────────────────────────────────────────────────────────────────────────────
# SB3 integration
# ─────────────────────────────────────────────────────────────────────────────

def patch_sb3_device(device: str) -> None:
    """
    Configure stable-baselines3 to default to the given device.

    SB3 uses th.device internally and also respects torch's default.
    Calling this ensures that any SAC/PPO/TD3 created after this call
    will use the right device even without passing device= explicitly.
    (Passing device= to the constructor is still the most reliable approach.)
    """
    if device.startswith("cuda"):
        idx = int(device.split(":")[1]) if ":" in device else 0
        torch.cuda.set_device(idx)
        # cuDNN benchmark mode speeds up fixed-size networks significantly
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False   # faster, non-deterministic
    elif device == "mps":
        os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")


# ─────────────────────────────────────────────────────────────────────────────
# Performance tuning helpers
# ─────────────────────────────────────────────────────────────────────────────

def recommended_batch_size(device: str, base: int = 256) -> int:
    """
    Return a batch size tuned for the device.

    GPU training benefits from larger batches to saturate memory bandwidth.
    CPU training does not, and large batches just waste RAM.

    Args:
        device : Device string from get_device().
        base   : The batch size you would use on CPU.

    Returns:
        Recommended batch size.
    """
    if device.startswith("cuda"):
        props = torch.cuda.get_device_properties(device)
        vram_gb = props.total_memory / 1024 ** 3
        # Scale: 8GB → 512, 16GB+ → 1024
        if vram_gb >= 16:
            return 1024
        elif vram_gb >= 8:
            return 512
        else:
            return 256
    elif device == "mps":
        return 512   # unified memory — generous batches are fine
    return base      # CPU: keep as-is


def recommended_buffer_size(device: str, base: int = 200_000) -> int:
    """Return a replay-buffer size tuned for the device."""
    if device.startswith("cuda"):
        props = torch.cuda.get_device_properties(device)
        vram_gb = props.total_memory / 1024 ** 3
        if vram_gb >= 16:
            return 1_000_000
        elif vram_gb >= 8:
            return 500_000
        else:
            return 200_000
    return base


# ─────────────────────────────────────────────────────────────────────────────
# Quick benchmark
# ─────────────────────────────────────────────────────────────────────────────

def run_benchmark(device: str, size: int = 2048, reps: int = 50) -> None:
    """
    Micro-benchmark: time a matmul on the given device vs CPU.
    Gives a rough idea of actual speedup for the network sizes used here.
    """
    print(f"\n[BENCH] Matrix multiply {size}×{size}  ×{reps} reps")

    dev = torch.device(device)
    a = torch.randn(size, size, device=dev)
    b = torch.randn(size, size, device=dev)

    # Warm-up
    for _ in range(5):
        _ = torch.mm(a, b)
    if device.startswith("cuda"):
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(reps):
        _ = torch.mm(a, b)
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    elapsed_gpu = time.perf_counter() - t0

    print(f"  {device:<10}: {elapsed_gpu*1000/reps:.2f} ms/op")

    # CPU baseline
    a_cpu = a.cpu()
    b_cpu = b.cpu()
    for _ in range(3):
        _ = torch.mm(a_cpu, b_cpu)
    t0 = time.perf_counter()
    for _ in range(min(reps, 10)):
        _ = torch.mm(a_cpu, b_cpu)
    elapsed_cpu = time.perf_counter() - t0
    elapsed_cpu_per = elapsed_cpu / min(reps, 10)

    print(f"  cpu       : {elapsed_cpu_per*1000:.2f} ms/op")
    if device != "cpu":
        speedup = elapsed_cpu_per / (elapsed_gpu / reps)
        print(f"  Speedup   : {speedup:.1f}×")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GPU diagnostics for options-engine")
    parser.add_argument("--device", default="auto",
                        choices=["auto", "cuda", "mps", "cpu"],
                        help="Which device to report on / benchmark")
    parser.add_argument("--bench", action="store_true",
                        help="Run matmul benchmark")
    args = parser.parse_args()

    device_banner()
    device = get_device(prefer=args.device)

    if args.bench:
        run_benchmark(device)