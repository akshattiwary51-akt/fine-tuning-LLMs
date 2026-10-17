"""
Environment detection helpers.

This pipeline requires an NVIDIA CUDA GPU (bitsandbytes 4-bit quantization and
Unsloth's fused kernels are CUDA-only; there is no CPU or Intel/AMD-integrated-
GPU fallback). Failing fast with a clear message here is much more useful than
letting the failure happen deep inside bitsandbytes/Unsloth with a cryptic
traceback — especially in notebook environments (Colab/Kaggle) where the most
common mistake is simply not having switched the runtime to a GPU yet.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def require_cuda() -> None:
    """Raises a clear, actionable RuntimeError if no CUDA GPU is available.

    Call this at the start of any entrypoint that will load model weights
    (training, inference) — before any heavy import/download happens.
    """
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            "No CUDA GPU detected. This project requires an NVIDIA GPU with CUDA "
            "(bitsandbytes 4-bit quantization and Unsloth are CUDA-only; there is "
            "no CPU / Intel / AMD-integrated-GPU fallback).\n\n"
            "If you're running locally: verify `nvidia-smi` works and that your "
            "PyTorch build matches your installed CUDA version.\n\n"
            "If you're on a laptop without an NVIDIA GPU (e.g. Intel Iris Xe, "
            "AMD integrated graphics, Apple Silicon): use a cloud GPU instead — "
            "Google Colab or Kaggle Notebooks both offer a free NVIDIA T4. See "
            "docs/COLAB.md for exact copy-paste steps, or the README's "
            "'Running Without a Local GPU' section.\n\n"
            "If you're on Colab specifically: Runtime > Change runtime type > "
            "Hardware accelerator > GPU (T4), then re-run this cell."
        )


def describe_gpu() -> str:
    """Returns a short, human-readable description of the active GPU, for
    logging at the start of a training/inference run. Assumes require_cuda()
    has already been called (does not itself validate CUDA availability)."""
    import torch

    if not torch.cuda.is_available():
        return "No CUDA GPU available."

    idx = torch.cuda.current_device()
    name = torch.cuda.get_device_name(idx)
    total_mem_gb = torch.cuda.get_device_properties(idx).total_memory / (1024**3)
    return f"{name} ({total_mem_gb:.1f} GB total VRAM)"


def reset_peak_memory_stats() -> None:
    """Resets CUDA's peak-memory counters. Call once at the start of training
    so `gpu_memory_stats()` afterwards reports the peak for THIS run, not
    whatever happened earlier in the process."""
    import torch

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def supports_bf16() -> bool:
    """Whether the active GPU supports bfloat16 in hardware.

    bfloat16 requires Ampere-generation or newer NVIDIA GPUs (A100, RTX
    30xx/40xx, L4, ...). Older GPUs — notably the Tesla T4 that Colab's free
    tier provides (Turing, compute capability 7.5) — do not support it, and
    `TrainingArguments(bf16=True)` raises a hard error on them rather than
    silently falling back. Training precision must be chosen based on this,
    not assumed; see app/train.py for where this drives bf16 vs fp16.
    Returns False (the safe default) if no GPU is available at all.
    """
    import torch

    if not torch.cuda.is_available():
        return False
    return torch.cuda.is_bf16_supported()


def gpu_memory_stats() -> dict:
    """Returns current and peak GPU memory usage in GB, for logging/saving
    alongside a training run (Phase 11: VRAM usage should be measured, not
    just claimed). Returns zeros if no GPU is available rather than raising,
    since this is diagnostic information, not a hard requirement to run."""
    import torch

    if not torch.cuda.is_available():
        return {
            "allocated_gb": 0.0,
            "reserved_gb": 0.0,
            "max_allocated_gb": 0.0,
            "max_reserved_gb": 0.0,
        }

    bytes_to_gb = 1024**3
    return {
        "allocated_gb": round(torch.cuda.memory_allocated() / bytes_to_gb, 3),
        "reserved_gb": round(torch.cuda.memory_reserved() / bytes_to_gb, 3),
        "max_allocated_gb": round(torch.cuda.max_memory_allocated() / bytes_to_gb, 3),
        "max_reserved_gb": round(torch.cuda.max_memory_reserved() / bytes_to_gb, 3),
    }


def log_environment_summary() -> None:
    """Logs GPU + key package versions. Cheap, useful at the start of any run,
    especially in ephemeral notebook environments where "what's actually
    installed here right now" is easy to lose track of."""
    import torch

    logger.info("CUDA available: %s", torch.cuda.is_available())
    if torch.cuda.is_available():
        logger.info("GPU: %s", describe_gpu())
    logger.info("PyTorch: %s", torch.__version__)
