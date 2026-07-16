"""Device/dtype selection — same cuda/mps/cpu logic as pipeline_V5, so the app
behaves identically on the local MacBook (MPS) and on Colab (CUDA T4)."""

import torch


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_dtype() -> torch.dtype:
    # fp16 everywhere except plain CPU (brief requires fp16 for VRAM budget on the T4).
    if torch.cuda.is_available() or torch.backends.mps.is_available():
        return torch.float16
    return torch.float32


def empty_cache(device: str) -> None:
    if device == "cuda":
        torch.cuda.empty_cache()
        if hasattr(torch.cuda, "ipc_collect"):
            torch.cuda.ipc_collect()
    elif device == "mps" and hasattr(torch.mps, "empty_cache"):
        torch.mps.empty_cache()
