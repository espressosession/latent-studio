"""Loads/unloads the SD1.5 base checkpoint and LoRA weights, keeping exactly one
active pipeline at a time. Base-model switches fully reload; LoRA identity
switches are cheap; LoRA *weight* is free — applied per-call, not reloaded."""

import gc
import os
import shutil
from typing import Callable

import torch
from diffusers import DiffusionPipeline, StableDiffusionPipeline
from huggingface_hub import hf_hub_download

from .device import empty_cache, get_device, get_dtype
from .registry import CHECKPOINTS, LORAS, get_checkpoint, get_lora
from .upscaler import UPSCALER_REPO

StatusCallback = Callable[[str], None] | None

# A local-debugging escape hatch — the public-facing app must always run with the
# safety checker enabled. The Colab notebook exposes a Settings-section toggle for
# this (off by default, leave it off for a public run); local dev can also just set
# the env var directly.
_SAFETY_CHECKER_DISABLED = bool(os.environ.get("DISABLE_SAFETY_CHECKER"))

# Where training/export.py stages locally-trained LoRAs; preferred over the HF
# repo id when present, so a fully-local training run drives the app with no network.
APP_LORAS_DIR = "app_loras"


def _noop(_: str) -> None:
    pass


def preload_models(on_status: StatusCallback = None) -> None:
    """Downloads every checkpoint + LoRA + the upscaler into the local cache without
    loading any into VRAM, so a live demo's first model switch — or first Upscale
    click — isn't network-bound."""
    on_status = on_status or _noop
    for cp in CHECKPOINTS:
        on_status(f"Fetching {cp.label}…")
        try:
            # fp16 halves the download; not every checkpoint ships that variant.
            StableDiffusionPipeline.download(cp.repo_id, variant="fp16", use_safetensors=True)
        except Exception:
            StableDiffusionPipeline.download(cp.repo_id, use_safetensors=True)
    on_status("Fetching the upscaler…")
    try:
        DiffusionPipeline.download(UPSCALER_REPO, variant="fp16", use_safetensors=True)
    except Exception:
        DiffusionPipeline.download(UPSCALER_REPO, use_safetensors=True)
    os.makedirs(APP_LORAS_DIR, exist_ok=True)
    for lora in LORAS:
        if lora.path is None:
            continue
        local = os.path.join(APP_LORAS_DIR, f"{lora.id}-lora.safetensors")
        if os.path.exists(local):
            continue  # a local training run already staged it
        on_status(f"Fetching {lora.label}…")
        weights = hf_hub_download(lora.path, "pytorch_lora_weights.safetensors")
        shutil.copy(weights, local)
    on_status("All models cached.")


class PipelineManager:
    def __init__(self) -> None:
        self.device = get_device()
        self.dtype = get_dtype()
        self.pipe: StableDiffusionPipeline | None = None
        self.checkpoint_id: str | None = None
        self.lora_id: str = "none"

    @property
    def is_ready(self) -> bool:
        return self.pipe is not None

    def load_checkpoint(self, checkpoint_id: str, on_status: StatusCallback = None) -> None:
        on_status = on_status or _noop
        if self.pipe is not None and self.checkpoint_id == checkpoint_id:
            return  # already loaded, nothing to do

        checkpoint = get_checkpoint(checkpoint_id)

        if self.pipe is not None:
            on_status(f"Unloading previous model ({self.checkpoint_id})...")
            self._unload_pipe()

        on_status(f"Loading {checkpoint.label}...")
        safety_kwargs = (
            {"safety_checker": None, "requires_safety_checker": False}
            if _SAFETY_CHECKER_DISABLED
            else {}
        )
        # low_cpu_mem_usage=False avoids accelerate's meta-device load path, which on
        # some torch/accelerate combos (notably Colab) leaves .to(device) unable to
        # copy ("Cannot copy out of meta tensor") — same fix as controlnet.py's load().
        kwargs = {
            "torch_dtype": self.dtype, "use_safetensors": True,
            "low_cpu_mem_usage": False, **safety_kwargs,
        }

        # Ask for fp16 weights directly rather than downloading fp32 and casting down;
        # not every checkpoint ships an fp16 variant (epiCRealism doesn't), so fall back.
        if self.dtype == torch.float16:
            try:
                pipe = StableDiffusionPipeline.from_pretrained(
                    checkpoint.repo_id, variant="fp16", **kwargs
                ).to(self.device)
            except Exception:
                on_status(f"No fp16 build for {checkpoint.label} — loading full weights...")
                pipe = StableDiffusionPipeline.from_pretrained(
                    checkpoint.repo_id, **kwargs
                ).to(self.device)
        else:
            pipe = StableDiffusionPipeline.from_pretrained(
                checkpoint.repo_id, **kwargs
            ).to(self.device)

        # is_ready reads `self.pipe is not None` and is polled from another thread (the
        # state panel) without a lock — checkpoint_id must already be correct before
        # self.pipe flips non-None, not after. Build into a local, assign checkpoint_id
        # first, self.pipe last.
        self.checkpoint_id = checkpoint_id
        self.lora_id = "none"
        self.pipe = pipe
        on_status(f"{checkpoint.label} ready.")

    def set_lora(self, lora_id: str, on_status: StatusCallback = None) -> None:
        on_status = on_status or _noop
        if self.pipe is None:
            raise RuntimeError("Load a base checkpoint before selecting a LoRA.")
        if lora_id == self.lora_id:
            return

        if self.lora_id != "none":
            self.pipe.unload_lora_weights()

        lora = get_lora(lora_id)
        if lora.path is not None:
            local = os.path.join(APP_LORAS_DIR, f"{lora.id}-lora.safetensors")
            source = local if os.path.exists(local) else lora.path
            on_status(f"Loading LoRA {lora.label}...")
            # low_cpu_mem_usage=False: diffusers defaults this to True whenever peft/
            # transformers are recent enough (true here and on Colab), which takes the
            # same accelerate meta-device path as the checkpoint load above — and here
            # it left a layer's bias in its original fp32 instead of the pipe's fp16
            # ("Input type (c10::Half) and bias type (float) should be the same").
            self.pipe.load_lora_weights(source, low_cpu_mem_usage=False)

        self.lora_id = lora_id
        on_status(f"LoRA set to {lora.label}.")

    def _unload_pipe(self) -> None:
        # Plain assignment, not `del self.pipe` — the state panel's polling loop reads
        # `is_ready` from a different thread, and `del` briefly leaves the attribute
        # missing entirely, which raced into an AttributeError in production.
        self.pipe = None
        self.checkpoint_id = None
        self.lora_id = "none"
        gc.collect()
        empty_cache(self.device)

    def unload(self) -> None:
        if self.pipe is not None:
            self._unload_pipe()

    def generator(self, seed: int) -> torch.Generator:
        gen_device = self.device if self.device != "mps" else "cpu"  # mps generator support is unreliable
        return torch.Generator(device=gen_device).manual_seed(seed)
