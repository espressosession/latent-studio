"""ADV polish (optional): a 2x latent upscaler for a single generated image, via
diffusers' own stabilityai/sd-x2-latent-upscaler — deliberately not img2img on the
base SD1.5 pipeline (which starts duplicating anatomy/composition well before
1024px) and not a GAN upscaler (would add a non-diffusers dependency with its own
version-pinning risk). The core app works fully without this module ever loading."""

import gc
import os

import torch
from diffusers import StableDiffusionLatentUpscalePipeline
from PIL import Image

from .device import empty_cache, get_device, get_dtype

UPSCALER_REPO = "stabilityai/sd-x2-latent-upscaler"

# Opt-in flag, off by default everywhere (locally and in the Colab notebook) — this
# is still a rough-edged, experimental feature (the button is labelled "(Experimental)"
# in the UI for the same reason) that needs more tweaking before it's on by default.
# Set ENABLE_UPSCALING=1 locally, or flip the Colab notebook's top-of-notebook
# "Activate Upscaling Pipeline (Experimental)" toggle, to turn it on.
UPSCALING_ENABLED = bool(os.environ.get("ENABLE_UPSCALING"))


class UpscalerManager:
    def __init__(self) -> None:
        self.device = get_device()
        self.dtype = get_dtype()
        self.pipe: StableDiffusionLatentUpscalePipeline | None = None

    @property
    def is_ready(self) -> bool:
        return self.pipe is not None

    def load(self, on_status=None) -> None:
        on_status = on_status or (lambda _msg: None)
        if self.pipe is not None:
            return

        on_status("Loading the upscaler...")
        # low_cpu_mem_usage=False avoids accelerate's meta-device load path — same fix
        # as pipeline_manager.load_checkpoint and controlnet.load.
        kwargs = {"torch_dtype": self.dtype, "low_cpu_mem_usage": False}
        try:
            pipe = StableDiffusionLatentUpscalePipeline.from_pretrained(
                UPSCALER_REPO, variant="fp16", **kwargs
            ).to(self.device)
        except Exception:
            pipe = StableDiffusionLatentUpscalePipeline.from_pretrained(
                UPSCALER_REPO, **kwargs
            ).to(self.device)

        # is_ready reads `self.pipe is not None` and is polled from the state-panel
        # thread without a lock — assign self.pipe only once it's fully built, same
        # ordering fix as pipeline_manager.py / controlnet.py.
        self.pipe = pipe
        on_status("Upscaler ready.")

    def unload(self) -> None:
        if self.pipe is not None:
            self.pipe = None
            gc.collect()
            empty_cache(self.device)

    def upscale(self, image: Image.Image, prompt: str, seed: int, steps: int = 20) -> Image.Image:
        if self.pipe is None:
            raise RuntimeError("Load the upscaler first.")
        gen_device = self.device if self.device != "mps" else "cpu"  # mps generator support is unreliable
        generator = torch.Generator(device=gen_device).manual_seed(seed)
        try:
            result = self.pipe(
                prompt=prompt,
                image=image,
                num_inference_steps=steps,
                guidance_scale=0,
                generator=generator,
            )
        except RuntimeError as exc:
            # The caller caps chaining at MAX_UPSCALE_HOPS (metadata_view.upscale_hops)
            # since each hop doubles both dimensions with no natural ceiling — but that
            # cap is a measured-safe default, not a hard guarantee for every starting
            # size/device, so this stays a real, reachable failure to catch here too.
            # empty_cache() gives the *next* call a clean slate instead of failing
            # again on fragmented-but-unallocated memory.
            if "out of memory" not in str(exc).lower():
                raise
            empty_cache(self.device)
            w, h = image.size
            raise RuntimeError(
                f"Ran out of GPU memory upscaling a {w}×{h} image to {w * 2}×{h * 2}. "
                "Try upscaling from a smaller/less-upscaled image, or free up GPU "
                "memory by switching models first."
            ) from None
        return result.images[0]
