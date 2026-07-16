"""ADV (optional): ControlNet conditioning via Canny edges and/or
Depth-Anything-V2 depth maps, ported from L06/Controlnet_Canny_01_edited.ipynb.
Supports single or simultaneous multi-ControlNet with per-net conditioning
scales. The core app (app.py) works without ever importing this module."""

import gc
import os

import cv2
import numpy as np
import torch
from diffusers import ControlNetModel, StableDiffusionControlNetPipeline, UniPCMultistepScheduler
from PIL import Image

from .device import empty_cache, get_device, get_dtype
from .pipeline_manager import APP_LORAS_DIR
from .registry import get_checkpoint, get_lora

CONTROLNET_REPOS = {
    "canny": "lllyasviel/sd-controlnet-canny",
    "depth": "lllyasviel/sd-controlnet-depth",
}

_depth_processor = None
_depth_model = None


def canny_preprocess(image: Image.Image, low_threshold: int = 100, high_threshold: int = 200) -> Image.Image:
    array = np.array(image.convert("RGB"))
    edges = cv2.Canny(array, low_threshold, high_threshold)
    edges = np.stack([edges] * 3, axis=-1)
    return Image.fromarray(edges)


def _load_depth_model():
    global _depth_processor, _depth_model
    if _depth_model is None:
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation

        _depth_processor = AutoImageProcessor.from_pretrained("Depth-Anything/Depth-Anything-V2-base-hf")
        _depth_model = AutoModelForDepthEstimation.from_pretrained("Depth-Anything/Depth-Anything-V2-base-hf")
    return _depth_processor, _depth_model


def unload_depth_model() -> None:
    """Frees the standalone depth-estimation model. It has no other owner and, unlike
    the ControlNet/SD pipelines, was never freed anywhere — once loaded it stayed
    resident in system RAM (it runs on CPU, not the GPU pipeline) for the rest of the
    process, compounding with whatever checkpoint loads next until the runtime ran out
    of RAM. Called from ControlNetManager.unload() so it shares the pipeline's own
    lifecycle instead of outliving it."""
    global _depth_processor, _depth_model
    if _depth_model is not None:
        _depth_processor = None
        _depth_model = None
        gc.collect()


def depth_preprocess(image: Image.Image) -> Image.Image:
    processor, model = _load_depth_model()
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        predicted_depth = model(**inputs).predicted_depth

    depth = torch.nn.functional.interpolate(
        predicted_depth.unsqueeze(1),
        size=image.size[::-1],
        mode="bicubic",
        align_corners=False,
    )[0, 0]
    depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
    depth_array = (depth.numpy() * 255).astype(np.uint8)
    return Image.fromarray(depth_array).convert("RGB")


class ControlNetManager:
    """Separate from PipelineManager since it needs its own pipeline class
    (StableDiffusionControlNetPipeline) and a different reload trigger: the
    active set of ControlNet *types*, not just the checkpoint."""

    def __init__(self) -> None:
        self.device = get_device()
        self.dtype = get_dtype()
        self.pipe: StableDiffusionControlNetPipeline | None = None
        self.checkpoint_id: str | None = None
        self.active_types: list[str] = []
        self.lora_id: str = "none"

    @property
    def is_ready(self) -> bool:
        return self.pipe is not None

    def load(self, checkpoint_id: str, controlnet_types: list[str], on_status=None) -> None:
        on_status = on_status or (lambda _msg: None)
        if self.pipe is not None and self.checkpoint_id == checkpoint_id and self.active_types == controlnet_types:
            return

        if self.pipe is not None:
            self.unload()

        checkpoint = get_checkpoint(checkpoint_id)
        on_status(f"Loading {checkpoint.label} + ControlNet ({', '.join(controlnet_types)})...")
        # low_cpu_mem_usage=False avoids accelerate's meta-device load path, which on
        # some torch/accelerate combos (notably Colab) leaves .to(device) unable to copy.
        controlnets = [
            ControlNetModel.from_pretrained(CONTROLNET_REPOS[t], torch_dtype=self.dtype, low_cpu_mem_usage=False)
            for t in controlnet_types
        ]
        controlnet_arg = controlnets[0] if len(controlnets) == 1 else controlnets

        pipe = StableDiffusionControlNetPipeline.from_pretrained(
            checkpoint.repo_id, controlnet=controlnet_arg, torch_dtype=self.dtype, low_cpu_mem_usage=False
        ).to(self.device)
        pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

        # is_ready reads `self.pipe is not None` and is polled from another thread (the
        # state panel) without a lock — every other attribute it depends on must already
        # be correct before self.pipe flips non-None, not after. Build the pipeline into
        # a local first, then assign checkpoint_id/active_types/lora_id, and only then
        # self.pipe last — same fix as pipeline_manager.load_checkpoint.
        self.checkpoint_id = checkpoint_id
        self.active_types = controlnet_types
        self.lora_id = "none"  # a fresh pipeline carries no adapter
        self.pipe = pipe
        on_status("ControlNet pipeline ready.")

    def set_lora(self, lora_id: str, on_status=None) -> None:
        """Load/unload a style LoRA on the ControlNet pipeline — mirror of
        PipelineManager.set_lora, so a style can be combined with a reference image."""
        on_status = on_status or (lambda _msg: None)
        if self.pipe is None:
            raise RuntimeError("Load a ControlNet pipeline before selecting a LoRA.")
        if lora_id == self.lora_id:
            return
        if self.lora_id != "none":
            self.pipe.unload_lora_weights()
        lora = get_lora(lora_id)
        if lora.path is not None:
            local = os.path.join(APP_LORAS_DIR, f"{lora.id}-lora.safetensors")
            source = local if os.path.exists(local) else lora.path
            on_status(f"Loading LoRA {lora.label}...")
            # low_cpu_mem_usage=False: see the identical note in pipeline_manager.set_lora.
            self.pipe.load_lora_weights(source, low_cpu_mem_usage=False)
        self.lora_id = lora_id
        on_status(f"LoRA set to {lora.label}.")

    def unload(self) -> None:
        if self.pipe is not None:
            # Plain assignment, not `del self.pipe` — a concurrent reader of `is_ready`
            # (the state panel polls from another thread) would briefly see the
            # attribute missing entirely instead of just None. See pipeline_manager.py.
            self.pipe = None
            self.checkpoint_id = None
            self.active_types = []
            self.lora_id = "none"
            gc.collect()
            empty_cache(self.device)
        # Unconditional, not nested in the branch above: Process can load the depth
        # model without ever running a generation (self.pipe staying None), and this
        # is the only place that ever frees it.
        unload_depth_model()

    def generate(
        self,
        prompt: str,
        negative_prompt: str,
        control_images: list[Image.Image],
        conditioning_scales: list[float],
        cfg_scale: float,
        steps: int,
        seed: int,
        lora_scale: float | None = None,
    ) -> Image.Image:
        if self.pipe is None:
            raise RuntimeError("Load a ControlNet pipeline first.")

        control_image_arg = control_images[0] if len(control_images) == 1 else control_images
        # diffusers checks `isinstance(controlnet_conditioning_scale, float)` for the
        # single-ControlNet case and rejects ints / numpy scalars — cast explicitly.
        scales = [float(s) for s in conditioning_scales]
        scale_arg = scales[0] if len(scales) == 1 else scales
        gen_device = self.device if self.device != "mps" else "cpu"
        generator = torch.Generator(device=gen_device).manual_seed(seed)
        # Same knob as the plain pipeline: the LoRA weight rides on the UNet cross-
        # attention, so ControlNet residuals and the style adapter stack cleanly.
        cross_attention_kwargs = {"scale": lora_scale} if lora_scale is not None else None

        result = self.pipe(
            prompt=prompt,
            image=control_image_arg,
            negative_prompt=negative_prompt or None,
            controlnet_conditioning_scale=scale_arg,
            guidance_scale=cfg_scale,
            num_inference_steps=steps,
            generator=generator,
            cross_attention_kwargs=cross_attention_kwargs,
        )
        # See generation._generate_plain's identical check: the safety checker swaps a
        # flagged image for a plain black square and only ever says so via a console
        # warning otherwise.
        if getattr(result, "nsfw_content_detected", None) and result.nsfw_content_detected[0]:
            raise RuntimeError(
                "The safety checker flagged this result as potential NSFW content and blocked it "
                "(a black square, not a real image) — try a different prompt or seed."
            )
        return result.images[0]
