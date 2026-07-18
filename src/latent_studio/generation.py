"""Single-generation entry point. generate() dispatches to the plain SD1.5 path
or the ControlNet path and returns the image together with the metadata dict
that gets shown in the UI and embedded into the PNG."""

from dataclasses import asdict, dataclass, field

from PIL import Image

from .controlnet import ControlNetManager
from .pipeline_manager import PipelineManager
from .registry import get_checkpoint, get_lora


@dataclass
class GenerationParams:
    prompt: str
    negative_prompt: str
    width: int
    height: int
    cfg_scale: float
    steps: int
    seed: int
    lora_weight: float = 1.0
    controlnet_types: list[str] = field(default_factory=list)
    controlnet_scales: dict[str, float] = field(default_factory=dict)
    # Compare's own sweep target for Reference strength: the real per-type scale
    # lives in controlnet_scales, but that's a dict and dataclasses.replace() needs a
    # plain scalar field to sweep over (see grids.sweep) — None outside a sweep, so it
    # never shows up in metadata for an ordinary generation (generate() pops it below).
    controlnet_scale: float | None = None


def _effective_prompt(lora_id: str, prompt: str) -> str:
    """Auto-prepend the active LoRA's trigger word so the style actually fires
    without the user having to know/type it. No-op if there's no LoRA/trigger or
    the trigger is already present."""
    lora = get_lora(lora_id)
    if lora.path is None or not lora.trigger_word:
        return prompt
    if lora.trigger_word.lower() in prompt.lower():
        return prompt
    return f"{lora.trigger_word}, {prompt}".strip().rstrip(",").strip()


def _tag_hardware(metadata: dict, mgr) -> None:
    """Record the device + dtype the image was actually rendered on. Same seed and
    settings only reproduce exactly on the same backend — cuda/fp16 and mps/fp32
    diverge — so without this the metadata block can't uniquely identify its image."""
    metadata["device"] = mgr.device
    metadata["dtype"] = str(mgr.dtype).replace("torch.", "")


def _generate_plain(manager: PipelineManager, params: GenerationParams) -> tuple[Image.Image, str]:
    if not manager.is_ready:
        raise RuntimeError("No model loaded — select a checkpoint first.")

    # Only pass a scale if a LoRA is actually loaded (lora.path is not None).
    lora = get_lora(manager.lora_id)
    cross_attention_kwargs = {"scale": params.lora_weight} if lora.path is not None else None

    prompt = _effective_prompt(manager.lora_id, params.prompt)
    generator = manager.generator(params.seed)
    result = manager.pipe(
        prompt=prompt,
        negative_prompt=params.negative_prompt or None,
        width=params.width,
        height=params.height,
        guidance_scale=params.cfg_scale,
        num_inference_steps=params.steps,
        generator=generator,
        cross_attention_kwargs=cross_attention_kwargs,
    )
    # The safety checker replaces a flagged image with a plain black square and only
    # ever mentions it via a console warning — silently "succeeding" with that black
    # square would be far more confusing than a clear error in the state panel.
    if getattr(result, "nsfw_content_detected", None) and result.nsfw_content_detected[0]:
        raise RuntimeError(
            "The safety checker flagged this result as potential NSFW content and blocked it "
            "(a black square, not a real image) — try a different prompt or seed."
        )
    return result.images[0], prompt


def generate(
    manager: PipelineManager,
    controlnet_manager: ControlNetManager,
    params: GenerationParams,
    control_images: dict[str, Image.Image] | None = None,
) -> tuple[Image.Image, dict]:
    metadata = asdict(params)
    if metadata.get("controlnet_scale") is None:
        metadata.pop("controlnet_scale", None)  # sweep-only override, never a "real" recorded setting

    if params.controlnet_types:
        control_images = control_images or {}
        ordered_images = [control_images[t] for t in params.controlnet_types]
        scale_override = params.controlnet_scale
        ordered_scales = [
            scale_override if scale_override is not None else params.controlnet_scales[t]
            for t in params.controlnet_types
        ]
        if scale_override is not None:
            # Keep the recorded controlnet_scales in sync with what was actually
            # applied — otherwise a swept cell's metadata would show the unswept
            # base value, breaking "every image reproducible from its own metadata".
            metadata["controlnet_scales"] = {t: scale_override for t in params.controlnet_types}
        # LoRA (patches the UNet) and ControlNet (injects residuals) are orthogonal and combine freely.
        lora = get_lora(controlnet_manager.lora_id)
        lora_scale = params.lora_weight if lora.path is not None else None
        effective_prompt = _effective_prompt(controlnet_manager.lora_id, params.prompt)
        image = controlnet_manager.generate(
            prompt=effective_prompt,
            negative_prompt=params.negative_prompt,
            control_images=ordered_images,
            conditioning_scales=ordered_scales,
            cfg_scale=params.cfg_scale,
            steps=params.steps,
            seed=params.seed,
            lora_scale=lora_scale,
        )
        checkpoint = get_checkpoint(controlnet_manager.checkpoint_id)
        metadata["checkpoint_id"] = checkpoint.id
        metadata["checkpoint_label"] = checkpoint.label
        metadata["lora_id"] = lora.id
        metadata["lora_label"] = lora.label
        if lora.path is None:
            metadata.pop("lora_weight", None)
        elif effective_prompt != params.prompt:
            metadata["effective_prompt"] = effective_prompt  # trigger word auto-added
        _tag_hardware(metadata, controlnet_manager)
    else:
        image, effective_prompt = _generate_plain(manager, params)
        checkpoint = get_checkpoint(manager.checkpoint_id)
        lora = get_lora(manager.lora_id)
        metadata["checkpoint_id"] = checkpoint.id
        metadata["checkpoint_label"] = checkpoint.label
        metadata["lora_id"] = lora.id
        metadata["lora_label"] = lora.label
        if lora.path is None:
            metadata.pop("lora_weight", None)
        elif effective_prompt != params.prompt:
            metadata["effective_prompt"] = effective_prompt  # trigger word auto-added
        metadata.pop("controlnet_types", None)
        metadata.pop("controlnet_scales", None)
        _tag_hardware(metadata, manager)

    return image, metadata
