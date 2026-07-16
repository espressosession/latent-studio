"""Step 5 — the mandatory validation gate: a LoRA is "done" only once these grids
have been looked at (checkpoint comparison, weight sweep, on/off-domain
generalisation, plus a saturation-based recommended snapshot — see the
notebook's Step 5 section). Self-contained (no latent_studio import)."""

import glob
import os

import numpy as np
import torch
from diffusers import StableDiffusionPipeline
from PIL import Image, ImageDraw


def _device_dtype() -> tuple[str, torch.dtype]:
    if torch.cuda.is_available():
        return "cuda", torch.float16
    if torch.backends.mps.is_available():
        return "mps", torch.float16
    return "cpu", torch.float32


def _generator(device: str, seed: int) -> torch.Generator:
    gen_device = "cpu" if device == "mps" else device  # mps generator is unreliable
    return torch.Generator(device=gen_device).manual_seed(seed)


def _grid(images: list[Image.Image], rows: int, cols: int, labels=None, cell: int = 256) -> Image.Image:
    grid = Image.new("RGB", (cols * cell, rows * cell), (28, 28, 28))
    draw = ImageDraw.Draw(grid)
    for i, image in enumerate(images):
        r, c = divmod(i, cols)
        grid.paste(image.resize((cell, cell)), (c * cell, r * cell))
        if labels and i < len(labels) and labels[i]:
            draw.rectangle([c * cell, r * cell, c * cell + cell, r * cell + 14], fill=(0, 0, 0))
            draw.text((c * cell + 4, r * cell + 3), labels[i], fill=(255, 255, 255))
    return grid


def _burn_in_score(image: Image.Image) -> float:
    """Mean HSV saturation in [0, 1]; overfit/cooked styles oversaturate."""
    hsv = np.asarray(image.convert("HSV"), dtype=np.float32)
    return float(hsv[..., 1].mean() / 255.0)


def _checkpoint_step(name: str) -> int:
    if name == "final":
        return 10**9  # sort final last (it's the most-trained)
    return int(name.rsplit("-", 1)[-1]) if name.startswith("checkpoint-") else 0


def find_lora_dirs(output_dir: str) -> list[tuple[str, str]]:
    """(label, dir) for every LoRA snapshot that has loadable weights: the final
    weights in output_dir, plus any checkpoint-* subdir that carries a
    .safetensors (depends on the diffusers version's checkpoint save hook)."""
    dirs: list[tuple[str, str]] = []
    if glob.glob(os.path.join(output_dir, "*.safetensors")):
        dirs.append(("final", output_dir))
    for ckpt in glob.glob(os.path.join(output_dir, "checkpoint-*")):
        if glob.glob(os.path.join(ckpt, "*.safetensors")):
            dirs.append((os.path.basename(ckpt), ckpt))
    return sorted(dirs, key=lambda pair: _checkpoint_step(pair[0]))


def _recommend(scores: dict[str, float]) -> str | None:
    """Latest snapshot whose saturation isn't an outlier vs. the calmest one."""
    if not scores:
        return None
    calmest = min(scores.values())
    healthy = [name for name, score in scores.items() if score <= calmest * 1.25]
    pool = healthy or list(scores)
    return max(pool, key=_checkpoint_step)


def validate(
    output_dir: str,
    base_model: str,
    trigger_word: str,
    content_prompt: str,
    seeds: list[int],
    cfg: float = 7.5,
    steps: int = 30,
    weights: tuple[float, ...] = (0.5, 0.75, 1.0),
    offdomain_prompts: tuple[str, ...] = (),
) -> dict:
    """Runs the validation gate. Returns a dict with `checkpoint_grid`,
    `weight_grid`, `generalization_grid`, per-snapshot `scores`, the `recommended`
    snapshot label, and the discovered `lora_dirs`."""
    device, dtype = _device_dtype()
    pipe = StableDiffusionPipeline.from_pretrained(
        base_model, torch_dtype=dtype, safety_checker=None, requires_safety_checker=False
    ).to(device)

    styled_prompt = f"{trigger_word}, {content_prompt}"
    cols = len(seeds)

    def gen(prompt, seed, scale=None):
        kwargs = {"cross_attention_kwargs": {"scale": scale}} if scale is not None else {}
        return pipe(
            prompt=prompt, num_inference_steps=steps, guidance_scale=cfg,
            generator=_generator(device, seed), **kwargs,
        ).images[0]

    # --- grid 1: no-LoRA baseline + one row per snapshot at strength 1.0 ---
    baseline = [gen(content_prompt, s) for s in seeds]
    images = list(baseline)
    labels = [f"no-LoRA s={s}" for s in seeds]

    lora_dirs = find_lora_dirs(output_dir)
    scores: dict[str, float] = {}
    for name, path in lora_dirs:
        pipe.load_lora_weights(path)
        row = [gen(styled_prompt, s, scale=1.0) for s in seeds]
        pipe.unload_lora_weights()
        scores[name] = float(np.mean([_burn_in_score(im) for im in row]))
        images.extend(row)
        labels.extend(f"{name} s={s}" for s in seeds)

    checkpoint_grid = _grid(images, 1 + len(lora_dirs), cols, labels)

    # Pick the sweep snapshot before grids 2/3, so they judge the recommended
    # snapshot rather than `final`, which is often already past burn-in.
    recommended = _recommend(scores)
    sweep_label, sweep_path = None, None
    if lora_dirs:
        by_label = dict(lora_dirs)
        sweep_label = recommended if recommended in by_label else lora_dirs[-1][0]
        sweep_path = by_label[sweep_label]

    # --- grid 2: recommended snapshot across strengths (burn-in) ---
    weight_grid = None
    if sweep_path is not None:
        pipe.load_lora_weights(sweep_path)
        w_images, w_labels = [], []
        for w in weights:
            for s in seeds:
                w_images.append(gen(styled_prompt, s, scale=w))
                w_labels.append(f"w={w} s={s}")
        pipe.unload_lora_weights()
        weight_grid = _grid(w_images, len(weights), cols, w_labels)

    # --- grid 3: style generalisation across on/off-domain prompts (recommended) ---
    generalization_grid = None
    if sweep_path is not None and offdomain_prompts:
        pipe.load_lora_weights(sweep_path)
        gen_seeds = seeds[:2]  # keep the grid small — this is about subject, not seed spread
        prompts = [content_prompt, *offdomain_prompts]
        g_images, g_labels = [], []
        for prompt in prompts:
            for s in gen_seeds:
                g_images.append(gen(f"{trigger_word}, {prompt}", s, scale=1.0))
                g_labels.append(f"{prompt[:22]} s={s}")
        pipe.unload_lora_weights()
        generalization_grid = _grid(g_images, len(prompts), len(gen_seeds), g_labels)

    print("Burn-in scores (mean saturation, higher = more cooked):")
    for name, score in scores.items():
        print(f"  {name}: {score:.3f}")
    print(f"Recommended snapshot (CONFIRM VISUALLY): {recommended}")
    print(f"Weight + generalisation grids built on: {sweep_label}")

    return {
        "checkpoint_grid": checkpoint_grid,
        "weight_grid": weight_grid,
        "generalization_grid": generalization_grid,
        "scores": scores,
        "recommended": recommended,
        "sweep_snapshot": sweep_label,
        "lora_dirs": lora_dirs,
    }


def render_snapshot_grids(
    snapshot_dir: str,
    base_model: str,
    trigger_word: str,
    content_prompt: str,
    seeds: list[int],
    cfg: float = 7.5,
    steps: int = 30,
    weights: tuple[float, ...] = (0.0, 0.5, 1.0, 1.5),
    offdomain_prompts: tuple[str, ...] = (),
) -> dict:
    """Re-render just the weight-sweep + generalisation grids for ONE snapshot dir
    — no checkpoint comparison, no retraining. Returns {"weight_grid",
    "generalization_grid"}."""
    device, dtype = _device_dtype()
    pipe = StableDiffusionPipeline.from_pretrained(
        base_model, torch_dtype=dtype, safety_checker=None, requires_safety_checker=False
    ).to(device)

    def gen(prompt, seed, scale=None):
        kwargs = {"cross_attention_kwargs": {"scale": scale}} if scale is not None else {}
        return pipe(
            prompt=prompt, num_inference_steps=steps, guidance_scale=cfg,
            generator=_generator(device, seed), **kwargs,
        ).images[0]

    styled_prompt = f"{trigger_word}, {content_prompt}"
    cols = len(seeds)
    pipe.load_lora_weights(snapshot_dir)

    # weight sweep (weights x seeds) on the on-domain prompt
    w_images, w_labels = [], []
    for w in weights:
        for s in seeds:
            w_images.append(gen(styled_prompt, s, scale=w))
            w_labels.append(f"w={w} s={s}")
    weight_grid = _grid(w_images, len(weights), cols, w_labels)

    # generalisation across on/off-domain prompts (first 2 seeds) at strength 1.0
    generalization_grid = None
    if offdomain_prompts:
        gen_seeds = seeds[:2]
        prompts = [content_prompt, *offdomain_prompts]
        g_images, g_labels = [], []
        for prompt in prompts:
            for s in gen_seeds:
                g_images.append(gen(f"{trigger_word}, {prompt}", s, scale=1.0))
                g_labels.append(f"{prompt[:22]} s={s}")
        generalization_grid = _grid(g_images, len(prompts), len(gen_seeds), g_labels)

    pipe.unload_lora_weights()
    del pipe
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {"weight_grid": weight_grid, "generalization_grid": generalization_grid}
