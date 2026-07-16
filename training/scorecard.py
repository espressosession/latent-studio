"""Step 5b — the scorecard: does this LoRA look like the artist's real work?
Every metric is a delta vs. the same prompt+seed without the LoRA (style_gain,
prompt_drop, burn_in — see the notebook's Step 5b section). Takes a Hub repo id
or a local snapshot dir; self-contained (no latent_studio import)."""

import glob
import os

import numpy as np
import torch
from diffusers import StableDiffusionPipeline
from PIL import Image, ImageDraw
from transformers import CLIPModel, CLIPProcessor

from .config import (
    ARTISTS,
    BASE_MODEL,
    DOWNLOAD_WIDTH,
    MAX_ASPECT_RATIO,
    MIN_IMAGE_SIZE,
    RESOLUTION,
    SCORECARD_CLIP_MODEL,
    SCORECARD_MAX_BURN_IN,
    SCORECARD_MAX_PROMPT_DROP,
    SCORECARD_MIN_STYLE_GAIN,
    SCORECARD_REFERENCE_COUNT,
    SCORECARD_SEEDS,
    SCORECARD_WEIGHTS,
    VALIDATION_CFG,
    VALIDATION_OFFDOMAIN_PROMPTS,
    VALIDATION_STEPS,
)
from .download import download_artist
from .preprocess import preprocess_folder


def _device_dtype() -> tuple[str, torch.dtype]:
    if torch.cuda.is_available():
        return "cuda", torch.float16
    if torch.backends.mps.is_available():
        return "mps", torch.float16
    return "cpu", torch.float32


def _generator(device: str, seed: int) -> torch.Generator:
    gen_device = "cpu" if device == "mps" else device  # mps generator is unreliable
    return torch.Generator(device=gen_device).manual_seed(seed)


def _saturation(image: Image.Image) -> float:
    hsv = np.asarray(image.convert("HSV"), dtype=np.float32)
    return float(hsv[..., 1].mean() / 255.0)


# --- reference corpus ---------------------------------------------------------

def reference_images(artist_id: str, work_dir: str = "lora_work", count: int = SCORECARD_REFERENCE_COUNT) -> list[Image.Image]:
    """The artist's real artworks, to measure our generations against. Prefers the
    local training set; otherwise re-fetches + preprocesses a fresh CC0 sample from
    AIC, so a LoRA can be graded even when its training set lives elsewhere."""
    artist = ARTISTS[artist_id]
    dataset = os.path.join(work_dir, artist_id, "dataset")
    paths = sorted(glob.glob(os.path.join(dataset, "*.png")))[:count]

    if not paths:
        clean = os.path.join(work_dir, artist_id, "reference")
        paths = sorted(glob.glob(os.path.join(clean, "*.png")))[:count]
        if not paths:
            raw = os.path.join(work_dir, artist_id, "reference_raw")
            print(f"No local training set for '{artist_id}' — fetching a reference sample from AIC.")
            download_artist(
                artist.search_name, raw, artist.artist_match,
                target=count * 2, image_width=DOWNLOAD_WIDTH,
            )
            preprocess_folder(
                raw, clean, resolution=RESOLUTION, min_size=MIN_IMAGE_SIZE,
                max_aspect=MAX_ASPECT_RATIO, target=count,
            )
            paths = sorted(glob.glob(os.path.join(clean, "*.png")))[:count]

    if not paths:
        raise SystemExit(f"No reference images available for '{artist_id}'.")
    return [Image.open(p).convert("RGB") for p in paths]


# --- CLIP ---------------------------------------------------------------------

_clip_cache: dict[str, tuple] = {}


def _clip(model_name: str = SCORECARD_CLIP_MODEL):
    """CLIP, loaded once per process. fp32 on purpose — it is small, and the metrics
    are differences of cosines, where fp16 noise is the same size as the signal."""
    if model_name not in _clip_cache:
        device, _ = _device_dtype()
        model = CLIPModel.from_pretrained(model_name).to(device).eval()
        processor = CLIPProcessor.from_pretrained(model_name)
        _clip_cache[model_name] = (model, processor, device)
    return _clip_cache[model_name]


def _as_embedding(features) -> torch.Tensor:
    """transformers 4 returns the projected embedding as a plain tensor; transformers 5
    wraps it in a BaseModelOutputWithPooling (the embedding is `pooler_output`). Colab
    and this laptop are on different major versions, so accept both."""
    if isinstance(features, torch.Tensor):
        return features
    return features.pooler_output


def embed_images(images: list[Image.Image]) -> torch.Tensor:
    model, processor, device = _clip()
    with torch.no_grad():
        inputs = processor(images=images, return_tensors="pt").to(device)
        features = _as_embedding(model.get_image_features(**inputs))
    return torch.nn.functional.normalize(features, dim=-1).float().cpu()


def embed_texts(texts: list[str]) -> torch.Tensor:
    model, processor, device = _clip()
    with torch.no_grad():
        inputs = processor(text=texts, return_tensors="pt", padding=True, truncation=True).to(device)
        features = _as_embedding(model.get_text_features(**inputs))
    return torch.nn.functional.normalize(features, dim=-1).float().cpu()


def style_centroid(images: list[Image.Image]) -> torch.Tensor:
    """One vector standing for "what this artist's work looks like to CLIP"."""
    return torch.nn.functional.normalize(embed_images(images).mean(dim=0), dim=-1)


# --- contact sheet ------------------------------------------------------------

def _sheet(rows: list[tuple[str, list[Image.Image]]], cell: int = 224) -> Image.Image:
    """Rows of (label, images). Row 0 is the artist's real work; the rest are ours,
    one row per LoRA strength, columns aligned by prompt."""
    cols = max(len(images) for _, images in rows)
    label_h = 18
    sheet = Image.new("RGB", (cols * cell, len(rows) * (cell + label_h)), (24, 24, 24))
    draw = ImageDraw.Draw(sheet)
    for r, (label, images) in enumerate(rows):
        y = r * (cell + label_h)
        draw.text((4, y + 4), label, fill=(255, 255, 255))
        for c, image in enumerate(images[:cols]):
            sheet.paste(image.resize((cell, cell)), (c * cell, y + label_h))
    return sheet


# --- the scorecard itself -----------------------------------------------------

def score_lora(
    artist_id: str,
    lora_source: str | None = None,
    work_dir: str = "lora_work",
    weights: list[float] = SCORECARD_WEIGHTS,
    seeds: list[int] = SCORECARD_SEEDS,
    cfg: float = VALIDATION_CFG,
    steps: int = VALIDATION_STEPS,
    reference_count: int = SCORECARD_REFERENCE_COUNT,
) -> dict:
    """Grade one LoRA against the artist's real work. `lora_source` is a HF repo id
    or a local snapshot dir; None grades whatever is published (artist.hf_repo).
    Returns metrics per weight, a `recommended_weight`, `verdict`, and `sheet`."""
    artist = ARTISTS[artist_id]
    source = lora_source or artist.hf_repo
    device, dtype = _device_dtype()

    reference = reference_images(artist_id, work_dir, reference_count)
    centroid = style_centroid(reference)

    ondomain = artist.validation_prompt
    offdomain = list(VALIDATION_OFFDOMAIN_PROMPTS)
    prompts = [ondomain, *offdomain]

    pipe = StableDiffusionPipeline.from_pretrained(
        BASE_MODEL, torch_dtype=dtype, safety_checker=None, requires_safety_checker=False
    ).to(device)

    def render(prompt: str, seed: int, scale: float | None) -> Image.Image:
        # The trigger only belongs in the prompt when the adapter is loaded; the
        # baseline must be a *clean* base-model render of the same words.
        text = f"{artist.trigger_word}, {prompt}" if scale is not None else prompt
        kwargs = {"cross_attention_kwargs": {"scale": scale}} if scale is not None else {}
        return pipe(
            prompt=text, num_inference_steps=steps, guidance_scale=cfg,
            generator=_generator(device, seed), **kwargs,
        ).images[0]

    print(f"[{artist_id}] scoring {source} against {len(reference)} real artworks")

    # Baseline: no LoRA at all. Every metric below is measured relative to this.
    baseline = {(p, s): render(p, s, None) for p in prompts for s in seeds}

    pipe.load_lora_weights(source)
    styled = {
        (w, p, s): render(p, s, w)
        for w in weights for p in prompts for s in seeds
    }
    pipe.unload_lora_weights()

    def cosine_to_corpus(images: list[Image.Image]) -> float:
        return float((embed_images(images) @ centroid).mean())

    def cosine_to_prompts(pairs: list[tuple[Image.Image, str]]) -> float:
        images, texts = zip(*pairs)
        image_emb, text_emb = embed_images(list(images)), embed_texts(list(texts))
        return float((image_emb * text_emb).sum(dim=-1).mean())

    # style_gain rides on the off-domain prompts only: the corpus cannot contain a
    # bicycle, so similarity to the corpus there is style and nothing else.
    base_off = [baseline[(p, s)] for p in offdomain for s in seeds]
    base_style = cosine_to_corpus(base_off)
    base_prompt_fit = cosine_to_prompts([(baseline[(p, s)], p) for p in prompts for s in seeds])
    base_saturation = float(np.mean([_saturation(im) for im in baseline.values()]))

    metrics: dict[float, dict[str, float]] = {}
    for w in weights:
        lora_off = [styled[(w, p, s)] for p in offdomain for s in seeds]
        all_lora = [styled[(w, p, s)] for p in prompts for s in seeds]
        metrics[w] = {
            "style_gain": cosine_to_corpus(lora_off) - base_style,
            "prompt_drop": base_prompt_fit
            - cosine_to_prompts([(styled[(w, p, s)], p) for p in prompts for s in seeds]),
            "burn_in": float(np.mean([_saturation(im) for im in all_lora])) / max(base_saturation, 1e-6),
        }

    recommended = _recommend_weight(metrics)
    verdict = _verdict(metrics, recommended)

    rows: list[tuple[str, list[Image.Image]]] = [
        (f"REAL {artist.label} (training data)", reference[: len(prompts)]),
        ("no LoRA (base SD 1.5)", [baseline[(p, seeds[0])] for p in prompts]),
    ]
    for w in weights:
        m = metrics[w]
        rows.append((
            f"LoRA w={w}  style {m['style_gain']:+.3f}  prompt {-m['prompt_drop']:+.3f}  burn {m['burn_in']:.2f}x",
            [styled[(w, p, seeds[0])] for p in prompts],
        ))
    sheet = _sheet(rows)

    for w in weights:
        m = metrics[w]
        print(
            f"  w={w}: style_gain {m['style_gain']:+.3f} | prompt_drop {m['prompt_drop']:+.3f} "
            f"| burn_in {m['burn_in']:.2f}x"
        )
    print(f"  -> {verdict} (best strength: {recommended})")

    del pipe
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "artist": artist_id,
        "source": source,
        "metrics": metrics,
        "recommended_weight": recommended,
        "verdict": verdict,
        "passed": verdict == SHIP,  # what export_artist gates the Hub push on
        "sheet": sheet,
        "reference_count": len(reference),
    }


def _recommend_weight(metrics: dict[float, dict[str, float]]) -> float | None:
    """The strongest strength that still follows the prompt and hasn't cooked —
    i.e. the most style you can have without paying for it elsewhere. This is a
    sensible default for the app's Style-strength slider, per LoRA."""
    healthy = [
        w for w, m in metrics.items()
        if m["prompt_drop"] <= SCORECARD_MAX_PROMPT_DROP and m["burn_in"] <= SCORECARD_MAX_BURN_IN
    ]
    return max(healthy) if healthy else None


SHIP = "SHIP"


def _verdict(metrics: dict[float, dict[str, float]], recommended: float | None) -> str:
    best = max(metrics, key=lambda w: metrics[w]["style_gain"])
    if metrics[best]["style_gain"] < SCORECARD_MIN_STYLE_GAIN:
        return "WEAK — barely reads as this artist; retrain (more/cleaner data) or drop it"
    if recommended is None:
        return "OVERCOOKED — style is there but it ignores the prompt / burns in; export an earlier checkpoint"
    return SHIP


def show_scorecard(result: dict) -> None:
    """Display the contact sheet inline in a notebook. The gate must never just say
    "blocked" — the whole point is that you can SEE why (top row = the artist's real
    work). Degrades to a printed path outside Jupyter, so it is safe to call anywhere."""
    print(f"[{result['artist']}] {result['verdict']}  (best strength: {result['recommended_weight']})")
    try:
        from IPython.display import display
    except ImportError:
        return
    display(result["sheet"])


def save_scorecard(result: dict, work_dir: str = "lora_work") -> str:
    out_dir = os.path.join(work_dir, result["artist"], "validation")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{result['artist']}_scorecard.png")
    result["sheet"].save(path)
    print(f"Saved scorecard -> {path}")
    return path


def scorecard_table(results: list[dict]) -> str:
    """All scored LoRAs as one markdown table, ranked by style gain — the ship /
    fix / retrain decision on a single screen, and a drop-in for the docs page."""
    rows = sorted(
        results, key=lambda r: max(m["style_gain"] for m in r["metrics"].values()), reverse=True
    )
    lines = [
        "| LoRA | style gain | prompt drop | burn-in | best strength | verdict |",
        "|---|---|---|---|---|---|",
    ]
    for r in rows:
        w = r["recommended_weight"] or max(r["metrics"], key=lambda k: r["metrics"][k]["style_gain"])
        m = r["metrics"][w]
        lines.append(
            f"| {r['artist']} | {m['style_gain']:+.3f} | {m['prompt_drop']:+.3f} | "
            f"{m['burn_in']:.2f}x | {r['recommended_weight'] or '—'} | {r['verdict'].split(' — ')[0]} |"
        )
    return "\n".join(lines)
