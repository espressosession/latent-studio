"""Step 2 of the LoRA pipeline: clean the raw downloads into a 512² training set —
trim frame/mount borders, RGB, center-crop, drop outliers — and report a colour
diagnostic (colourfulness + mono/sepia cast) so a tinted corpus is flagged, not
silently trained on. Writes sequentially numbered PNGs for the caption step."""

import glob
import os

import numpy as np
from PIL import Image

RAW_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff")


def center_crop_square(image: Image.Image) -> Image.Image:
    width, height = image.size
    side = min(width, height)
    left = (width - side) // 2
    top = (height - side) // 2
    return image.crop((left, top, left + side, top + side))


def trim_border(
    image: Image.Image, tol: int = 32, std_max: float = 20.0, light_min: int = 160, max_trim_frac: float = 0.30
) -> Image.Image:
    """Crop a uniform light paper mount / plate margin from the edges. A row/col
    counts as border if it's close to the corner-median background colour and
    near-uniform (robust to foxing specks/caption text); only fires on a light
    border, so a dark painted edge is never mistaken for a mount."""
    rgb = image.convert("RGB")
    arr = np.asarray(rgb).astype(np.float32)
    h, w, _ = arr.shape
    patch = max(2, min(h, w) // 40)
    corners = np.concatenate([
        arr[:patch, :patch].reshape(-1, 3), arr[:patch, -patch:].reshape(-1, 3),
        arr[-patch:, :patch].reshape(-1, 3), arr[-patch:, -patch:].reshape(-1, 3),
    ])
    bg = np.median(corners, axis=0)
    if bg.min() <= light_min:
        return rgb  # dark/coloured edge — a painting, not a paper mount

    row_border = (np.abs(np.median(arr, axis=1) - bg).max(axis=1) <= tol) & (arr.std(axis=1).mean(axis=1) <= std_max)
    col_border = (np.abs(np.median(arr, axis=0) - bg).max(axis=1) <= tol) & (arr.std(axis=0).mean(axis=1) <= std_max)

    def leading(flags: np.ndarray) -> int:
        i = 0
        while i < len(flags) and flags[i]:
            i += 1
        return i

    top = min(leading(row_border), int(h * max_trim_frac))
    bottom = h - min(leading(row_border[::-1]), int(h * max_trim_frac))
    left = min(leading(col_border), int(w * max_trim_frac))
    right = w - min(leading(col_border[::-1]), int(w * max_trim_frac))

    if bottom - top < h * 0.4 or right - left < w * 0.4:
        return rgb  # trimming ate too much — leave it untouched
    return rgb.crop((left, top, right, bottom))


def _opponent(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Red-green and yellow-blue opponent channels (Hasler-Süsstrunk)."""
    r, g, b = (rgb[..., i].astype(np.float32) for i in range(3))
    return r - g, 0.5 * (r + g) - b


def colourfulness(rgb: np.ndarray) -> float:
    """Hasler-Süsstrunk (2003) no-reference colourfulness. ~0-15 grayscale/
    monochrome, ~15-25 muted, 30+ vibrant. `rgb` is HxWx3 uint8."""
    rg, yb = _opponent(rgb)
    return float(np.hypot(rg.std(), yb.std()) + 0.3 * np.hypot(rg.mean(), yb.mean()))


def colour_cast(rgb: np.ndarray) -> tuple[float, float, bool]:
    """Detect a uniform colour cast (e.g. sepia). Returns
    (cast_strength, mono_ratio, warm): `mono_ratio` near 1 means most chroma points
    the same direction (a monochrome tint); `warm` True (red>green, yellowish) is
    the sepia/brown direction. `rgb` is HxWx3 uint8."""
    rg, yb = _opponent(rgb)
    cast = float(np.hypot(rg.mean(), yb.mean()))
    chroma = float(np.hypot(rg, yb).mean())
    mono = cast / (chroma + 1e-6)
    warm = bool(rg.mean() > 0 and yb.mean() > 0)
    return cast, mono, warm


def _report_colour(cf_scores: list[float], cast_scores: list[tuple[float, float, bool]]) -> None:
    cf = float(np.mean(cf_scores))
    cast = float(np.mean([c[0] for c in cast_scores]))
    mono = float(np.mean([c[1] for c in cast_scores]))
    warm_frac = float(np.mean([c[2] for c in cast_scores]))
    print(f"colour: colourfulness={cf:.1f}  cast={cast:.1f}  mono_ratio={mono:.2f}  warm={warm_frac:.0%}")
    # A sepia/tinted corpus reads as HIGH mono_ratio (most chroma one direction),
    # not low colourfulness — the tint itself can be quite saturated.
    if mono > 0.8 or cf < 15:
        if mono > 0.8 and warm_frac > 0.6:
            tone = "sepia/brown"
        elif mono > 0.8:
            tone = "single-hue"
        else:
            tone = "near-grayscale"
        print(
            f"  ! dataset looks monochrome/tinted ({tone}); the LoRA will inherit this cast. "
            "Consider a more colourful artist, or document it as a known limitation."
        )


def preprocess_folder(
    in_dir: str,
    out_dir: str,
    resolution: int = 512,
    min_size: int = 384,
    max_aspect: float = 1.7,
    target: int = 30,
) -> int:
    """Cleans images from `in_dir` into `out_dir`: trim frame borders, drop images
    whose shorter side is below `min_size` or whose long/short ratio exceeds
    `max_aspect`, then center-crop + resize. Stops once `target` images are
    written; prints a colour diagnostic. Returns the count kept."""
    os.makedirs(out_dir, exist_ok=True)
    paths = sorted(
        p for p in glob.glob(os.path.join(in_dir, "*")) if p.lower().endswith(RAW_EXTENSIONS)
    )

    kept = 0
    cf_scores: list[float] = []
    cast_scores: list[tuple[float, float, bool]] = []
    for path in paths:
        if kept >= target:
            break
        try:
            image = Image.open(path)
            image.load()
        except Exception:
            continue
        image = trim_border(image)  # remove frame/mount bands before measuring aspect
        width, height = image.size
        if min(width, height) < min_size:
            continue
        if max(width, height) / min(width, height) > max_aspect:
            continue
        clean = center_crop_square(image).resize((resolution, resolution), Image.LANCZOS)
        clean.save(os.path.join(out_dir, f"{kept:03d}.png"))
        arr = np.asarray(clean)
        cf_scores.append(colourfulness(arr))
        cast_scores.append(colour_cast(arr))
        kept += 1

    print(f"kept {kept}/{len(paths)} images -> {out_dir}")
    if kept:
        _report_colour(cf_scores, cast_scores)
    return kept
