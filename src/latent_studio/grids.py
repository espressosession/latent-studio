"""Grid compositing + a generic parameter-sweep engine. This is the engine behind
the app's Compare mode, which is also how the Parameter Atlas is produced —
generate_grid()/sweep() take a plain callable rather than a PipelineManager
directly, so the same engine drives both the plain and the ControlNet path."""

from dataclasses import replace
from typing import Callable

from PIL import Image, ImageDraw, ImageFont

from .generation import GenerationParams

GenerateFn = Callable[[GenerationParams], tuple[Image.Image, dict]]

# Which GenerationParams fields can be swept, and how their values are typed.
# "seed" is handled separately (start + step * i, not an interpolated range) since
# "the range between two seeds" isn't meaningful.
SWEEPABLE_FIELDS: dict[str, type] = {
    "cfg_scale": float,
    "steps": int,
    "lora_weight": float,
    "controlnet_scale": float,
}

# Cell labels are burned into the grid PNG, which ends up in the docs and in the
# live demo — so they carry the names the UI uses, not the field names.
FIELD_LABELS: dict[str, str] = {
    "cfg_scale": "prompt strength",
    "steps": "detail",
    "lora_weight": "style strength",
    "seed": "seed",
    "controlnet_scale": "reference strength",
}


def interpolate_range(value_range: tuple[float, float], count: int, value_type=float) -> list:
    start, end = value_range
    if count <= 1:
        values = [start]
    else:
        step = (end - start) / (count - 1)
        values = [start + step * i for i in range(count)]

    if value_type is int:
        return [max(1, int(round(v))) for v in values]
    return [round(float(v), 2) for v in values]


def make_grid_image(
    images: list[Image.Image | None],
    width: int,
    height: int,
    grid_rows: int,
    grid_cols: int,
    labels: list[str] | None = None,
) -> Image.Image:
    grid_image = Image.new("RGB", (width * grid_cols, height * grid_rows), color=(32, 32, 32))
    draw = ImageDraw.Draw(grid_image)
    font = ImageFont.load_default()

    for idx in range(grid_rows * grid_cols):
        row, col = divmod(idx, grid_cols)
        x, y = col * width, row * height

        image = images[idx] if idx < len(images) else None
        if image is not None:
            grid_image.paste(image.resize((width, height)), (x, y))
        else:
            draw.rectangle([x, y, x + width - 1, y + height - 1], outline=(90, 90, 90), width=2)
            draw.text((x + 12, y + 12), "missing", fill=(220, 220, 220), font=font)

        if labels and idx < len(labels):
            label = labels[idx]
            bbox = draw.textbbox((0, 0), label, font=font)
            label_w, label_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
            padding = 6
            draw.rectangle(
                [x, y + height - label_h - padding * 2, x + label_w + padding * 2, y + height],
                fill=(0, 0, 0),
            )
            draw.text((x + padding, y + height - label_h - padding), label, fill=(255, 255, 255), font=font)

    return grid_image


def generate_grid(
    generate_fn: GenerateFn,
    params_list: list[GenerationParams],
    grid_rows: int,
    grid_cols: int,
    labels: list[str] | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> tuple[Image.Image, list[dict]]:
    """Runs each GenerationParams through generate_fn and composites the
    results. len(params_list) must equal grid_rows * grid_cols. If given,
    `progress(index, total)` is called before each cell — that's what drives the
    app's "Image 3 of 20" bar (progress.ProgressTracker.on_image)."""
    images: list[Image.Image] = []
    metadata_list: list[dict] = []
    total = len(params_list)
    for i, params in enumerate(params_list):
        if progress is not None:
            progress(i, total)
        image, metadata = generate_fn(params)
        images.append(image)
        metadata_list.append(metadata)

    width, height = params_list[0].width, params_list[0].height
    grid_image = make_grid_image(images, width, height, grid_rows, grid_cols, labels)
    return grid_image, metadata_list


def _sweep_values(field_name: str, value_range: tuple[float, float], count: int) -> list:
    if field_name == "seed":
        # For seeds the pair is (start, step), not (from, to) — interpolating
        # between two seeds is meaningless, but "every 10th seed" isn't.
        start, step = int(value_range[0]), int(value_range[1])
        return [start + i * step for i in range(count)]
    return interpolate_range(value_range, count, SWEEPABLE_FIELDS[field_name])


def sweep(
    generate_fn: GenerateFn,
    base_params: GenerationParams,
    dim1: tuple[str, tuple[float, float], int],
    dim2: tuple[str, tuple[float, float], int] | None = None,
    shape: tuple[int, int] | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> tuple[Image.Image, list[dict]]:
    """Generic 1-2 dimension sweep over any field in SWEEPABLE_FIELDS plus "seed".
    Each dim is (field_name, value_range, count) — (from, to) for numeric fields,
    (start, step) for "seed". One dimension -> a row of `count` cells (override
    with `shape=(rows, cols)`); two dimensions -> dim1 across columns, dim2 down
    rows."""
    name1, range1, count1 = dim1
    values1 = _sweep_values(name1, range1, count1)

    if dim2 is None:
        params_list = [replace(base_params, **{name1: v}) for v in values1]
        labels = [f"{FIELD_LABELS[name1]} {v}" for v in values1]
        rows, cols = shape or (1, count1)
        return generate_grid(generate_fn, params_list, rows, cols, labels, progress)

    name2, range2, count2 = dim2
    values2 = _sweep_values(name2, range2, count2)
    params_list = [
        replace(base_params, **{name1: v1, name2: v2}) for v2 in values2 for v1 in values1
    ]
    labels = [
        f"{FIELD_LABELS[name1]} {v1} | {FIELD_LABELS[name2]} {v2}"
        for v2 in values2
        for v1 in values1
    ]
    return generate_grid(generate_fn, params_list, count2, count1, labels, progress)
