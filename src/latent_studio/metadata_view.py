"""Turns a generation result into what the user sees and downloads: the settings
list, the settings JSON, and the PNG + JSON files behind the download buttons."""

import html
import json
import math
import os
import re
import tempfile
import time

import gradio as gr

from .design_tokens import HEIGHT, WIDTH
from .metadata import save_with_metadata
from .registry import DEFAULT_CHECKPOINT_ID, DEFAULT_LORA_ID

# (metadata key, human label) — the order they are shown in.
SETTINGS_ROWS = [
    ("checkpoint_label", "Model"),
    ("lora_label", "Style"),
    ("lora_weight", "Style strength"),
    ("prompt", "Prompt"),
    ("effective_prompt", "Prompt sent"),
    ("negative_prompt", "Avoid"),
    ("seed", "Seed"),
    ("cfg_scale", "Prompt strength"),
    ("steps", "Detail"),
]


def _cells(metadata) -> list[dict]:
    # A comparison grid stores {"cells": [...], "compare": {...}} (see _compare_spec);
    # a bare list is an older grid export with no stored sweep config. A single image
    # stores one plain dict.
    if isinstance(metadata, dict) and "cells" in metadata:
        return metadata["cells"]
    if isinstance(metadata, list):
        return metadata
    return [metadata] if metadata else []


def _compare_spec(metadata) -> dict | None:
    """The exact Compare-tab configuration that produced this grid (field/range/count
    per axis), if this generation came from Compare mode and stored one. None for a
    single image, or for an older bare-list grid export with no stored sweep config —
    either way the caller should reset Compare to off rather than guess at a sweep."""
    if isinstance(metadata, dict) and "compare" in metadata:
        return metadata["compare"]
    return None


def effective_prompt_of(metadata) -> str:
    cells = _cells(metadata)
    if not cells:
        return ""
    first = cells[0]
    return first.get("effective_prompt") or first.get("prompt") or ""


def _mono_box(text: str, center: bool = False) -> str:
    """Same monospace-panel look as settings_html, for a single line of text."""
    align = "text-align:center;" if center else ""
    return (
        f'<div style="font-family:var(--font-mono);font-size:.8rem;line-height:1.6;{align}'
        "background:var(--background-fill-secondary);border-radius:var(--radius-md);"
        f'padding:1rem;overflow-x:auto;word-break:break-word">{html.escape(text)}</div>'
    )


def prompt_used_html(metadata) -> str:
    """Centered like a caption under the gallery image, unlike settings_html's grid."""
    cells = _cells(metadata)
    if not cells:
        return _mono_box("", center=True)
    return _mono_box(effective_prompt_of(metadata), center=True)


def settings_rows(metadata) -> list[tuple[str, str]]:
    cells = _cells(metadata)
    if not cells:
        return []
    first = cells[0]
    rows: list[tuple[str, str]] = []
    for key, label in SETTINGS_ROWS:
        if key not in first:
            continue
        # In a grid, a swept setting differs cell to cell — say so instead of
        # silently showing the first cell's value.
        varies = len({json.dumps(cell.get(key), sort_keys=True) for cell in cells}) > 1
        value = "varies across the grid" if varies else first[key]
        rows.append((label, str(value) if str(value).strip() else "—"))
    rows.append(("Size", f"{first.get('width', WIDTH)} × {first.get('height', HEIGHT)}"))
    if first.get("controlnet_types"):
        rows.append(("Reference", ", ".join(first["controlnet_types"])))
        rows.append(("Reference strength", str(first.get("controlnet_scales", {}))))
    # Device + dtype decide reproducibility — the same seed only matches on the
    # same backend (cuda/fp16 vs mps/fp32 diverge).
    if first.get("device"):
        dtype = first.get("dtype", "")
        rows.append(("Rendered on", f"{first['device']}" + (f" · {dtype}" if dtype else "")))
    return rows


def settings_html(metadata) -> str:
    rows = settings_rows(metadata)
    if not rows:
        return '<p style="opacity:.6">Generate an image to see its settings.</p>'
    cells = "".join(
        f'<span style="opacity:.55;white-space:nowrap">{html.escape(label)}</span>'
        f'<span style="word-break:break-word">{html.escape(value)}</span>'
        for label, value in rows
    )
    return (
        '<div style="font-family:var(--font-mono);font-size:.8rem;line-height:1.6;'
        "background:var(--background-fill-secondary);border-radius:var(--radius-md);"
        'padding:1rem;overflow-x:auto">'
        '<div style="display:grid;grid-template-columns:auto 1fr;gap:.5rem .9rem">'
        f"{cells}</div></div>"
    )


def settings_json(metadata) -> str:
    return json.dumps(metadata or {}, indent=2, ensure_ascii=False)


def metadata_to_control_values(metadata) -> tuple:
    """Metadata dict -> the control values for "reuse these settings" (also used by
    Import). ControlNet collapses back to its single-select value; the seed mode
    switches to Fixed so reusing settings and then rolling Random can't silently change
    the seed.

    Compare's field1/field2/Start/End/Steps are restored too, from the "compare" block
    a grid export now carries (see callbacks.on_generate) — the exact sweep config used
    to produce it, not a reconstruction guessed from the cells' bare values (ambiguous:
    which field was the fast axis vs. the slow one isn't recoverable from values alone).
    A single image, or an older bare-list grid export with no stored sweep config, has
    no `compare` block — Compare resets to off in that case rather than leaving a stale
    sweep silently active (`_compare_spec` returns None for both).

    The trailing pair is a pair of "suppress the next reset" flags: Start/End/Steps
    have their own `field.change()` cascade that re-ranges them to that field's generic
    default the instant `field1`/`field2` changes — which would otherwise immediately
    clobber the exact values just restored here. `True` only when there's a real sweep
    to protect; `False` when resetting to off, where that cascade's normal (disabling)
    behavior is already exactly what's wanted."""
    cells = _cells(metadata)
    first = cells[0] if cells else {}
    types = first.get("controlnet_types", []) or []
    scales = first.get("controlnet_scales", {}) or {}
    cn_select = types[0] if types else "off"

    compare = _compare_spec(metadata)
    if compare:
        field1, field2 = compare.get("field1", "off"), compare.get("field2", "off")

        def _u(value):
            return gr.update(value=value) if value is not None else gr.update()

        from1_u, to1_u, count1_u = _u(compare.get("from1")), _u(compare.get("to1")), _u(compare.get("count1"))
        from2_u, to2_u, count2_u = _u(compare.get("from2")), _u(compare.get("to2")), _u(compare.get("count2"))
        suppress1 = suppress2 = True
    else:
        field1 = field2 = "off"
        from1_u = to1_u = count1_u = from2_u = to2_u = count2_u = gr.update()
        suppress1 = suppress2 = False

    return (
        first.get("checkpoint_id", DEFAULT_CHECKPOINT_ID),
        first.get("lora_id", DEFAULT_LORA_ID),
        first.get("lora_weight", 1.0),
        first.get("prompt", ""),
        first.get("negative_prompt", ""),
        first.get("cfg_scale", 7.5),
        first.get("steps", 30),
        "fixed",
        first.get("seed", 0),
        cn_select,
        scales.get(cn_select, 1.0) if cn_select != "off" else 1.0,
        field1, from1_u, to1_u, count1_u,
        field2, from2_u, to2_u, count2_u,
        suppress1, suppress2,
    )


def _slug(text: str, limit: int = 40) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", (text or "").lower()).strip("-")
    return slug[:limit].strip("-")


def _prepare_download_files(entry: dict) -> tuple[str, str]:
    """Builds (and caches on the entry) the PNG + settings-JSON files a download
    button hands out — cached so revisiting a gallery image doesn't rewrite files."""
    if entry.get("files"):
        return entry["files"]
    metadata = entry["metadata"]
    cells = _cells(metadata)
    stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime(entry.get("created_at", time.time())))
    slug = _slug(cells[0].get("prompt", "") if cells else "")
    base = f"latent-studio_{stamp}_{slug}" if slug else f"latent-studio_{stamp}"

    folder = tempfile.mkdtemp(prefix="latent-studio-")
    png_path = os.path.join(folder, f"{base}.png")
    save_with_metadata(entry["image"], metadata or {}, png_path)
    json_path = os.path.join(folder, f"{base}_settings.json")
    with open(json_path, "w", encoding="utf-8") as handle:
        handle.write(settings_json(metadata))

    entry["files"] = (png_path, json_path)
    return entry["files"]


def history_to_gallery(history: list[dict]) -> list:
    # No captions — the settings live under the image and in the Advanced column.
    return [entry["image"] for entry in history]


def square_shape(count: int) -> tuple[int, int]:
    cols = math.ceil(math.sqrt(count))
    rows = math.ceil(count / cols)
    return rows, cols
