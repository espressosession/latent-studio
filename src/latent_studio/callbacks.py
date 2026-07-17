"""Event handlers and the app's interactive logic: which controls grey each other
out, and what happens on every button click / control change. Holds the app's two
pipeline singletons (single-user/local, so module-global is fine)."""

import json
import os
import random
import threading
import time

import gradio as gr

from .controlnet import ControlNetManager, canny_preprocess, depth_preprocess
from .design_tokens import (
    ASPECT_SIZES,
    MAX_IMAGES,
    MAX_SEED,
    MAX_UPSCALE_HOPS,
    STATUS_REFERENCE_OFF,
    STATUS_SEED_RANDOM,
    STATUS_STYLE_OFF,
    STATUS_SWEPT,
    SWEEP_SPECS,
)
from .generation import GenerationParams, generate
from .grids import sweep
from .metadata import read_metadata
from .metadata_view import (
    _cells,
    _prepare_download_files,
    effective_prompt_of,
    history_to_gallery,
    import_warnings,
    is_grid,
    metadata_to_control_values,
    prompt_used_html,
    settings_html,
    square_shape,
    upscale_hops,
)
from .pipeline_manager import PipelineManager, preload_models
from .progress import ProgressTracker, status_html, track_tqdm
from .registry import get_checkpoint, get_lora
from .upscaler import UPSCALER_REPO, UpscalerManager

manager = PipelineManager()
controlnet_manager = ControlNetManager()
upscaler_manager = UpscalerManager()

# checkpoint/lora/aspect/weight/prompt/negative/cfg/steps/seed_mode/seed/controlnet/
# controlnet_scale/field1/from1/to1/count1/field2/from2/to2/count2/suppress1/suppress2
APPLY_OUTPUT_COUNT = 22


def model_status_html() -> str:
    if manager.is_ready:
        return status_html("check", f"Loaded: {get_checkpoint(manager.checkpoint_id).label}")
    if controlnet_manager.is_ready:
        return status_html("check", f"Loaded: {get_checkpoint(controlnet_manager.checkpoint_id).label} + reference")
    if upscaler_manager.is_ready:
        return status_html("check", "Loaded: Upscaler (2x)")
    return status_html("alert", "No model loaded yet — the first Generate loads one.")


def _image_suffix(count: int) -> str:
    return "image" if count <= 1 else f"{count} images"


def generate_button_label(
    checkpoint_id: str, lora_id: str, count: int = 1, controlnet_active: bool = False
) -> str:
    """The button's resting label: what a click will actually do, and how many images
    it will produce — folds in what used to be a separate "N images" note.
    controlnet_active picks which of the two pipeline singletons is actually going to
    serve the next click — a ControlNet generation loads/reads controlnet_manager, not
    manager (which work() unloads whenever ControlNet is on), so checking the wrong one
    here always reported "Load model & generate" even with a warm ControlNet pipeline."""
    suffix = _image_suffix(count)
    active = controlnet_manager if controlnet_active else manager
    if not active.is_ready or active.checkpoint_id != checkpoint_id:
        return f"Load model & generate {suffix}"
    if active.lora_id != lora_id:
        return f"Load style & generate {suffix}"
    return f"Generate {suffix}"


def _tracker_status(tracker: ProgressTracker):
    """Adapter: turns a manager's free-text on_status message into a tracker phase +
    coarse button state. "Unloading…" is the only sub-phase worth its own button text;
    everything else during a load reads as "Loading model…" on the button."""
    def on_status(message: str) -> None:
        button = "Unloading model…" if message.lower().startswith("unloading") else "Loading model…"
        tracker.set_phase(message, button)
    return on_status


# -- gating: which controls grey out, and why (shown in their info=) ----------

def compared_fields(field1: str, field2: str) -> set[str]:
    if field1 not in SWEEP_SPECS:  # field1 == "off" means Compare is off entirely
        return set()
    return {field for field in (field1, field2) if field in SWEEP_SPECS}


def gating_updates(field1, field2, lora_id, seed_mode, controlnet_select):
    """Updates for (cfg, steps, style strength, seed mode, seed, randomize, reference
    strength) — each control's info= carries only its live STATUS reason. Sweeping the
    seed also forces Fixed mode in the same update: a random base would silently change
    what a seed comparison even means."""
    compared = compared_fields(field1, field2)

    style_swept = "lora_weight" in compared
    style_off = lora_id == "none"
    style_info = STATUS_SWEPT if style_swept else (STATUS_STYLE_OFF if style_off else "")

    seed_swept = "seed" in compared
    seed_mode_now = "fixed" if seed_swept else seed_mode
    seed_info = STATUS_SWEPT if seed_swept else (STATUS_SEED_RANDOM if seed_mode_now == "random" else "")
    seed_mode_update = (
        gr.update(interactive=False, value="fixed") if seed_swept else gr.update(interactive=True)
    )

    return (
        gr.update(interactive="cfg_scale" not in compared, info=STATUS_SWEPT if "cfg_scale" in compared else ""),
        gr.update(interactive="steps" not in compared, info=STATUS_SWEPT if "steps" in compared else ""),
        gr.update(interactive=not style_swept and not style_off, info=style_info),
        seed_mode_update,
        gr.update(interactive=not seed_swept and seed_mode_now == "fixed", info=seed_info),
        gr.update(interactive=not seed_swept and seed_mode_now == "fixed"),
        gr.update(interactive=controlnet_select != "off", info="" if controlnet_select != "off" else STATUS_REFERENCE_OFF),
    )


def dim_updates(master_field: str, field: str, reset: bool, gate_field: bool = False, suppress: bool = False):
    """Updates for one Compare column: (field radio, start, end, steps, suppress-flag
    cleared back to False). The first column's radio is the on/off switch itself, so it
    always stays clickable (gate_field=False, the default) — but the second column only
    makes sense once the first is active, so its own radio greys out too when
    gate_field=True. Start/End grey out when off or when the field is seed (seed has no
    numeric range, only a count); `reset` re-ranges Start/End/Steps to the new field's
    default (range and count both).

    `suppress=True` means a restore (Reuse/Import) just set this column's exact values
    — Start/End/Steps alike — as part of the very same event that changed `field`; this
    cascade would otherwise fire right after and immediately overwrite them with the
    field's generic default. Skip the *value* reset for that one cycle and clear the flag
    so the next real field change resets as usual — but still realign every control's
    minimum/maximum(/step) to the restored field's own spec (never suppressed): each of
    Start/End/Steps carries its range over from whatever field was active before, and the
    four fields' ranges genuinely don't nest (lora_weight's floor of 0.0 sits below every
    other field's minimum; cfg_scale/steps/lora_weight cap Steps at 8 images where seed
    goes to 25). A restored value landing outside the previous field's leftover range
    crashes Gradio's own bounds check in preprocess() before on_generate ever runs — only
    realigning the range here (not widening it to some shared safe superset) fixes that
    without losing what each field's own range means, both what the Start/End sliders
    visibly drag across and Steps' per-field image-count ceiling."""
    compare_on = master_field in SWEEP_SPECS
    spec = SWEEP_SPECS.get(field)
    usable = compare_on and spec is not None
    is_seed = field == "seed"
    numeric_on = usable and not is_seed

    if suppress:
        field_update = gr.update(interactive=compare_on) if gate_field else gr.update()
        if usable:
            from_update = gr.update(
                interactive=numeric_on, minimum=spec.minimum, maximum=spec.maximum, step=spec.step
            )
            to_update = gr.update(
                interactive=numeric_on, minimum=spec.minimum, maximum=spec.maximum, step=spec.step
            )
            count_update = gr.update(interactive=usable, maximum=spec.max_count)
        else:
            from_update = gr.update(interactive=numeric_on)
            to_update = gr.update(interactive=numeric_on)
            count_update = gr.update(interactive=usable)
        return (field_update, from_update, to_update, count_update, False)

    if numeric_on and reset:
        from_update = gr.update(
            interactive=True, minimum=spec.minimum, maximum=spec.maximum, step=spec.step, value=spec.start
        )
        to_update = gr.update(
            interactive=True, minimum=spec.minimum, maximum=spec.maximum, step=spec.step, value=spec.end
        )
    else:
        from_update = gr.update(interactive=numeric_on)
        to_update = gr.update(interactive=numeric_on)

    count_update = (
        gr.update(interactive=True, maximum=spec.max_count, value=spec.count)
        if usable and reset
        else gr.update(interactive=usable)
    )
    field_update = gr.update(interactive=compare_on) if gate_field else gr.update()
    return field_update, from_update, to_update, count_update, False


def image_count(field1, count1, field2, count2) -> int:
    if field1 not in SWEEP_SPECS:
        return 1
    total = int(count1)
    if field2 in SWEEP_SPECS:
        total *= int(count2)
    return total


# -- event handlers -------------------------------------------------------------

def on_button_label_change(checkpoint_id, lora_id, field1, count1, field2, count2):
    count = image_count(field1, count1, field2, count2)
    return gr.update(value=generate_button_label(checkpoint_id, lora_id, count))


def on_randomize_seed():
    return random.randint(0, MAX_SEED)


def on_advanced_toggle(enabled: bool):
    return gr.update(visible=enabled)


def on_controlnet_toggle(enabled: bool):
    """Reference image is Advanced-only — it rides on the same toggle as the Settings
    panel and shows/hides alongside it. Turning Advanced off also resets the
    reference-type radio to "off" rather than leaving it selected underneath —
    turning Advanced back on later should mean picking a reference type again, not
    silently reactivating whatever was chosen before it was hidden."""
    return gr.update(visible=enabled), gr.update() if enabled else "off"


def on_controlnet_change(controlnet_select: str):
    # The upload + Process button grey out instead of the row disappearing while no
    # reference type is picked — matches the app's never-hide-a-control rule. Also
    # clears the stale preview so switching outlines<->depth re-runs the preprocessor.
    active = controlnet_select != "off"
    return gr.update(interactive=active), gr.update(interactive=active), None


def on_preprocess(image, controlnet_select: str):
    if image is None:
        raise gr.Error("Upload a reference image first.")
    if controlnet_select == "canny":
        return canny_preprocess(image)
    if controlnet_select == "depth":
        return depth_preprocess(image)
    raise gr.Error("Pick a reference type first.")


def on_preload():
    """Warms the disk cache with every checkpoint + LoRA (no VRAM used), on the same
    worker-thread + ProgressTracker machinery as Generate, so a live demo's first
    model switch isn't a download wait."""
    tracker = ProgressTracker()
    outcome: dict = {}

    def work():
        try:
            with track_tqdm(tracker):
                preload_models(lambda msg: tracker.set_phase(msg, "Preloading…"))
        except Exception as exc:  # noqa: BLE001 — surfaced in the panel, never a popup
            outcome["error"] = exc

    worker = threading.Thread(target=work, daemon=True)
    worker.start()
    while worker.is_alive():
        yield gr.update(interactive=False, value="Preloading…"), tracker.html()
        time.sleep(0.2)
    worker.join()

    if "error" in outcome:
        yield gr.update(interactive=True, value="Preload all models"), status_html("alert", str(outcome["error"]))
        return
    yield (
        gr.update(interactive=True, value="Preload all models"),
        status_html("check", "All models cached — the first switch will be instant."),
    )


def on_import_settings(file):
    """Import: apply a settings JSON, or a PNG's embedded metadata, to the live
    controls — exactly like "Reuse these settings", but from a file, so it works
    across sessions too. Errors are plain language in the state panel, never a popup.

    A generator, not a plain return: a large grid export's embedded metadata is a
    real parse (and the upload itself can take a moment), and the fields it's about
    to overwrite — checkpoint, style, Compare's own sweep config — are exactly what
    Generate reads. The interim "Importing…" yield disables Generate for that window
    so a click mid-import can't race the restore and fire on a half-applied state."""
    noop = tuple(gr.update() for _ in range(APPLY_OUTPUT_COUNT))
    if file is None:
        yield (*noop, status_html("alert", "No file selected."), gr.update())
        return

    yield (*noop, status_html("upload", "Importing settings…"), gr.update(interactive=False, value="Importing…"))

    path = file if isinstance(file, str) else file.name
    ext = os.path.splitext(path)[1].lower()
    try:
        if ext == ".json":
            with open(path, encoding="utf-8") as handle:
                metadata = json.load(handle)
        else:
            metadata = read_metadata(path)
            if metadata is None:
                yield (*noop, status_html(
                    "alert",
                    "This image has no Latent Studio metadata embedded — was it downloaded from this app?",
                ), gr.update(interactive=True))
                return
    except json.JSONDecodeError as exc:
        yield (*noop, status_html("alert", f"Not a valid settings file: {exc}"), gr.update(interactive=True))
        return
    except Exception as exc:  # noqa: BLE001 — surfaced in the panel, never a popup
        yield (*noop, status_html("alert", f"Couldn't read that file: {exc}"), gr.update(interactive=True))
        return

    values = metadata_to_control_values(metadata)
    checkpoint_id, lora_id, field1, field2 = values[0], values[1], values[12], values[16]
    count1, count2 = values[15].get("value", 1), values[19].get("value", 1)
    total = image_count(field1, count1, field2, count2)

    warnings = import_warnings(metadata)
    message = (
        f"Settings imported, but {' and '.join(warnings)} no longer exist — defaulted instead."
        if warnings
        else "Settings imported — press Generate to reproduce it."
    )
    yield (
        *values,
        status_html("alert" if warnings else "check", message),
        gr.update(interactive=True, value=generate_button_label(checkpoint_id, lora_id, total)),
    )


def on_generate(
    checkpoint_id,
    lora_id,
    aspect_id,
    prompt,
    negative_prompt,
    cfg_scale,
    steps,
    seed_mode,
    seed,
    lora_weight,
    advanced_view_on,
    controlnet_select,
    controlnet_preview,
    controlnet_scale,
    field1, from1, to1, count1,
    field2, from2, to2, count2,
    history,
):
    """The one dispatch point for both modes (single image / comparison grid) and
    both pipelines (plain / ControlNet). Runs the diffusers call on a worker thread
    and polls a ProgressTracker so the state panel can show a real bar + ETA."""
    compare_enabled = field1 in SWEEP_SPECS
    total_images = image_count(field1, count1, field2, count2)

    def label():
        return generate_button_label(checkpoint_id, lora_id, total_images, controlnet_active=bool(controlnet_types))

    def frozen(state_html):
        return (
            gr.update(), history, gr.update(), gr.update(), gr.update(), gr.update(), gr.update(),
            gr.update(interactive=True, value=label()),
            model_status_html(), state_html, gr.update(), gr.update(), gr.update(),
        )

    def failed(message):
        return frozen(status_html("alert", message))

    # ControlNet is experimental and Advanced-only — only build a real controlnet_types
    # list when Advanced view is actually on, regardless of what's selected underneath
    # (matches on_controlnet_toggle resetting controlnet_select to "off" whenever
    # Advanced itself goes off).
    controlnet_types = [controlnet_select] if advanced_view_on and controlnet_select != "off" else []
    control_images: dict = {}
    controlnet_scales: dict = {}
    if controlnet_types:
        if controlnet_preview is None:
            yield failed("Process the reference image first — upload one and press Process.")
            return
        control_images = {controlnet_select: controlnet_preview}
        controlnet_scales = {controlnet_select: controlnet_scale}

    if compare_enabled:
        if field2 in SWEEP_SPECS and field1 == field2:
            yield failed("Pick two different settings to compare.")
            return
        if total_images > MAX_IMAGES:
            yield failed(f"That's {total_images} images. Keep a comparison under {MAX_IMAGES}.")
            return

    # Random mode rolls the seed once, here, and reports it back into the (still
    # greyed) seed box — so a lucky result can be pinned by switching to Fixed.
    used_seed = random.randint(0, MAX_SEED) if seed_mode == "random" else int(seed)

    def dim(field, from_value, to_value, count):
        if field == "seed":
            # Compare-mode seeds always count up by 1 from the Tuning tab's own seed —
            # gating_updates() already forces Fixed mode whenever a seed sweep is picked.
            return ("seed", (used_seed, 1), int(count))
        return (field, (float(from_value), float(to_value)), int(count))

    width, height = ASPECT_SIZES[aspect_id]
    base_params = GenerationParams(
        prompt=prompt.strip(),
        negative_prompt=negative_prompt.strip(),
        width=width,
        height=height,
        cfg_scale=cfg_scale,
        steps=int(steps),
        seed=used_seed,
        lora_weight=lora_weight,
        controlnet_types=controlnet_types,
        controlnet_scales=controlnet_scales,
    )

    tracker = ProgressTracker()
    outcome: dict = {}

    def work():
        try:
            with track_tqdm(tracker):
                if controlnet_types:
                    if manager.is_ready:
                        manager.unload()  # only one full pipeline stays resident
                    stale = not (
                        controlnet_manager.is_ready
                        and controlnet_manager.checkpoint_id == checkpoint_id
                        and controlnet_manager.active_types == controlnet_types
                    )
                    if stale:
                        tracker.set_phase("Loading the reference model…", "Loading model…")
                        controlnet_manager.load(checkpoint_id, controlnet_types)
                    if controlnet_manager.lora_id != lora_id:
                        tracker.set_phase(f"Loading the {get_lora(lora_id).label} style…", "Loading model…")
                        controlnet_manager.set_lora(lora_id)
                else:
                    if controlnet_manager.is_ready:
                        controlnet_manager.unload()
                    if not manager.is_ready or manager.checkpoint_id != checkpoint_id:
                        checkpoint_label = get_checkpoint(checkpoint_id).label
                        tracker.set_phase(
                            f"Loading {checkpoint_label} — downloaded once, then cached", "Loading model…"
                        )
                        manager.load_checkpoint(checkpoint_id, on_status=_tracker_status(tracker))
                    if manager.lora_id != lora_id:
                        tracker.set_phase(f"Loading the {get_lora(lora_id).label} style…", "Loading model…")
                        manager.set_lora(lora_id, on_status=_tracker_status(tracker))

                tracker.set_phase("Generating…", "Generating…", image_total=total_images)

                def generate_fn(params):
                    return generate(manager, controlnet_manager, params, control_images)

                if not compare_enabled:
                    image, metadata = generate_fn(base_params)
                else:
                    dim1 = dim(field1, from1, to1, count1)
                    dim2 = dim(field2, from2, to2, count2) if field2 in SWEEP_SPECS else None
                    # Seeds have no order, so a row says less than a block; a swept
                    # number does have an order and reads best left to right.
                    shape = square_shape(int(count1)) if dim2 is None and field1 == "seed" else None
                    image, metadata = sweep(
                        generate_fn, base_params, dim1, dim2, shape=shape, progress=tracker.on_image
                    )
                    # Store the exact sweep config alongside the cells — Import/Reuse
                    # restore it verbatim (metadata_view.metadata_to_control_values),
                    # rather than trying to reverse-engineer which field was which axis
                    # from the cells' bare values (ambiguous, and doesn't work at all
                    # for "Reuse" on a grid either).
                    metadata = {
                        "cells": metadata,
                        "compare": {
                            "field1": field1, "from1": float(from1), "to1": float(to1), "count1": int(count1),
                            "field2": field2 if field2 in SWEEP_SPECS else "off",
                            "from2": float(from2) if field2 in SWEEP_SPECS else None,
                            "to2": float(to2) if field2 in SWEEP_SPECS else None,
                            "count2": int(count2) if field2 in SWEEP_SPECS else None,
                        },
                    }
                outcome["image"], outcome["metadata"] = image, metadata
        except Exception as exc:  # noqa: BLE001 — every failure belongs in the panel
            outcome["error"] = exc

    worker = threading.Thread(target=work, daemon=True)
    worker.start()
    while worker.is_alive():
        yield (
            gr.update(), history, gr.update(), gr.update(), gr.update(), gr.update(), gr.update(),
            gr.update(interactive=False, value=tracker.button),
            model_status_html(), tracker.html(), gr.update(), gr.update(), gr.update(),
        )
        time.sleep(0.2)
    worker.join()

    if "error" in outcome:
        yield failed(str(outcome["error"]))
        return

    image, metadata = outcome["image"], outcome["metadata"]
    entry = {"image": image, "metadata": metadata, "created_at": time.time()}
    history = [entry] + history
    png_path, json_path = _prepare_download_files(entry)

    cells = _cells(metadata)
    done = f"Done — {len(cells)} images" if len(cells) > 1 else f"Done — seed {cells[0].get('seed', '?')}"

    yield (
        metadata, history,
        gr.update(value=history_to_gallery(history), selected_index=0), 0,
        gr.update(visible=True, interactive=not is_grid(metadata) and upscale_hops(metadata) < MAX_UPSCALE_HOPS),
        gr.update(value=png_path, visible=True), json_path,
        gr.update(interactive=True, value=label()),
        model_status_html(), status_html("check", done),
        prompt_used_html(metadata), settings_html(metadata),
        gr.update(value=used_seed),
    )


def on_upscale(history, selected_index):
    """Runs the currently selected single image through the 2x latent upscaler
    (UpscalerManager) and pushes the result as a new history entry — the original
    stays untouched. is_grid() is the same guard the button's own `interactive`
    wiring uses; checked again here since a click shouldn't be trusted to always
    arrive after the UI has caught up (see on_generate's identical shape)."""

    def frozen(state_html):
        return (
            gr.update(), history, gr.update(), gr.update(),
            gr.update(interactive=True),
            gr.update(), gr.update(),
            model_status_html(), state_html, gr.update(), gr.update(),
        )

    def failed(message):
        return frozen(status_html("alert", message))

    if not history:
        yield failed("Generate an image first.")
        return

    entry = history[selected_index]
    metadata = entry["metadata"]
    if is_grid(metadata):
        yield failed("Upscale works on a single image, not a comparison grid.")
        return
    hops = upscale_hops(metadata)
    if hops >= MAX_UPSCALE_HOPS:
        # Each hop doubles both dimensions with no natural ceiling — chaining past
        # MAX_UPSCALE_HOPS is a real, reachable CUDA OOM, not just a theoretical one.
        times = "once" if hops == 1 else f"{hops} times"
        yield failed(
            f"This image has already been upscaled {times} — further upscaling risks "
            "running out of GPU memory. Pick the original generation instead."
        )
        return

    cell = _cells(metadata)[0]
    prompt = effective_prompt_of(metadata)
    seed = cell.get("seed", 0)
    image = entry["image"]

    tracker = ProgressTracker()
    outcome: dict = {}

    def work():
        try:
            with track_tqdm(tracker):
                # Only one full pipeline stays resident — same VRAM discipline as the
                # ControlNet/plain swap in on_generate's own work().
                if manager.is_ready:
                    manager.unload()
                if controlnet_manager.is_ready:
                    controlnet_manager.unload()
                if not upscaler_manager.is_ready:
                    tracker.set_phase("Loading the upscaler…", "Loading model…")
                    upscaler_manager.load(on_status=_tracker_status(tracker))
                tracker.set_phase("Upscaling…", "Upscaling…")
                outcome["image"] = upscaler_manager.upscale(image, prompt, seed)
        except Exception as exc:  # noqa: BLE001 — every failure belongs in the panel
            outcome["error"] = exc

    worker = threading.Thread(target=work, daemon=True)
    worker.start()
    while worker.is_alive():
        yield (
            gr.update(), history, gr.update(), gr.update(),
            gr.update(interactive=False, value=tracker.button),
            gr.update(), gr.update(),
            model_status_html(), tracker.html(), gr.update(), gr.update(),
        )
        time.sleep(0.2)
    worker.join()

    if "error" in outcome:
        yield failed(str(outcome["error"]))
        return

    upscaled_image = outcome["image"]
    # Every original key (prompt, seed, checkpoint, ...) stays untouched, so "Reuse
    # these settings" on this entry still reproduces the base generation, not a
    # regeneration at the doubled size — the nested block mirrors how a Compare grid
    # carries its own "compare" block alongside "cells".
    new_metadata = dict(cell)
    new_hop = hops + 1
    new_metadata["upscale"] = {
        "model": UPSCALER_REPO, "scale": 2 ** new_hop, "hop": new_hop,
        "width": upscaled_image.width, "height": upscaled_image.height,
    }
    new_entry = {"image": upscaled_image, "metadata": new_metadata, "created_at": time.time()}
    history = [new_entry] + history
    png_path, json_path = _prepare_download_files(new_entry)

    yield (
        new_metadata, history,
        gr.update(value=history_to_gallery(history), selected_index=0), 0,
        gr.update(interactive=True),
        gr.update(value=png_path, visible=True), json_path,
        model_status_html(), status_html("check", "Done — upscaled 2x"),
        prompt_used_html(new_metadata), settings_html(new_metadata),
    )


def on_gallery_select(evt: gr.SelectData, history):
    # Shows the selected thumbnail's prompt/settings and re-points the download
    # buttons — deliberately doesn't touch the live controls ("Reuse" does that).
    entry = history[evt.index]
    png_path, json_path = _prepare_download_files(entry)
    metadata = entry["metadata"]
    return (
        metadata, evt.index,
        gr.update(interactive=not is_grid(metadata) and upscale_hops(metadata) < MAX_UPSCALE_HOPS),
        gr.update(value=png_path, visible=True), json_path,
        prompt_used_html(metadata), settings_html(metadata),
    )
