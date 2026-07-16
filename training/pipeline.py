"""Orchestrator for the LoRA automation pipeline: artist id -> downloaded ->
preprocessed -> captioned -> trained -> validated -> exported LoRA. Exposed as
separate steps (not one monolithic run) so the validation gate can be looked
at before exporting. Per-artist work lives under work_dir/<artist_id>/."""

import glob
import os

from .captioning import caption_folder
from .config import (
    ARTISTS,
    BASE_MODEL,
    CHECKPOINT_COUNT,
    DOWNLOAD_WIDTH,
    LEARNING_RATE,
    MAX_ASPECT_RATIO,
    MAX_TRAIN_STEPS,
    MIN_IMAGE_SIZE,
    MIN_TRAIN_STEPS,
    RANK,
    RESOLUTION,
    STEPS_PER_IMAGE,
    TARGET_IMAGE_COUNT,
    TRAIN_SEED,
    VALIDATION_CFG,
    VALIDATION_OFFDOMAIN_PROMPTS,
    VALIDATION_SEEDS,
    VALIDATION_STEPS,
    VALIDATION_WEIGHTS,
)
from .download import download_artist
from .export import export
from .preprocess import preprocess_folder
from .scorecard import save_scorecard, score_lora, scorecard_table, show_scorecard
from .train_lora import run_training
from .validation import render_snapshot_grids, validate


def _paths(artist_id: str, work_dir: str) -> dict[str, str]:
    base = os.path.join(work_dir, artist_id)
    return {
        "raw": os.path.join(base, "raw"),          # CC0 originals (provenance)
        "dataset": os.path.join(base, "dataset"),  # 512² images + metadata.jsonl
        "output": os.path.join(base, "output"),    # final weights + checkpoint-*
        "validation": os.path.join(base, "validation"),  # saved grids (docs)
        "export": os.path.join(base, "export"),    # staged .safetensors for the app
    }


def prepare_data(artist_id: str, work_dir: str = "lora_work") -> str:
    """Steps 1-3: download -> preprocess -> caption. Returns the dataset dir
    (containing the cleaned 512² images + metadata.jsonl)."""
    artist = ARTISTS[artist_id]
    paths = _paths(artist_id, work_dir)
    # download extra candidates since preprocessing drops some
    download_artist(
        artist.search_name, paths["raw"], artist.artist_match, target=TARGET_IMAGE_COUNT * 2,
        image_width=DOWNLOAD_WIDTH,
    )
    kept = preprocess_folder(
        paths["raw"], paths["dataset"], resolution=RESOLUTION,
        min_size=MIN_IMAGE_SIZE, max_aspect=MAX_ASPECT_RATIO, target=TARGET_IMAGE_COUNT,
    )
    if kept < 10:
        raise SystemExit(f"Only {kept} usable images for '{artist_id}' — too few for a stable LoRA.")
    caption_folder(paths["dataset"], artist.trigger_word)
    return paths["dataset"]


def _training_steps(dataset_dir: str) -> tuple[int, int]:
    """Derive training length from the dataset actually being trained on. Steps *per
    image* is the meaningful knob: a fixed total silently overtrains a small set (19
    images x 2400 steps = ~126 passes per image, which memorises those 19 pictures
    instead of abstracting a style) and undertrains a large one. Returns
    (max_train_steps, checkpointing_steps)."""
    count = len(glob.glob(os.path.join(dataset_dir, "*.png")))
    steps = min(MAX_TRAIN_STEPS, max(MIN_TRAIN_STEPS, STEPS_PER_IMAGE * count))
    checkpointing = max(1, steps // CHECKPOINT_COUNT)
    print(
        f"{count} images -> {steps} steps ({steps / max(count, 1):.0f}/image), "
        f"checkpoint every {checkpointing}"
    )
    return steps, checkpointing


def train(
    artist_id: str,
    dataset_dir: str | None = None,
    work_dir: str = "lora_work",
    diffusers_repo: str = "diffusers",
    resume_from_checkpoint: str | None = None,
) -> str:
    """Step 4: LoRA training. Returns the output dir (final weights + checkpoints).
    Pass `resume_from_checkpoint="latest"` to continue an interrupted run instead
    of restarting (only useful when `work_dir` persists, e.g. on Google Drive)."""
    artist = ARTISTS[artist_id]
    paths = _paths(artist_id, work_dir)
    dataset = dataset_dir or paths["dataset"]
    max_train_steps, checkpointing_steps = _training_steps(dataset)
    run_training(
        dataset_dir=dataset,
        output_dir=paths["output"],
        trigger_word=artist.trigger_word,
        model_name=BASE_MODEL,
        diffusers_repo=diffusers_repo,
        max_train_steps=max_train_steps,
        learning_rate=LEARNING_RATE,
        rank=RANK,
        resolution=RESOLUTION,
        seed=TRAIN_SEED,
        checkpointing_steps=checkpointing_steps,
        resume_from_checkpoint=resume_from_checkpoint,
    )
    return paths["output"]


def validate_artist(artist_id: str, output_dir: str | None = None, work_dir: str = "lora_work") -> dict:
    """Step 5: the validation gate. Returns validate()'s result dict — inspect
    its `checkpoint_grid` / `weight_grid` before exporting."""
    artist = ARTISTS[artist_id]
    paths = _paths(artist_id, work_dir)
    return validate(
        output_dir=output_dir or paths["output"],
        base_model=BASE_MODEL,
        trigger_word=artist.trigger_word,
        content_prompt=artist.validation_prompt,
        seeds=VALIDATION_SEEDS,
        cfg=VALIDATION_CFG,
        steps=VALIDATION_STEPS,
        weights=tuple(VALIDATION_WEIGHTS),
        offdomain_prompts=tuple(VALIDATION_OFFDOMAIN_PROMPTS),
    )


def save_validation_grids(artist_id: str, validation_result: dict, work_dir: str = "lora_work") -> list[str]:
    """Persist the three validation grids as PNGs under work_dir/<artist>/validation/,
    named by artist + the snapshot they were built on."""
    out_dir = _paths(artist_id, work_dir)["validation"]
    os.makedirs(out_dir, exist_ok=True)
    snapshot = validation_result.get("sweep_snapshot") or "final"
    targets = [
        ("checkpoint_grid", f"{artist_id}_checkpoint.png"),
        ("weight_grid", f"{artist_id}_weights_{snapshot}.png"),
        ("generalization_grid", f"{artist_id}_generalization_{snapshot}.png"),
    ]
    saved: list[str] = []
    for key, filename in targets:
        grid = validation_result.get(key)
        if grid is not None:
            path = os.path.join(out_dir, filename)
            grid.save(path)
            saved.append(path)
    print(f"Saved {len(saved)} validation grid(s) to {out_dir}")
    return saved


def recommended_dir(validation_result: dict) -> str | None:
    """The snapshot dir for validate()'s recommended label, once you've eyeballed
    the grids and agree with the recommendation."""
    recommended = validation_result.get("recommended")
    for name, path in validation_result.get("lora_dirs", []):
        if name == recommended:
            return path
    return None


def snapshot_dir(validation_result: dict, step) -> str:
    """Resolve a checkpoint by step number (e.g. 1200, or "final") to its snapshot
    dir, from validation_result["lora_dirs"] — for exporting a specific snapshot
    instead of the recommended one."""
    label = "final" if str(step) == "final" else f"checkpoint-{step}"
    dirs = dict(validation_result.get("lora_dirs", []))
    if label not in dirs:
        available = [name for name, _ in validation_result.get("lora_dirs", [])]
        raise ValueError(f"Snapshot '{label}' not available. Choose a step from: {available}")
    return dirs[label]


def compare_snapshot(artist_id: str, validation_result: dict, step, work_dir: str = "lora_work") -> dict:
    """Re-render the weight-sweep + generalisation grids for ONE checkpoint (by
    step number, or "final") — no retraining. Saves the PNGs to
    work_dir/<artist>/validation/ and returns {"weight_grid",
    "generalization_grid", "snapshot"}."""
    artist = ARTISTS[artist_id]
    label = "final" if str(step) == "final" else f"checkpoint-{step}"
    grids = render_snapshot_grids(
        snapshot_dir=snapshot_dir(validation_result, step),
        base_model=BASE_MODEL,
        trigger_word=artist.trigger_word,
        content_prompt=artist.validation_prompt,
        seeds=VALIDATION_SEEDS,
        cfg=VALIDATION_CFG,
        steps=VALIDATION_STEPS,
        weights=tuple(VALIDATION_WEIGHTS),
        offdomain_prompts=tuple(VALIDATION_OFFDOMAIN_PROMPTS),
    )
    out_dir = _paths(artist_id, work_dir)["validation"]
    os.makedirs(out_dir, exist_ok=True)
    for key, filename in [
        ("weight_grid", f"{artist_id}_weights_{label}.png"),
        ("generalization_grid", f"{artist_id}_generalization_{label}.png"),
    ]:
        if grids.get(key) is not None:
            grids[key].save(os.path.join(out_dir, filename))
    grids["snapshot"] = label
    print(f"Rendered + saved grids for {label} -> {out_dir}")
    return grids


def score_artist(
    artist_id: str,
    lora_source: str | None = None,
    work_dir: str = "lora_work",
    save: bool = True,
    seeds: list[int] | None = None,
) -> dict:
    """Step 5b: grade a LoRA against the artist's real artworks. Pass a snapshot
    dir to grade a candidate, or leave `lora_source` None to grade what's already
    published on the Hub. Returns score_lora()'s dict."""
    extra = {"seeds": seeds} if seeds else {}
    result = score_lora(artist_id, lora_source=lora_source, work_dir=work_dir, **extra)
    if save:
        save_scorecard(result, work_dir)
    return result


def save_scorecard_table(results: list[dict], work_dir: str = "lora_work", tag: str = "") -> str:
    """Write the combined, ranked scorecard table + a copy of every contact sheet
    into work_dir/scorecards/. `tag` (e.g. "before"/"after") keeps two runs from
    overwriting each other. Returns the table's path."""
    out_dir = os.path.join(work_dir, "scorecards")
    os.makedirs(out_dir, exist_ok=True)
    suffix = f"_{tag}" if tag else ""
    table_path = os.path.join(out_dir, f"scorecard{suffix}.md")
    with open(table_path, "w", encoding="utf-8") as f:
        f.write(scorecard_table(results) + "\n")
    for r in results:
        r["sheet"].save(os.path.join(out_dir, f"{r['artist']}_scorecard{suffix}.png"))
    print(f"Wrote {table_path} + {len(results)} contact sheet(s) to {out_dir}")
    return table_path


def revalidate_all(
    artist_ids: list[str],
    work_dir: str = "lora_work",
    validate: bool = True,
    grade_published: bool = False,
    tag: str = "",
) -> list[dict]:
    """Re-run validation + scorecard over already-trained artists (reading
    checkpoints from work_dir/<artist>/output) without retraining — the batched,
    single-device way to produce a before/after comparison across the roster
    (MPS and CUDA don't render the same seed identically, so scores from two
    machines aren't comparable). Grades the local snapshot by default;
    `grade_published=True` grades the Hub LoRA instead. `validate=False` skips
    the slower validation grids. One unreadable artist is skipped, not fatal."""
    results: list[dict] = []
    for artist_id in artist_ids:
        try:
            if validate:
                result = validate_artist(artist_id, work_dir=work_dir)
                save_validation_grids(artist_id, result, work_dir)
            source = None if grade_published else _paths(artist_id, work_dir)["output"]
            results.append(score_artist(artist_id, lora_source=source, work_dir=work_dir))
        except Exception as exc:
            print(f"[{artist_id}] SKIPPED — {exc}")
    if results:
        save_scorecard_table(results, work_dir, tag)
    return results


def export_artist(
    artist_id: str,
    snapshot_dir: str,
    app_loras_dir: str = "app_loras",
    push: bool = True,
    work_dir: str = "lora_work",
    gate: bool = True,
    force: bool = False,
    score: dict | None = None,
) -> str:
    """Step 6: stage the confirmed snapshot locally + push to its HF repo.

    The scorecard gates the push — a LoRA that can't show it resembles the artist
    doesn't get published by accident, but is always staged locally first (a
    refused push loses no work). Pass `score` to reuse an existing scorecard
    instead of regenerating it. Escape hatches: `force=True` pushes despite a
    blocking verdict (after you've looked); `gate=False` skips scoring entirely."""
    artist = ARTISTS[artist_id]

    if push and gate:
        score = score or score_artist(artist_id, lora_source=snapshot_dir, work_dir=work_dir)
        show_scorecard(score)  # always look — this is the visual comparison, not a formality
        if not score["passed"] and not force:
            staged = export(
                snapshot_dir=snapshot_dir, artist_id=artist.id, hf_repo=artist.hf_repo,
                artist_label=artist.label, trigger_word=artist.trigger_word,
                app_loras_dir=app_loras_dir, push=False,
            )
            raise SystemExit(
                f"PUSH BLOCKED — the scorecard says: {score['verdict']}\n"
                f"The weights are staged locally at {staged}, so nothing is lost.\n"
                "Compare the contact sheet above (top row = the artist's REAL work) — also saved to\n"
                f"  {os.path.join(work_dir, artist_id, 'validation', artist_id + '_scorecard.png')}\n"
                "  * OVERCOOKED -> export an earlier snapshot instead, no retraining needed:\n"
                "      export_artist(ARTIST, snapshot_dir(result, 900), ...)\n"
                "  * WEAK -> a data problem; retrain with more/cleaner images, or drop the artist.\n"
                "  * You looked and you disagree -> export_artist(..., force=True)."
            )

    return export(
        snapshot_dir=snapshot_dir,
        artist_id=artist.id,
        hf_repo=artist.hf_repo,
        artist_label=artist.label,
        trigger_word=artist.trigger_word,
        app_loras_dir=app_loras_dir,
        push=push,
    )
