"""Local runner for the LoRA pipeline — the offline counterpart to the Colab
notebook, running one artist end-to-end (prepare_data -> train -> validate -> export)
on Apple Silicon (MPS)/CPU. Same code path (auto-clones the matching diffusers
release, drops CUDA-only 8-bit-Adam/fp16 off CUDA). Saves the validation grids to
disk and stages into app_loras/ without a Hub push unless --push (which runs the
scorecard gate first; --force overrides after you've looked). Run from the repo root:

    python -m training.run_local --artist hokusai
    python -m training.run_local --artist turner --push        # also push to HF
    python -m training.run_local --artist hokusai --skip-train  # re-validate/export only
"""

import argparse
import os

from .config import ARTISTS
from .pipeline import export_artist, prepare_data, recommended_dir, train, validate_artist


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--artist", required=True, choices=sorted(ARTISTS), help="which artist LoRA to build")
    parser.add_argument("--work-dir", default="lora_work")
    parser.add_argument("--skip-prepare", action="store_true", help="reuse an existing dataset")
    parser.add_argument("--skip-train", action="store_true", help="reuse an existing training output")
    parser.add_argument("--push", action="store_true", help="also push the exported LoRA to the Hugging Face Hub")
    parser.add_argument("--snapshot-dir", default=None, help="export this snapshot instead of the recommended one")
    parser.add_argument(
        "--force", action="store_true",
        help="push even if the scorecard blocks it (only after looking at the contact sheet)",
    )
    args = parser.parse_args()

    if not args.skip_prepare:
        prepare_data(args.artist, args.work_dir)
    if not args.skip_train:
        train(args.artist, work_dir=args.work_dir)

    result = validate_artist(args.artist, work_dir=args.work_dir)
    out_dir = os.path.join(args.work_dir, args.artist)
    for key, name in [
        ("checkpoint_grid", "validation_checkpoints.png"),
        ("weight_grid", "validation_weights.png"),
        ("generalization_grid", "validation_generalization.png"),
    ]:
        if result.get(key) is not None:
            path = os.path.join(out_dir, name)
            result[key].save(path)
            print(f"saved {path}")
    print(f"Recommended snapshot (confirm by looking at the grids above): {result['recommended']}")

    snapshot = args.snapshot_dir or recommended_dir(result)
    if snapshot is None:
        print("No trained LoRA snapshot found — nothing to export.")
        return
    # --push runs the scorecard gate first (see pipeline.export_artist): nothing gets
    # published that can't show it resembles the artist. Staging locally is ungated.
    export_artist(
        args.artist, snapshot, app_loras_dir="app_loras", push=args.push,
        work_dir=args.work_dir, force=args.force,
    )
    print("Done. Staged into app_loras/ — the app loads that local file in preference to the Hub copy.")


if __name__ == "__main__":
    main()
