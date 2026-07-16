"""Grade the published LoRAs against the artists' real work — locally, no Colab quota.

For each artist it renders a contact sheet (row 1 = the artist's actual artworks,
below = our generations at each strength) and three numbers, then prints one ranked
table: which LoRAs to ship, which to fix by exporting an earlier checkpoint, and
which are not worth keeping. See training/scorecard.py for what the numbers mean.

    python scripts/score_loras.py                      # every LoRA in the app registry
    python scripts/score_loras.py hokusai turner       # just these
    python scripts/score_loras.py --quick              # one seed instead of two (2x faster)
    python scripts/score_loras.py hokusai --lora lora_work/hokusai/output/checkpoint-900

Reference artworks come from the local training set if it is on this machine, else
they are re-fetched from the Art Institute of Chicago (CC0) — so this also grades a
LoRA whose training data lives on a Drive somewhere.
"""

import argparse
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from training.config import ARTISTS, SCORECARD_SEEDS  # noqa: E402
from training.pipeline import score_artist  # noqa: E402
from training.scorecard import scorecard_table  # noqa: E402

OUTPUT_DIR = os.path.join(REPO_ROOT, "outputs", "scorecards")


def _default_artists() -> list[str]:
    """Whatever the app currently offers — those are the LoRAs that need to be good."""
    sys.path.insert(0, os.path.join(REPO_ROOT, "src"))
    from latent_studio.registry import LORAS

    return [lora.id for lora in LORAS if lora.path]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("artists", nargs="*", help="artist ids (default: the app's registry)")
    parser.add_argument("--lora", help="grade this snapshot dir / repo id instead of the published one")
    parser.add_argument("--work-dir", default="lora_work", help="where training data / outputs live")
    parser.add_argument("--quick", action="store_true", help="one seed per prompt instead of two")
    args = parser.parse_args()

    artists = args.artists or _default_artists()
    unknown = [a for a in artists if a not in ARTISTS]
    if unknown:
        raise SystemExit(f"Unknown artist(s): {unknown}. Known: {sorted(ARTISTS)}")
    if args.lora and len(artists) != 1:
        raise SystemExit("--lora grades one snapshot, so name exactly one artist.")

    seeds = SCORECARD_SEEDS[:1] if args.quick else SCORECARD_SEEDS

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    results = []
    for artist_id in artists:
        try:
            result = score_artist(
                artist_id, lora_source=args.lora, work_dir=args.work_dir, save=False, seeds=seeds
            )
        except Exception as exc:
            # One unloadable LoRA must not throw away the scores already computed —
            # a half-pushed repo is exactly the case this run is here to catch.
            print(f"[{artist_id}] FAILED: {exc}")
            continue
        result["sheet"].save(os.path.join(OUTPUT_DIR, f"{artist_id}_scorecard.png"))
        results.append(result)

    if not results:
        raise SystemExit("Nothing scored.")

    print(f"\nContact sheets -> {OUTPUT_DIR}\n")
    print(scorecard_table(results))
    with open(os.path.join(OUTPUT_DIR, "scorecard.md"), "w") as f:
        f.write(scorecard_table(results) + "\n")


if __name__ == "__main__":
    main()
