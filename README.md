# Latent Studio

SD 1.5 image generation web app (Gradio) — Creative Coding Advanced capstone, TH Nürnberg.

A prompt box, a style, and a few honest controls: pick one of **3 SD 1.5 checkpoints** and one of
**6 self-trained artist styles**, and every image comes back with the full settings that made it —
embedded in the PNG, and reusable with one click. The styles are not downloaded from a model hub:
they are LoRAs trained here, by an automated pipeline that turns *an artist's name* into a validated,
scored, published adapter.

> **Scope.** The brief's optional *advanced* capability is ControlNet. It was built here (canny/depth,
> multi-ControlNet, combinable with a style LoRA) and is functional in local testing (Apple MPS: loading,
> single images and grids all generate). On the Colab demo runtime, though, it raises errors that don't
> reproduce cleanly — different runs fail differently and none of them surface locally under the same
> library versions — so it is **disabled in the shipped app**; the code stays in the tree. The advanced
> work in this project is the automated, measured LoRA pipeline below.

**Write-up (live):** https://espressosession.github.io/latent-studio/

## Quickstart — local (MacBook, MPS)

```bash
python3.13 -m venv .venv
source .venv/bin/activate
pip install -e .
python -m latent_studio.app
```

Opens the Gradio app on `http://127.0.0.1:7860`. Generation runs on MPS for real (slower than a T4,
but no stubbing needed for local development). LoRA weights are pulled from the Hugging Face Hub on
first use.

## Quickstart — Colab (GPU)

Open [notebooks/latent_studio_colab.ipynb](notebooks/latent_studio_colab.ipynb) in Colab, switch to a
GPU (T4) runtime, and run all cells top-to-bottom. A preflight cell fetches every model up front and
retries past the Hub's CDN, so a flaky download can't kill the run halfway in.

Both notebooks are **generated** from the source — never edit them by hand, regenerate instead:

```bash
python scripts/build_colab_notebook.py    # rebuilds BOTH notebooks (app + LoRA training)
```

## The styles: an automated LoRA pipeline

The style LoRAs are not curated by hand. The pipeline takes an **artist's name** and produces a
trained, validated, scored `.safetensors` on the Hub:

> artist name → Art Institute of Chicago open-access download (CC0, public-domain verified per object)
> → preprocess → BLIP caption → train → validate → **score** → push

```bash
pip install -e ".[train]"
python -m training.run_local --artist hokusai   # full pipeline, local MPS/CPU (--push to publish)
python scripts/score_loras.py                   # score the shipped LoRAs against the artists' real work
```

Colab (faster, and what the submission used): open
[notebooks/lora_training_colab.ipynb](notebooks/lora_training_colab.ipynb) on a GPU runtime.

**Step 5b — the scorecard — is the point.** Ordinary LoRA validation only compares the adapter against
the *base model*; it never looks at the artist's actual work, which leaves "does this look like
Hokusai?" a matter of opinion. The scorecard makes it a measurement: every number is a delta against
the same prompt+seed rendered *without* the LoRA, and style is measured only on deliberately **modern**
prompts (a bicycle, traffic lights) that no pre-1900 corpus can contain — so a gain can only be style,
never a memorised training motif. It gates the Hub push, and it prints a contact sheet with the real
artworks in the top row, because the numbers rank but the eye decides.

Currently shipped: **Hokusai, Hiroshige, Turner, Monet, Dürer, Rembrandt** — six styles, all trained
on CC0 public-domain works and scored on CUDA. Every style was retrained under the corrected pipeline
and compared before/after: dataset size turned out to be the ceiling, so three thinner styles
(Cézanne, Cassatt, Van Gogh) are kept as **documented limitations** rather than shipped — and the eye,
not the metric, drew that line (CLIP scores Cézanne ≈ Hokusai, yet only Hokusai convinces). The full
before/after study, with side-by-side contact sheets, is in the
[write-up](https://espressosession.github.io/latent-studio/); scorecards land in `outputs/scorecards/`
(gitignored).

## Parameter atlas

The atlas is generated **in the app itself**, with its Compare mode — there is no separate script: the
parameter study and the product run on the same engine (`grids.sweep()`), so every cell is reproducible
by typing its settings back into the interface. Each grid downloads as a PNG plus a JSON holding every
cell's full settings.

The submitted grids — CFG×steps, coarse and fine style-strength sweeps, a 5-seed × 6-style-strength
(0→1.25) sweep per checkpoint (pinning exactly where a photoreal checkpoint's resistance to a style
breaks down, which a coarser sweep can only bracket), and a whole-roster figure (all nine styles on one
prompt and the same six seeds) — live in [`docs/`](docs/) (images in `docs/images/atlas/`, per-cell
settings in `docs/atlas/`), full resolution in `outputs/atlas/` (gitignored). Everything is rendered on
CUDA (the demo device), and the observations are in the "Parameter atlas" section of the
[write-up](https://espressosession.github.io/latent-studio/).

## Layout

- `src/latent_studio/` — the app package (source of truth)
- `training/` — the LoRA automation pipeline (`run_local.py` for local runs)
- `scripts/` — LoRA scoring + the notebook build script
- `notebooks/` — generated Colab deliverables (app + training)
- `docs/` — GitHub Pages write-up (public — keep curated)
- `outputs/` — generated grids + scorecards (gitignored; the submitted subset is copied into `docs/`)
- `reference/` — old course notebooks, local only (gitignored, not part of the repo)
- `local-data/` — local-only raw data (gitignored, multi-GB): the before/after retrain snapshots
  (`verification/before`, `verification/after`), the raw atlas/roster grids that fed `docs/`
  (`documentation-grids/`), and the raw example-generation renders (`documentation-examples/`). The
  shipped LoRAs live on the HF Hub and reproduce via the training notebook; the committed verification
  evidence is the before/after contact sheets in `docs/images/comparison/`.

## Licences

SD 1.5 and both community checkpoints are CreativeML OpenRAIL-M; the LoRAs are trained on CC0
public-domain works from the Art Institute of Chicago and released under the same licence. No LoRA is
trained on the likeness of a living person, and the safety checker stays enabled in the public-facing
app. Full table in the [write-up](https://espressosession.github.io/latent-studio/).
