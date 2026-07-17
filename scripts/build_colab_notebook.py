"""Generates the two Colab notebooks from source:

  * notebooks/latent_studio_colab.ipynb        <- src/latent_studio/*.py  (the app)
  * notebooks/lora_training_colab.ipynb         <- training/*.py           (the LoRA pipeline)

The parameter atlas has no notebook of its own: its grids are generated in the
app's own Compare mode, so the study and the product run on the same code.

Unlike a normal package, each notebook has no files and no imports between
sections — every module's code is flattened into the notebook's own namespace
(external imports collected into one cell up top, internal `from .x import y`
lines dropped since those names are already in scope by the time later cells
run). This mirrors how pipeline_V5.ipynb was structured: one flat, readable,
top-to-bottom script.

src/latent_studio/ and training/ remain the real source of truth (git-versioned,
used for local dev on the Mac) — never edit the notebooks by hand, rerun this:

    python scripts/build_colab_notebook.py
"""

import json
import os
import re

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(REPO_ROOT, "src", "latent_studio")
TRAINING_DIR = os.path.join(REPO_ROOT, "training")
APP_OUTPUT_PATH = os.path.join(REPO_ROOT, "notebooks", "latent_studio_colab.ipynb")
TRAINING_OUTPUT_PATH = os.path.join(REPO_ROOT, "notebooks", "lora_training_colab.ipynb")

# Order matters: a later module may reference names a `from .earlier_module
# import ...` used to bring in — once flattened, that only works if the earlier
# cell has already run and defined that name in the notebook's namespace. Each
# description explains what the section does and which constants to tweak.
APP_MODULE_FILES = [
    (
        "device.py",
        "Device detection",
        "Picks `cuda` / `mps` / `cpu` and the matching dtype (fp16 everywhere except "
        "plain CPU) so the exact same code runs on this Colab GPU and on a MacBook's "
        "Apple Silicon (MPS) during local development. Nothing here needs tweaking.",
    ),
    (
        "upscaler.py",
        "Upscaler (optional)",
        "A 2x latent upscaler (`stabilityai/sd-x2-latent-upscaler`, via diffusers' "
        "`StableDiffusionLatentUpscalePipeline`) for the app's Advanced-only Upscale "
        "button — deliberately not `StableDiffusionImg2ImgPipeline` cranked to a "
        "larger width/height, which is where SD1.5 starts duplicating anatomy/"
        "composition well before 1024px. Defined here, ahead of `pipeline_manager.py`, "
        "only because that module's `preload_models()` below references "
        "`UPSCALER_REPO` to warm this model's cache too.",
    ),
    (
        "registry.py",
        "Model & LoRA registry",
        "The single source of truth for what shows up in the app's Model and Style "
        "pickers. **Tweak `CHECKPOINTS`** to add/remove curated SD1.5 checkpoints "
        "(any Hugging Face repo compatible with `StableDiffusionPipeline`). **`LORAS`** "
        "points at the six shipped project LoRAs (Hokusai, Turner, Monet, Dürer, "
        "Hiroshige, Rembrandt) by their Hugging Face repo "
        "id — those are produced by the separate LoRA training notebook "
        "(`lora_training_colab.ipynb`) and download from the Hub automatically. "
        "Selecting a LoRA before it has been trained/pushed just fails gracefully.",
    ),
    (
        "pipeline_manager.py",
        "Pipeline manager",
        "Loads/unloads the SD1.5 base checkpoint and LoRA weights, keeping exactly "
        "one active pipeline at a time. Switching the **base model** is the expensive "
        "operation — it fully reloads and frees the old pipeline's memory first. "
        "Switching **which LoRA** is loaded is cheaper; changing only the LoRA "
        "**weight** (strength) is free — it's applied per-generation instead of "
        "reloading anything. **Tweak:** `DISABLE_SAFETY_CHECKER` is a local-only debug "
        "escape hatch — set it before importing this module to skip the safety checker "
        "while iterating offline. This notebook never sets it: the deployed app always "
        "runs with the safety checker enabled.",
    ),
    (
        "controlnet.py",
        "ControlNet (optional)",
        "Optional conditioning on an uploaded image via Canny edge detection and/or "
        "a Depth-Anything-V2 depth map, feeding into a `StableDiffusionControlNetPipeline` "
        "instead of the plain one. Supports one or both ControlNet types "
        "simultaneously (each with its own conditioning-scale weight). The rest of "
        "the app works fully without this section ever being used — it's defined "
        "here (before generation.py) only because generate() below type-hints "
        "against `ControlNetManager`.",
    ),
    (
        "generation.py",
        "Single-image generation",
        "`GenerationParams` is the one struct describing everything needed to "
        "reproduce a single image (prompt, negative prompt, size, CFG, steps, seed, "
        "LoRA weight, and optionally ControlNet type(s)/scale(s)). `generate()` is "
        "the single dispatch point that turns a `GenerationParams` into an actual "
        "image: it routes to the plain SD1.5 pipeline, or to the ControlNet pipeline "
        "if `controlnet_types` is set — the UI and the sweep engine below both call "
        "this one function, so neither has to know which pipeline is actually running.",
    ),
    (
        "metadata.py",
        "Metadata embed/read",
        "Every generated image's full settings are embedded directly into the PNG's "
        "own text chunk (`save_with_metadata`) and can be read back out of just the "
        "image file (`read_metadata`) — this is what makes 'reproduce this exact "
        "image from nothing but the file itself' possible, which is one of the "
        "mandatory live-demo steps.",
    ),
    (
        "grids.py",
        "Grid compositing + parameter sweeps",
        "`make_grid_image` composites a list of generated images into one labeled "
        "grid PNG. `sweep()` is a generic 1-2 dimension parameter sweep: give it "
        "`dim1=(field_name, (min, max), count)` — any of `cfg_scale`, `steps`, "
        "`lora_weight`, or `seed` — and optionally a `dim2` for a second axis, and "
        "it generates every combination and lays them out as a grid (`dim1` across "
        "columns, `dim2` across rows). For `seed` the pair means `(start, step)` "
        "rather than `(from, to)`: interpolating between two seeds is meaningless, "
        "but 'every 10th seed' isn't. This engine backs the Compare mode in the app "
        "below — which is also what generated the submitted Parameter Atlas.",
    ),
    (
        "icons.py",
        "Icons",
        "One monochrome stroke-icon set for the whole UI, so nothing in it is an emoji. "
        "Buttons get theirs as a `data:` URI (Gradio renders a button icon as an `<img>`, "
        "and this notebook has no files to point one at); the live state panel gets inline "
        "`<svg>`, where `currentColor` still works.",
    ),
    (
        "progress.py",
        "Live progress",
        "What puts a real progress bar — with an ETA — in front of the user instead of a "
        "spinner. A diffusers call blocks, so the app runs it on a worker thread and polls "
        "this `ProgressTracker`. The numbers come for free: diffusers' denoising loop and "
        "huggingface_hub's downloader are both **tqdm** subclasses, so patching `tqdm` once "
        "(`track_tqdm`) reports megabytes while a model downloads and steps while an image "
        "renders, with no callback plumbed through the pipeline. A second bar counts images "
        "('image 7 of 25') when a comparison grid is running.",
    ),
    (
        "design_tokens.py",
        "Design tokens: theme, copy, sweep ranges",
        "Pure data — the Gradio `THEME`, every string of UI copy, and the Compare "
        "tab's per-setting ranges. Nothing here calls `gr.Blocks`, so a wording or "
        "range tweak never touches app logic. Two kinds of copy are kept apart on "
        "purpose: a `DESC_*` is the static explanation rendered as Markdown under a "
        "control's title; a `STATUS_*` is the live reason a control is greyed out, "
        "and is the *only* thing a control's own `info=` ever carries (empty when "
        "the control is active) — `callbacks.gating_updates()` swaps a STATUS in "
        "and out. `SweepSpec` gives each Compare-tab setting its own slider range, "
        "default sweep range and image count — one global default would be "
        "meaningless across four differently-scaled fields (a CFG range of 3-15 "
        "vs. a seed range in the billions). `THEME` drives colour entirely through "
        "Gradio's own hue system (`primary_hue` — swap it to re-skin); the `.set()` "
        "override zeros the border/background on plain component-level blocks, while "
        "`gr.Column(variant=\"panel\")` still gets its box from its own separate "
        "`panel_*` tokens. `gr.Group()` is deliberately not used anywhere in app.py: "
        "its compiled CSS reads its background from `--border-color-primary`, not "
        "`--block-background-fill`, and that variable is also the real border color "
        "for dozens of unrelated elements (gallery thumbnails, focus rings) — no clean "
        "theme lever exists to switch it off without side effects elsewhere, so plain "
        "title+description+control spacing (via `_heading()`) replaces it instead. "
        "`THEME` is passed to `launch()` in the app.py section below, not the `Blocks` "
        "constructor (Gradio 6 moved it there).",
    ),
    (
        "metadata_view.py",
        "Metadata display + downloads",
        "Turns a generation result into what the user sees and can keep. A single "
        "image's metadata is a plain dict; a comparison grid's is "
        "`{\"cells\": [...], \"compare\": {...}}` — one metadata dict per cell plus "
        "the exact Compare-tab config (field/range/count per axis) that produced it. "
        "`_cells()`/`_compare_spec()` are the only things that read that shape "
        "directly; every other function goes through them, including for an older "
        "bare-list grid export with no stored sweep config, which `_compare_spec` "
        "reports as `None`. `settings_rows`/`settings_html`/`settings_json` read the "
        "cells — a swept field shows 'varies across the grid' rather than silently "
        "displaying the first cell's value — into the Advanced column and the "
        "downloadable JSON. `metadata_to_control_values` is the 'reuse these "
        "settings' path (also used by Import): it collapses ControlNet back to its "
        "single-select value, forces the seed mode to Fixed, and — new — restores "
        "Compare's field1/field2/Start/End/Steps from the stored `compare` block "
        "(or resets Compare to off if there isn't one, rather than leaving a stale "
        "sweep silently active). A pair of 'suppress the next reset' flags rides "
        "along in the same return tuple: Start/End/Steps have their own field-change "
        "cascade that re-ranges them to that field's generic default the instant "
        "field1/field2 changes, which would otherwise immediately clobber the exact "
        "values just restored here (see callbacks.dim_updates's `suppress` "
        "parameter). `_prepare_download_files` builds — and caches on the history "
        "entry — the PNG + settings-JSON pair a download button hands out. It's "
        "cached, and both download buttons carry their value reactively (set "
        "whenever generation finishes or a gallery thumbnail is selected) rather "
        "than building the file inside their own click handler, because a "
        "`gr.DownloadButton` that builds on click serves the *previous* click's "
        "file and lags one step behind.",
    ),
    (
        "callbacks.py",
        "Event handlers + interactive logic",
        "The app's interactive brain: the `manager`/`controlnet_manager` pipeline "
        "singletons (module-global, since the app is single-user/local), the "
        "gating logic that greys out controls, and every `on_*` event handler. "
        "Two rules drive the greying: a setting the Compare tab is currently "
        "sweeping is locked to that tab, so it has only one source; and Style "
        "strength / Reference strength need a style / reference picked first. The "
        "Compare tab's own field radio is its on/off switch (\"Nothing\" = off, no "
        "separate checkbox), and picking `seed` there forces Seed mode to Fixed in "
        "the same `gating_updates()` call — a random base would silently change "
        "what a seed comparison even means. `on_generate` is the one dispatch point "
        "for both modes (single image / comparison grid) and both pipelines "
        "(plain / ControlNet): the actual diffusers call blocks, so it runs on a "
        "worker thread while the generator yields the polled `ProgressTracker` "
        "HTML a few times a second — that's what puts a real bar and ETA in the "
        "state panel for the model download, the denoising steps, and 'image 7 of "
        "25' alike. The Generate button itself carries the live phase "
        "('Loading model…' / 'Generating…' / 'Unloading model…') plus how many "
        "images a click produces ('Generate 5 images'); the state panel carries "
        "only the bars/ETA and any error in plain language, since repeating the "
        "phase there too would just be the same words twice — errors never become "
        "a popup, so there's never a duplicate message either. `on_import_settings` "
        "is the same idea from the other direction: a settings JSON, or a PNG's "
        "embedded metadata via `metadata.read_metadata`, applied to the live "
        "controls exactly like 'Reuse these settings' — the only difference is the "
        "source is a file, not the current result, so it works across sessions too.",
    ),
    (
        "app.py",
        "App layout",
        "`build_app()` assembles the `gr.Blocks` UI and wires every control to its "
        "handler from callbacks.py — left column: prompt, avoid, the Generate "
        "button, the live state panel; center: the image and the prompt that made it; "
        "right (Advanced only): the exact settings, 'Reuse these settings', "
        "Import/Export (a settings JSON or a PNG this app produced, applied the same "
        "way Reuse does), and the image download button. Everything else sits in four "
        "tabs (Model & style, Tuning, Reference image, Compare).\n\n"
        "Two rules shape the layout. **Nothing containing a slider is ever "
        "hidden** — Gradio mounts a slider inside a `display:none` container at "
        "zero width and it stays invisible until touched, so a control that "
        "doesn't apply right now is greyed out instead, with the reason in its "
        "info line (`callbacks.gating_updates`). And **a setting has exactly one "
        "source** — a setting the Compare tab is sweeping greys out its normal "
        "control and says so. The `_title`/`_description`/`_heading`/"
        "`_compare_column` helpers exist because every control goes `container=False` "
        "(dropping Gradio's own component label), so each control's title/description "
        "is a plain Markdown line above it instead — deliberately not wrapped in "
        "`gr.Group()`, whose own box comes from a Gradio CSS quirk "
        "(`--border-color-primary`, not `--block-background-fill`) with no clean theme "
        "lever to switch off, since that variable is also the real border color for "
        "unrelated elements elsewhere (gallery thumbnails, focus rings). The last cell "
        "calls "
        "`build_app().launch(theme=THEME)` — Gradio 6 takes `theme` on `launch()`, "
        "not the `Blocks` constructor.",
    ),
]

# The LoRA automation pipeline (training/*.py), in dependency order (pipeline.py
# references every other module, so it comes last).
TRAINING_MODULE_FILES = [
    (
        "config.py",
        "Config",
        "Central config: the artist roster with their Art Institute of Chicago search terms, "
        "each one's **on-domain validation prompt** (a subject that artist actually worked "
        "on — one shared prompt would judge Rembrandt on a mountain landscape), the shared "
        "neutral `sks` trigger, the **`HF_USERNAME`** the LoRAs are pushed to, and every "
        "hyperparameter (dataset size, LoRA rank, steps-per-image, validation + scorecard "
        "settings). **Tweak here** to change artists or training settings — nothing "
        "downstream hard-codes them.\n\n"
        "**Why the roster looks the way it does.** Every artist is public domain "
        "(CC0) via the AIC API, but AIC's public-domain *coverage* per artist "
        "varies enormously — and that turned out to be the real constraint on "
        "this pipeline, not model quality. Measured image yield, and the outcome "
        "it led to:\n\n"
        "| artist | style | AIC public-domain images | outcome |\n"
        "|---|---|---|---|\n"
        "| Hokusai | ukiyo-e woodblock | plenty (trained 150) | **SHIP** |\n"
        "| Turner | atmospheric oil landscape | plenty (trained 150) | **SHIP** |\n"
        "| Dürer | Renaissance engraving/woodcut | plenty (trained 150) | **SHIP** (biggest gain from +data) |\n"
        "| Hiroshige | ukiyo-e landscape | ~2270 (trained 150) | **SHIP** |\n"
        "| Rembrandt | etching, chiaroscuro | ~980 (trained 150) | **SHIP** |\n"
        "| Monet | Impressionist landscape | 40 (ceiling) | **SHIP** (on-domain only, exception) |\n"
        "| Cézanne | Post-Impressionist | 32 (ceiling) | limitation — style never holds |\n"
        "| Cassatt | Impressionist figures | 46 (ceiling; ~236 exist, most lost in preprocessing) | limitation — data-thin |\n"
        "| Van Gogh | Post-Impressionist impasto | 19 (ceiling) | limitation — thin, and the corpus skews to B&W drawings |\n"
        "| Toulouse-Lautrec | Art Nouveau poster | ~2600 | untrained |\n"
        "| Renoir | Impressionist figures | ~780 | untrained |\n"
        "| Daumier | lithograph caricature | ~630 | untrained |\n"
        "| Utamaro | ukiyo-e portraits | ~280 | untrained |\n"
        "| Goya | dark expressive prints | ~275 | untrained |\n"
        "| Cameron | Victorian photography | ~150 | untrained (the only other woman with enough) |\n"
        "| Kollwitz | Expressionist prints | 0 | impossible — d. 1945, still in copyright |\n\n"
        "The public domain skews heavily male: of the women checked, only Cassatt "
        "and Cameron have enough public-domain work to train on (Morisot 20, "
        "Kauffmann 9, Bonheur 6) — a property of the commons, not of the "
        "shortlist.\n\n"
        "All nine trained artists were (re)trained under the current pipeline "
        "(150-image target where the public domain allows, steps = images × 40) "
        "and scored on CUDA in a before/after study. Dataset size is the ceiling: "
        "the five that reached 150 images ship; Monet (40) ships as an "
        "on-domain-only exception; Cézanne/Cassatt/Van Gogh stay documented "
        "limitations. `registry.LORAS` is the shipped cut of six.\n\n"
        "**`artist_match`** is the lowercased substring the returned "
        "`artist_title` must contain, and it must match AIC's *stored* spelling "
        "exactly (not necessarily the search spelling) — it's the guard against a "
        "same-surname artist slipping in. Notably: Dürer keeps the umlaut while "
        "Cézanne drops the accent in AIC's records; Daumier and Goya are stored "
        "under their full honorifics but the surname substring still matches; "
        "Cameron and Rembrandt need their *full* names, since AIC also holds "
        "unrelated artists sharing just the surname (David Young Cameron, "
        "Rembrandt Peale) whose work would otherwise blend into the style.",
    ),
    (
        "download.py",
        "Step 1 — download (Art Institute of Chicago)",
        "Downloads public-domain (CC0) artworks from the Art Institute of Chicago "
        "Open Access API (no API key). Unlike the Met's full-text search, AIC lets us "
        "query the `artist_title` field directly and filter to `is_public_domain` — we "
        "additionally verify each result's artist name (`artist_match`) so no loose "
        "match (e.g. a different 'Turner') slips into the dataset.",
    ),
    (
        "preprocess.py",
        "Step 2 — preprocess",
        "Cleans the raw downloads into a training set: convert to RGB, center-crop to "
        "a square, resize to 512², and drop outliers (too small, or too elongated to "
        "crop sensibly). Targets ~25-30 usable images per artist.",
    ),
    (
        "captioning.py",
        "Step 3 — caption (BLIP)",
        "Captions each image with BLIP-large (a local model, no API key) and writes "
        "diffusers' `metadata.jsonl`. Each caption is `<trigger>, <content>`, and "
        "medium/style words (painting, drawing, engraving, sepia, …) are stripped from "
        "the content so those attributes attach only to the trigger word — that's what "
        "makes the trigger monopolise the *style* while the words carry the *content*.",
    ),
    (
        "train_lora.py",
        "Step 4 — train",
        "The training run itself: a thin wrapper around diffusers' official "
        "`train_text_to_image_lora.py` (LoRA via peft on the UNet, text encoder "
        "frozen, fp16). Needs the cloned diffusers checkout from the setup cell above; "
        "saves the final weights + intermediate `checkpoint-*` snapshots.",
    ),
    (
        "validation.py",
        "Step 5 — validate (mandatory gate)",
        "Generates a checkpoint-comparison grid (a no-LoRA baseline plus each snapshot), "
        "a strength sweep to surface burn-in, and a style-generalisation grid across on- "
        "AND off-domain prompts (portrait, animal, city, still life) to check the style "
        "transfers beyond the training motifs — plus a saturation-based recommendation. "
        "The sweep + generalisation grids are built on the **recommended** snapshot (the "
        "one you'll export), not blindly on `final`. **A LoRA counts as done only once you "
        "have actually looked at these grids** — 'trained automatically' must not mean "
        "'shipped unchecked'.",
    ),
    (
        "scorecard.py",
        "Step 5b — scorecard (does it look like the artist?)",
        "Step 5 only ever compares the LoRA against a *no-LoRA baseline* — it never "
        "looks at the artist's actual work, so 'does this resemble the source data?' "
        "stayed a gut call. This step measures it. Against the same prompt+seed rendered "
        "without the LoRA, it reports **style gain** (CLIP similarity to the real corpus, "
        "measured on deliberately *modern* prompts — a bicycle, traffic lights — that no "
        "public-domain corpus can contain, so the gain can only be style and never a "
        "memorised training motif), **prompt drop** (did the LoRA stop listening to the "
        "prompt? the tell-tale of overfitting) and **burn-in** (oversaturation). It also "
        "renders a contact sheet with the artist's REAL artworks in the top row — the "
        "numbers rank the LoRAs, your eye still decides. Works on a Hub repo id too, so "
        "you can grade what is actually *published*.",
    ),
    (
        "export.py",
        "Step 6 — export + push",
        "Ships the confirmed LoRA: stages a local copy in `app_loras/` and pushes it "
        "to its Hugging Face repo (with a model card) under the standard "
        "`pytorch_lora_weights.safetensors` filename, so the app's registry can load "
        "it by repo id.",
    ),
    (
        "pipeline.py",
        "Orchestrator",
        "Ties the steps together as separate callables (`prepare_data` → `train` → "
        "`validate_artist` → `score_artist` → `export_artist`), keeping the validation "
        "gate and the scorecard as their own steps so you can inspect the grids before "
        "exporting. `revalidate_all` re-runs validation + scorecard over a whole list of "
        "already-trained artists (from their Drive checkpoints, no retraining) and writes "
        "one combined ranked table to Drive via `save_scorecard_table` — that's the "
        "single-device before/after comparison. These are what the sections below call.",
    ),
]

IMPORT_RE = re.compile(r"^(import \S.*|from \S+ import .*)$")
# Imports of our OWN code, in either spelling: `from .grids import ...` inside a package
# module, or `from latent_studio.grids import ...` in a script that imports the installed
# package. Both are dropped when flattening — those names are already in the notebook's
# namespace from an earlier cell.
INTERNAL_IMPORT_RE = re.compile(r"^from (\.\w+|latent_studio(\.\w+)?|training(\.\w+)?) import .*$")
MULTILINE_IMPORT_RE = re.compile(r"from (\S+) import \(([^)]*)\)")

# Hugging Face setup — must run BEFORE the imports cell. Both env vars below are read by
# huggingface_hub at *import* time (they become module constants), and diffusers pulls
# huggingface_hub in, so setting them later in the session is a silent no-op.
#
# Why this exists: anonymous Hub requests are rate-limited, and these notebooks pull
# several GB (SD 1.5 plus a second checkpoint plus LoRAs) — an unauthenticated first run
# crawls. Any token fixes that, even a read-only one.
#
# Why Xet is turned OFF: huggingface_hub 1.x routes downloads through Xet by default
# (hf_xet ships as its dependency), and in Colab that path *hangs* — reproducibly, part-way
# into a large file, with no error and no timeout. Measured on one runtime, same file, same
# minute: raw HTTPS straight at the CDN pulled 246 MB in 1s (206 MB/s) and the plain hub
# downloader finished in 1.2s, while the Xet path stalled at ~78 MB and never recovered
# (the Xet bridge also answered 403 to a plain HEAD). So the network, the region, the disk
# and the token are all fine — only the Xet backend is broken here. Disabling it costs some
# throughput on a warm cache and buys a download that actually completes.
# (Not the older HF_HUB_ENABLE_HF_TRANSFER either: current huggingface_hub deprecated it —
# it earns a FutureWarning and buys nothing.)
#
# Token lookup, in order: an already-set HF_TOKEN -> a Colab secret named HF_TOKEN (key
# icon in the sidebar; note it is only readable from the Colab UI, and times out
# otherwise) -> a cached login from a previous session -> an interactive login. The
# training notebook NEEDS a write token (it pushes); the app only reads.
HF_SETUP = (
    "import os\n"
    "\n"
    "# Read by huggingface_hub at import, so this must run before anything imports it.\n"
    "# Xet is the default download backend and it hangs in Colab (see the note above).\n"
    'os.environ["HF_HUB_DISABLE_XET"] = "1"\n'
    "\n"
    "\n"
    "def _hf_token():\n"
    '    if os.environ.get("HF_TOKEN"):\n'
    '        return "HF_TOKEN env var"\n'
    "    try:  # Colab secret (key icon in the sidebar, 'Notebook access' ON)\n"
    "        from google.colab import userdata\n"
    "\n"
    '        token = userdata.get("HF_TOKEN")\n'
    "        if token:\n"
    '            os.environ["HF_TOKEN"] = token\n'
    '            return "Colab secret"\n'
    "    except Exception:\n"
    "        pass  # not on Colab, or no secret set / not shared with this notebook\n"
    "    from huggingface_hub import get_token\n"
    "\n"
    '    return "cached login" if get_token() else None\n'
    "\n"
    "\n"
    "source = _hf_token()\n"
    "if source is None:\n"
    "    print(\n"
    '        "No token found — anonymous downloads are rate-limited, so logging in.\\n"\n'
    '        "Tip: save it once as a Colab secret named HF_TOKEN (key icon, left sidebar)\\n"\n'
    '        "and this cell will pick it up silently from now on."\n'
    "    )\n"
    "    from huggingface_hub import login\n"
    "\n"
    "    login()\n"
    "else:\n"
    '    print(f"Hugging Face token found ({source}) — downloads are authenticated and fast.")'
)

# Colab preinstalls torchao 0.10.0, and current diffusers refuses to run alongside
# anything below 0.16 ("Found an incompatible version of torchao"). We never use it —
# it is a quantization library — so remove it rather than upgrade it. The training
# notebook already did this; the app needs it too.
DROP_TORCHAO = "\n%pip uninstall -q -y torchao"

APP_PIP_INSTALL = (
    "%pip install -q "
    '"torch>=2.2" "torchvision>=0.17" "diffusers>=0.27" "transformers>=4.41" '
    '"accelerate>=0.31" "peft>=0.11" "safetensors>=0.4" "gradio>=6.0" '
    '"pillow>=10.0" "opencv-python-headless>=4.9" "numpy>=1.26" "huggingface_hub>=0.34"'
    + DROP_TORCHAO
)

# Pre-fetch every model before the app runs, retrying past the Hub's flaky CDN — but
# opt-in (RUN_PREFLIGHT defaults to False), not automatic.
#
# Why it exists at all: since 2026-07-13 the Hub's Xet CDN has intermittently rejected
# its own presigned URLs (403 "Auth failed: SignatureError: invalid key pair id" — see
# huggingface/datasets#8328). It hits public and private repos alike, with and without a
# token, and it is *intermittent*: the same file can fail and then succeed a minute
# later. As of the last check (2026-07-15), still open — an upstream comment that same
# day reported it recurring after an earlier "seems to work again". There is nothing to
# fix on our side, and no way to know from here whether it's live on any given run.
#
# Why it's opt-in rather than automatic: the app already has its own "Preload all
# models" button doing the same job on demand, so this cell is redundant on a happy
# path — and it costs something real to run automatically every time: every checkpoint
# and every style gets downloaded whether or not this session ever uses it (cache
# space, bandwidth, minutes), and it replaces the app's own lazy-loading animations
# (the whole point of a live demo) with a wall of preflight logs before you ever see
# the UI. Flip RUN_PREFLIGHT to True below and re-run this cell if a Generate click
# hits repeated download errors — the retry loop and its reasoning are unchanged,
# just no longer mandatory.
#
# What saves us when it IS needed: the Hub cache RESUMES. A failed attempt still keeps
# every file it already got, so a retry loop converges even when half the requests
# fail. This only works with Xet disabled (see HF_SETUP): the Xet client *hangs* on the
# broken CDN instead of raising, and you cannot retry a hang.
#
# Doing this here rather than lazily inside the app also puts a multi-GB download
# somewhere it can be watched and repeated, instead of inside a UI worker thread. LoRAs
# are staged into app_loras/, which pipeline_manager.set_lora already prefers over a
# repo id — so once this cell has actually run, styles load with no network at all.
PREFLIGHT_MARKDOWN = (
    "## 5. Download the models (optional — off by default)\n"
    "The app already downloads each model itself the first time you select it, with its "
    "own loading animation, and has a **\"Preload all models\"** button in its Setup tab "
    "for the same one-shot warm-up this cell does. Leaving `RUN_PREFLIGHT` at `False` "
    "skips this cell and relies on that — faster to reach the UI, and no cache space "
    "spent on a checkpoint or style this session never uses.\n\n"
    "Turn it on if a Generate click hits repeated download errors. Since "
    "**2026-07-13** the Hub's CDN has intermittently rejected its own signed download "
    "links (`403 … SignatureError: invalid key pair id`, "
    "[huggingface/datasets#8328](https://github.com/huggingface/datasets/issues/8328)) — "
    "a Hugging Face outage, unrelated to this project, that hits public repos too and is "
    "still open as of the last check. This cell's retry loop pulls every checkpoint and "
    "style **now** and works its way through the flakiness: the cache **resumes**, so "
    "each attempt keeps whatever it already downloaded even if most of them fail. If a "
    "run dies anyway, just run the cell again — nothing is lost.\n\n"
    "The styles are copied into `app_loras/`, which the pipeline prefers over the Hub, so "
    "after this cell they load with no network at all."
)

# The retry loop itself — shared by both notebooks, since both download from the same
# flaky Hub. Kept as its own string so the app's preflight and the training preflight
# cannot drift apart.
HUB_RETRY = (
    "import time\n"
    "\n"
    "RETRIES = 40\n"
    "\n"
    "\n"
    "def fetch(label, call):\n"
    '    """Retry past the Hub CDN\'s intermittent 403s. The cache resumes, so every\n'
    '    attempt makes progress even if most of them fail."""\n'
    "    for attempt in range(1, RETRIES + 1):\n"
    "        try:\n"
    "            return call()\n"
    "        except Exception as exc:\n"
    "            flaky = any(s in str(exc) for s in (\"403\", \"SignatureError\", \"Connection\"))\n"
    "            if not flaky or attempt == RETRIES:\n"
    "                raise\n"
    '            print(f"  {label}: Hub CDN hiccup ({attempt}/{RETRIES}), retrying...")\n'
    "            time.sleep(3)\n"
)

_PREFLIGHT_BODY = (
    HUB_RETRY
    + "\n"
    "import os\n"
    "import shutil\n"
    "\n"
    "from diffusers import DiffusionPipeline\n"
    "from huggingface_hub import hf_hub_download\n"
    "\n"
    "\n"
    "def fetch_checkpoint(cp):\n"
    "    # fp16 halves the download; not every community checkpoint publishes that\n"
    "    # variant (epiCRealism doesn't), so fall back rather than fail.\n"
    "    try:\n"
    "        return fetch(\n"
    "            cp.label,\n"
    "            lambda: DiffusionPipeline.download(cp.repo_id, variant=\"fp16\", use_safetensors=True),\n"
    "        )\n"
    "    except Exception:\n"
    '        print(f"  {cp.label}: no fp16 build, fetching full weights...")\n'
    "        return fetch(cp.label, lambda: DiffusionPipeline.download(cp.repo_id, use_safetensors=True))\n"
    "\n"
    "\n"
    "for cp in CHECKPOINTS:\n"
    '    print(f"Fetching {cp.label}...")\n'
    "    fetch_checkpoint(cp)\n"
    "\n"
    'print("Fetching the upscaler...")\n'
    "try:\n"
    "    fetch(\n"
    '        "Upscaler",\n'
    '        lambda: DiffusionPipeline.download(UPSCALER_REPO, variant="fp16", use_safetensors=True),\n'
    "    )\n"
    "except Exception:\n"
    '    print("  Upscaler: no fp16 build, fetching full weights...")\n'
    '    fetch("Upscaler", lambda: DiffusionPipeline.download(UPSCALER_REPO, use_safetensors=True))\n'
    "\n"
    "os.makedirs(APP_LORAS_DIR, exist_ok=True)\n"
    "for lora in LORAS:\n"
    "    if lora.path is None:\n"
    "        continue\n"
    '    print(f"Fetching {lora.label}...")\n'
    "    weights = fetch(\n"
    "        lora.label,\n"
    '        lambda lora=lora: hf_hub_download(lora.path, "pytorch_lora_weights.safetensors"),\n'
    "    )\n"
    "    # Staged locally, so the app loads styles without touching the Hub again.\n"
    '    shutil.copy(weights, os.path.join(APP_LORAS_DIR, f"{lora.id}-lora.safetensors"))\n'
    "\n"
    'print("\\nAll models cached. The app will load them from disk.")'
)


def _indent(code: str, spaces: int = 4) -> str:
    """Reindent a whole source block by a fixed amount — safer than hand-indenting a
    multi-line string literal (blank lines stay blank; nested indentation is preserved
    exactly since every line shifts by the same amount)."""
    pad = " " * spaces
    return "\n".join(pad + line if line else "" for line in code.split("\n"))


# The cell still runs top-to-bottom on a fresh runtime either way (the notebook must,
# per the brief) — RUN_PREFLIGHT just decides whether its body does anything. A Colab
# form field (the `# @param` comment), not a plain constant, so switching it on is a
# checkbox in the rendered cell, not an edit.
PREFLIGHT = (
    'RUN_PREFLIGHT = False  # @param {type:"boolean"}\n'
    "\n"
    "if RUN_PREFLIGHT:\n"
    + _indent(_PREFLIGHT_BODY)
    + "\n"
    "else:\n"
    '    print(\n'
    '        "Skipped — the app downloads each model itself on first use (with its own "\n'
    '        "loading animation), or use its \'Preload all models\' button. Tick "\n'
    '        "RUN_PREFLIGHT above and re-run this cell if a Generate click hits repeated "\n'
    '        "download errors."\n'
    "    )"
)

# The training notebook downloads from the same Hub, so it hits the same CDN outage —
# and it does so *three* times over, in three different steps: BLIP when captioning,
# SD 1.5 when training, CLIP when scoring. A 403 an hour into a run costs the run.
#
# One difference from the app's preflight, and it matters: the training modules load
# SD 1.5 through `from_pretrained(BASE_MODEL, torch_dtype=...)` with **no `variant`**
# (unlike pipeline_manager, which asks for fp16), so they read the repo's *default*
# weights. Pre-fetching the fp16 variant here would warm files nothing then reads, and
# the real download would still happen mid-run — exactly what this cell exists to
# prevent. So: no variant.
#
# BLIP and CLIP are fetched by calling the very loaders the pipeline calls, rather than
# by guessing filenames — whatever those functions pull is by definition what the run
# needs. Both cache their model in a module global, so they stay loaded for the steps
# that follow instead of being loaded twice.
TRAINING_PREFLIGHT = (
    HUB_RETRY
    + "\n"
    "from diffusers import DiffusionPipeline\n"
    "\n"
    'print("Fetching SD 1.5 (default weights — training does not use the fp16 variant)...")\n'
    'fetch("SD 1.5", lambda: DiffusionPipeline.download(BASE_MODEL, use_safetensors=True))\n'
    "\n"
    'print("Fetching BLIP (captioning, step 3)...")\n'
    'fetch("BLIP", _load_blip)\n'
    "\n"
    'print("Fetching CLIP (scorecard, step 5b)...")\n'
    'fetch("CLIP", _clip)\n'
    "\n"
    'print("\\nAll models cached. Download failures can no longer interrupt a training run.")'
)

# Non-diffusers deps the pipeline modules import (download/preprocess/caption/
# validation). diffusers itself is installed from the clone below (source), so
# it's deliberately not pinned here.
TRAINING_PIP_INSTALL = (
    "%pip install -q "
    '"torch>=2.2" "torchvision>=0.17" "transformers>=4.41" "safetensors>=0.4" '
    '"requests>=2.31" "pillow>=10.0" "numpy>=1.26" "huggingface_hub>=0.34"'
)

# Mirrors the course's known-good SD15_LoRA_Trainer setup exactly: use diffusers'
# maintained example training script from a source install (its `main` scripts
# pin check_min_version to the dev release, so the pip release mismatches), the
# training deps, then remove Colab's old torchao (which current peft rejects
# during LoRA injection — we don't use torchao). The `[ -d diffusers ]` guard
# makes re-running the cell cheap.
CLONE_DIFFUSERS = (
    "![ -d diffusers ] || git clone --depth 1 https://github.com/huggingface/diffusers.git\n"
    "%pip install -q ./diffusers\n"
    "%pip install -q -r diffusers/examples/text_to_image/requirements.txt\n"
    '%pip install -q "accelerate>=0.31" "transformers>=4.41" "peft>=0.11" '
    '"datasets>=2.19" "bitsandbytes>=0.43" safetensors xformers ftfy\n'
    "# Colab ships an old torchao (0.10) that current peft rejects during LoRA\n"
    "# injection (`Found an incompatible version of torchao`); we don't use it.\n"
    "%pip uninstall -q -y torchao"
)


def markdown_cell(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code_cell(text: str, title: str | None = None) -> dict:
    """`title` folds the cell into a collapsed Colab form (`#@title` + `cellView:
    form`): a one-line bar showing just the title, code hidden until clicked open.
    Reserved for boilerplate a reader doesn't need to see to follow the notebook —
    installs, login, the flattened module source — never for a cell whose value is
    meant to be read or edited directly (parameters, pipeline-run calls, the final
    launch call), which stay plain and always visible."""
    source = f"#@title {title}\n{text}" if title else text
    return {
        "cell_type": "code",
        "metadata": {"cellView": "form"} if title else {},
        "execution_count": None,
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


def _join_multiline_imports(content: str) -> str:
    """Collapse parenthesized multi-line imports (`from x import (\\n a,\\n b,\\n)`)
    into a single line so the line-by-line splitter below can classify them.
    Only touches parenthesized imports, so single-line imports are unaffected."""
    def repl(match: re.Match) -> str:
        names = ", ".join(n.strip() for n in match.group(2).replace("\n", " ").split(",") if n.strip())
        return f"from {match.group(1)} import {names}"

    return MULTILINE_IMPORT_RE.sub(repl, content)


def merge_imports(raw_imports: list[str]) -> list[str]:
    """Dedupes import lines and merges repeated `from X import ...` lines for
    the same X into one. Plain `import X` lines are just deduped. Order follows
    first appearance."""
    plain_imports: list[str] = []
    from_imports: dict[str, list[str]] = {}
    from_order: list[str] = []

    for line in raw_imports:
        match = re.match(r"^from (\S+) import (.+)$", line)
        if match:
            module, names = match.group(1), match.group(2)
            if module not in from_imports:
                from_imports[module] = []
                from_order.append(module)
            for name in (n.strip() for n in names.split(",")):
                if name not in from_imports[module]:
                    from_imports[module].append(name)
        elif line not in plain_imports:
            plain_imports.append(line)

    return plain_imports + [f"from {module} import {', '.join(from_imports[module])}" for module in from_order]


def split_imports_and_body(content: str) -> tuple[list[str], str]:
    """Pulls every import line out of a module's source. Imports of our own code
    (`from .x import y`, `from latent_studio.x import y`) are dropped outright — in a
    flattened notebook those names are already in scope from an earlier cell. External
    imports are returned separately so the caller can dedupe them into one shared cell."""
    content = _join_multiline_imports(content)
    imports: list[str] = []
    body_lines: list[str] = []
    for line in content.splitlines():
        if INTERNAL_IMPORT_RE.match(line):
            continue
        if IMPORT_RE.match(line):
            imports.append(line)
            continue
        body_lines.append(line)

    body = "\n".join(body_lines)
    body = body.split('if __name__ == "__main__":')[0]  # notebooks call things explicitly
    body = re.sub(r"\n{3,}", "\n\n", body).strip() + "\n"
    return imports, body


def flatten_modules(module_files: list[tuple], base_dir: str) -> tuple[list[tuple], list[str]]:
    """Returns (module_sources, merged_imports) for a list of modules."""
    module_sources = []
    raw_imports: list[str] = []
    for filename, title, description in module_files:
        with open(os.path.join(base_dir, filename), encoding="utf-8") as f:
            content = f.read()
        imports, body = split_imports_and_body(content)
        raw_imports.extend(imports)
        module_sources.append((filename, title, description, body))
    return module_sources, merge_imports(raw_imports)


def module_cells(module_sources: list[tuple], path_prefix: str) -> list[dict]:
    cells = []
    for filename, title, description, body in module_sources:
        cells.append(markdown_cell(f"### {title}\n_(`{path_prefix}/{filename}`)_\n\n{description}"))
        cells.append(code_cell(body, title=filename))
    return cells


def notebook(cells: list[dict]) -> dict:
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def build_app_notebook() -> dict:
    module_sources, all_imports = flatten_modules(APP_MODULE_FILES, SRC_DIR)

    cells = [
        markdown_cell(
            "# Latent Studio — SD 1.5 Image Generator\n\n"
            "Creative Coding Advanced capstone. Generated from `src/latent_studio/` "
            "(see `scripts/build_colab_notebook.py`) — flattened into one notebook "
            "namespace, no separate files or imports between sections, so you can "
            "read and run it top-to-bottom like a single script.\n\n"
            "The six shipped project LoRAs (Hokusai, Turner, Monet, Dürer, Hiroshige, "
            "Rembrandt) are produced by a **separate** notebook, "
            "`lora_training_colab.ipynb`, and loaded here from the Hugging Face Hub "
            "via the registry.\n\n"
            "**Runtime:** switch to a GPU runtime (T4) before running the generation "
            "cells. Building/debugging the interface can be done on a CPU runtime."
        ),
        markdown_cell("## 1. Install dependencies"),
        code_cell(APP_PIP_INSTALL, title="Install dependencies"),
        markdown_cell(
            "## 2. Hugging Face setup\n"
            "Everything this app loads is public, so a token is not needed for *access* — "
            "but anonymous Hub requests are **rate-limited**, and this notebook pulls "
            "several GB of checkpoints. A plain **read** token removes the throttle; "
            "without it the first model load crawls.\n\n"
            "The cell also **switches off Xet**, `huggingface_hub`'s default download "
            "backend, which hangs part-way into a large file on Colab (see the note in the "
            "source). Both settings are read when `huggingface_hub` is first imported, so "
            "**this cell must run before the imports below** — setting them later does "
            "nothing. If you have already imported anything, restart the runtime.\n\n"
            "On Colab, store the token once as a secret named **`HF_TOKEN`** (🔑 icon in "
            "the left sidebar, 'Notebook access' on). Otherwise this falls back to an "
            "interactive login."
        ),
        code_cell(HF_SETUP, title="Hugging Face login"),
        markdown_cell("## 3. Imports and Globals"),
        code_cell("\n".join(all_imports) + "\n", title="Imports"),
        markdown_cell(
            "## 4. App modules (reference)\n"
            "One card per file in `src/latent_studio/`, in the order they're defined — run "
            "them all to build the app below. Skip ahead if you're only here to launch it."
        ),
    ]

    cells.extend(module_cells(module_sources, "src/latent_studio"))

    cells.append(markdown_cell(PREFLIGHT_MARKDOWN))
    cells.append(code_cell(PREFLIGHT, title="Download the models (optional)"))

    cells.append(
        markdown_cell(
            "## 6. Launch the app\nSwitch to a GPU runtime first. `share=True` creates a "
            "public link outside this Colab session — the way to reach the app from "
            "another device, e.g. for a live demo. Load the printed link once to "
            "confirm it works before depending on it. `inbrowser=True` only opens a "
            "tab automatically on a **local** runtime — on Google's hosted runtime "
            "(the normal case for the T4 GPU) there's no local browser for it to "
            "reach, so use the printed share link instead; it's a harmless no-op "
            "either way, not an error."
        )
    )
    # Gradio 6 takes theme on launch(), not on the Blocks constructor — THEME is a
    # module-level name from the app.py section above. share=True creates a public
    # link reachable outside this Colab session. inline=False: don't embed an
    # iframe in the notebook output (Colab's default) now that a share link exists
    # to click instead. footer_links=[]: no API/Settings footer.
    cells.append(code_cell(
        "demo = build_app()\n"
        "demo.launch(share=True, theme=THEME, inline=False, inbrowser=True, footer_links=[])"
    ))

    return notebook(cells)


def build_training_notebook() -> dict:
    module_sources, all_imports = flatten_modules(TRAINING_MODULE_FILES, TRAINING_DIR)

    cells = [
        markdown_cell(
            "# Latent Studio — LoRA Training Pipeline\n\n"
            "Automated **artist → downloaded → preprocessed → captioned → trained → "
            "validated → published LoRA** pipeline for the Latent Studio capstone. "
            "Generated from `training/*.py` (see `scripts/build_colab_notebook.py`), "
            "flattened top-to-bottom.\n\n"
            "Produces the project's style LoRAs from public-domain (CC0) Art Institute of "
            "Chicago artworks and pushes the shipped ones to the Hugging Face Hub, from "
            "where the main app (`latent_studio_colab.ipynb`) loads them. **Six ship: "
            "Hokusai, Turner, Monet, Dürer, Hiroshige, Rembrandt.** The pipeline also "
            "trained Cézanne, Cassatt and Van Gogh — kept as documented limitations "
            "(dataset size was the ceiling, not model quality; see the roster table "
            "below) rather than pushed. This pipeline is a Colab workflow — deliberately "
            "**not** a live feature of the app.\n\n"
            "**Runtime:** needs a **GPU** runtime (T4) for the training + validation "
            "steps. Run one artist per pass (see the last section)."
        ),
        markdown_cell("## 1. Install dependencies"),
        code_cell(TRAINING_PIP_INSTALL, title="Install dependencies"),
        markdown_cell(
            "## 2. Hugging Face setup (required)\n"
            "This notebook **pushes** the trained LoRAs to your account (one repo per "
            "artist, `" + f"{_hf_username()}/latent-studio-<artist>-lora" + "`, created on "
            "first push), so the token here **must have `write` access** — a read token "
            "will create the repo and then fail on the upload, leaving an empty repo "
            "behind. It also lifts the anonymous rate limit on the multi-GB downloads.\n\n"
            "The cell also **switches off Xet**, `huggingface_hub`'s default download "
            "backend, which hangs part-way into a large file on Colab (see the note in the "
            "source). Both settings are read when `huggingface_hub` is first imported, so "
            "**this cell must run before the imports below** — setting them later does "
            "nothing. If you have already imported anything, restart the runtime.\n\n"
            "On Colab, store the token once as a secret named **`HF_TOKEN`** (🔑 icon in "
            "the left sidebar, 'Notebook access' on). Otherwise this falls back to an "
            "interactive login."
        ),
        code_cell(HF_SETUP, title="Hugging Face login"),
        markdown_cell("## 3. Clone the diffusers training script"),
        code_cell(CLONE_DIFFUSERS, title="Clone diffusers"),
        markdown_cell("## 4. Imports and Globals"),
        code_cell("\n".join(all_imports) + "\n", title="Imports"),
        markdown_cell(
            "## 5. Training modules (reference)\n"
            "One card per file in `training/`, in dependency order (`pipeline.py` "
            "references every other module, so it comes last) — run them all to define "
            "the pipeline below. Skip ahead if you're only here to run it."
        ),
    ]

    cells.extend(module_cells(module_sources, "training"))

    cells.append(
        markdown_cell(
            "## 6. Persistent storage (Google Drive)\n"
            "Mount Drive so datasets, checkpoints, validation grids and exports "
            "survive a runtime disconnect and are reusable across sessions — "
            "everything lands under `WORK_DIR/<artist>/{raw,dataset,output,validation,"
            "export}`. This is also what makes training **resumable** (below) and "
            "keeps the grids around for the documentation. Off Colab it falls back to "
            "a local `lora_work/` folder."
        )
    )
    cells.append(
        code_cell(
            "import os\n"
            "try:\n"
            "    from google.colab import drive\n"
            '    drive.mount("/content/drive")\n'
            '    WORK_DIR = "/content/drive/MyDrive/latent-studio-training"\n'
            "except ModuleNotFoundError:\n"
            '    WORK_DIR = "lora_work"  # not on Colab\n'
            "os.makedirs(WORK_DIR, exist_ok=True)\n"
            'print("Work dir:", WORK_DIR)',
            title="Mount Google Drive",
        )
    )
    cells.append(
        markdown_cell(
            "## 7. Pre-fetch the models (run this before training)\n"
            "This pipeline downloads from the Hub **three times**, in three different steps: "
            "**BLIP** when it captions, **SD 1.5** when it trains, **CLIP** when it scores. "
            "Since **2026-07-13** the Hub's CDN intermittently rejects its own signed download "
            "links (`403 … SignatureError: invalid key pair id`, "
            "[huggingface/datasets#8328](https://github.com/huggingface/datasets/issues/8328)) — "
            "a Hugging Face outage, unrelated to this project. Hitting that an hour into a "
            "training run costs you the run.\n\n"
            "So pull all three **now**, with retries. The cache **resumes**: every attempt keeps "
            "whatever it already downloaded, so a retry loop works its way through even when most "
            "requests fail. If the cell dies anyway, just run it again — nothing is lost. "
            "Afterwards the pipeline reads everything from disk."
        )
    )
    cells.append(code_cell(TRAINING_PREFLIGHT, title="Pre-fetch the models"))
    cells.append(
        markdown_cell(
            "## 8. Run the pipeline (one artist per pass)\n"
            "Pick an artist, then run the cells **in order**. The validation step is a "
            "gate: look at the grids before exporting. Repeat for the second artist by "
            "setting `ARTIST = \"turner\"` and re-running this section."
        )
    )
    cells.append(
        code_cell(
            '# Trained: "hokusai", "turner", "monet", "cezanne", "duerer", "hiroshige", "rembrandt"\n'
            '# Plenty of public-domain data, not trained yet: "lautrec", "renoir", "daumier",\n'
            '#   "utamaro", "goya", "cassatt", "cameron"\n'
            '# Too little public-domain data at AIC: "vangogh" (19), "kollwitz" (0 — still in copyright)\n'
            'ARTIST = "hokusai"\n'
            "dataset_dir = prepare_data(ARTIST, WORK_DIR)"
        )
    )
    cells.append(
        markdown_cell(
            "Train (GPU). Training length **scales with the dataset**: ~40 steps per image, "
            "floored at 800 and capped at 6000 — so a 150-image set runs ~6000 steps (roughly "
            "20 min on an L4/G4), while a 20-image set runs 800 instead of memorising itself at "
            "2400. The cell prints the derived step count. "
            "**Interrupted mid-run?** Re-run this cell as "
            '`train(ARTIST, work_dir=WORK_DIR, resume_from_checkpoint="latest")` to '
            "continue from the last checkpoint on Drive. Resume continues the *same* "
            "run only — after changing artist / trigger / hyperparameters, train fresh "
            "(leave resume off, and clear `WORK_DIR/<artist>/output` if it still holds "
            "an old run's checkpoints)."
        )
    )
    cells.append(code_cell("output_dir = train(ARTIST, work_dir=WORK_DIR)"))
    cells.append(
        markdown_cell(
            "Validate — inspect all three grids before trusting the recommendation. "
            "The weight-sweep + generalization grids are built on the **recommended** "
            "snapshot (the one you'll export), not blindly on `final`:"
        )
    )
    cells.append(code_cell('result = validate_artist(ARTIST, work_dir=WORK_DIR)\nresult["checkpoint_grid"]'))
    cells.append(code_cell('result["weight_grid"]'))
    cells.append(code_cell('result["generalization_grid"]  # does the style transfer to off-domain prompts?'))
    cells.append(markdown_cell("Persist the three grids to Drive (named by artist + snapshot — doubles as documentation evidence):"))
    cells.append(code_cell("save_validation_grids(ARTIST, result, WORK_DIR)"))
    cells.append(
        markdown_cell(
            "**(Optional) Re-check a specific checkpoint.** The recommendation is only "
            "a starting point. If a different snapshot looks better in the checkpoint "
            "grid above, judge it the same way here: set `STEP` to that checkpoint's "
            "step **number** (e.g. `1200`, or `\"final\"`) and run — it re-renders that "
            "snapshot's weight sweep + generalization grids (no retraining) and saves "
            "them to Drive. `result[\"lora_dirs\"]` lists the available steps."
        )
    )
    cells.append(code_cell('STEP = 1200  # checkpoint step to inspect (or "final")\ncmp = compare_snapshot(ARTIST, result, STEP, WORK_DIR)\ncmp["generalization_grid"]'))
    cells.append(code_cell('cmp["weight_grid"]'))
    cells.append(
        markdown_cell(
            "### Scorecard — does it actually look like the artist?\n"
            "The grids above compare the LoRA against the *base model*. They cannot tell "
            "you whether it resembles the **artist's real work**, which is the thing you "
            "actually trained for. This does: it renders a contact sheet with the real "
            "training artworks in the top row and our generations below, and reports "
            "**style gain**, **prompt drop** and **burn-in** (see the Step 5b section for "
            "what each one means).\n\n"
            "Grade the snapshot you're about to ship — or pass nothing to grade whatever "
            "is already **published on the Hub**, i.e. exactly what the app will load:\n\n"
            "```python\n"
            "score_artist(ARTIST, work_dir=WORK_DIR)                              # the published LoRA\n"
            "score_artist(ARTIST, recommended_dir(result), work_dir=WORK_DIR)     # a local snapshot\n"
            "```"
        )
    )
    cells.append(
        code_cell(
            "SNAPSHOT = recommended_dir(result)  # or: snapshot_dir(result, 1200)\n"
            "score = score_artist(ARTIST, SNAPSHOT, work_dir=WORK_DIR)\n"
            'print(score["verdict"], "| best strength:", score["recommended_weight"])\n'
            'score["sheet"]  # top row = the artist\'s REAL work — compare it against the rows below'
        )
    )
    cells.append(
        markdown_cell(
            "### Export the chosen snapshot — the scorecard **gates the push**\n"
            "Publishing is the step that matters: the Hub copy is what the app loads. So a "
            "LoRA that cannot show it resembles the artist **does not get pushed by "
            "accident** — `export_artist` refuses, stages the weights locally anyway (no "
            "work is lost) and tells you which fix applies:\n\n"
            "- **OVERCOOKED** → export an *earlier* snapshot. **No retraining** — the "
            "checkpoints are already on Drive: `export_artist(ARTIST, snapshot_dir(result, 900), ...)`.\n"
            "- **WEAK** → a *data* problem. More/cleaner images, or drop the artist.\n"
            "- **You looked at the sheet and disagree** → `force=True`. A heuristic must "
            "never overrule a human who has actually looked.\n\n"
            "Passing `score=score` reuses the scorecard from the cell above instead of "
            "generating everything a second time. The contact sheet is displayed either "
            "way — blocked or not, you always see what you shipped."
        )
    )
    cells.append(
        code_cell(
            "export_artist(\n"
            "    ARTIST, SNAPSHOT,\n"
            '    app_loras_dir=os.path.join(WORK_DIR, ARTIST, "export"),\n'
            "    work_dir=WORK_DIR,\n"
            "    score=score,  # reuse the scorecard above; drop this and it re-scores\n"
            "    # force=True,  # push despite a blocking verdict (only after you've looked)\n"
            ")"
        )
    )
    cells.append(
        markdown_cell(
            "### Re-push a snapshot that is already trained\n"
            "If an upload died halfway — the Hub repo exists but has no "
            "`pytorch_lora_weights.safetensors` — nothing is lost: the weights are on "
            "Drive. Push them again **without retraining**. Run the setup cells "
            "(install → login → imports → Drive), then this one, pointing it straight at "
            "the snapshot folder: `export_artist` takes a plain directory, so no `result` "
            "and no training run are needed.\n\n"
            "Two ways, pick by whether this snapshot has ever been judged:\n"
            "- **Never scored** (the usual case) → leave the gate on and run this on a "
            "**GPU** runtime. It scores first, shows you the contact sheet, and only then "
            "publishes.\n"
            "- **Already scored and you just want the bytes on the Hub** → `gate=False`. "
            "Then no images are generated at all and a **CPU runtime is enough** (costs no "
            "GPU quota). Only use this when you have actually looked at the LoRA before."
        )
    )
    cells.append(
        code_cell(
            '# ARTIST = "rembrandt"\n'
            '# SNAPSHOT = os.path.join(WORK_DIR, ARTIST, "output")  # or ".../output/checkpoint-3600"\n'
            "\n"
            "# GPU runtime — scores it, shows the sheet, then pushes if it passes:\n"
            '# export_artist(ARTIST, SNAPSHOT, app_loras_dir=os.path.join(WORK_DIR, ARTIST, "export"), work_dir=WORK_DIR)\n'
            "\n"
            "# CPU runtime — upload only, no scoring (ONLY for a snapshot you already judged):\n"
            '# export_artist(ARTIST, SNAPSHOT, app_loras_dir=os.path.join(WORK_DIR, ARTIST, "export"), gate=False)'
        )
    )

    cells.append(
        markdown_cell(
            "## 9. Re-validate & score everything already trained (batch, no retraining)\n"
            "The section above trains and ships **one** artist. This one does the reverse: it "
            "takes the artists you have **already** trained — whose checkpoints sit in Drive — "
            "and re-runs step 5 (validation) + step 5b (scorecard) over all of them in one pass, "
            "**without retraining**. It is the batched form of exactly those two steps, which is "
            "why it belongs here between *train* and a per-artist *export*.\n\n"
            "**Why it exists — the single-device rule.** MPS (a laptop) and CUDA (this T4) do "
            "**not** render the same image for the same seed: different backend kernels, "
            "different precision (fp32 vs fp16). So a score measured on one machine cannot be "
            "compared against a score from the other, and *everything that goes into the "
            "write-up has to come from one device*. Running this on Colab produces that one "
            "consistent set. Tag a run `\"before\"`, retrain the roster, run it again `\"after\"`, "
            "and the two tables in `WORK_DIR/scorecards/` are directly comparable — that is how "
            "you answer *\"did the hyperparameter fix actually help?\"* with a number instead of a "
            "feeling. A *\"it changed nothing\"* is just as publishable as a win.\n\n"
            "**Everything lands on Drive:** each artist's grids + contact sheet under "
            "`WORK_DIR/<artist>/validation/`, and one combined ranked table + all sheets under "
            "`WORK_DIR/scorecards/scorecard_<tag>.md`.\n\n"
            "**Source & cost.** It grades the **local Drive snapshot** by default (no Hub "
            "download — immune to the CDN outage; for the already-published LoRAs that is the "
            "same generation that is live). Pass `grade_published=True` to grade each artist's "
            "**published Hub** LoRA instead — exactly what the app loads today. Validation is the "
            "expensive part (a checkpoint grid + sweeps per artist); `validate=False` skips it "
            "and only scores. The list is ordered by demo importance, so a capped runtime still "
            "finishes the ones that matter."
        )
    )
    cells.append(
        code_cell(
            "# Already trained (checkpoints on Drive), ordered by how much each matters for the demo.\n"
            'ARTISTS_TO_SCORE = ["hiroshige", "hokusai", "turner", "rembrandt", "monet", "cezanne", "duerer"]\n'
            "\n"
            'before = revalidate_all(ARTISTS_TO_SCORE, work_dir=WORK_DIR, validate=True, tag="before")\n'
            "print(scorecard_table(before))"
        )
    )
    cells.append(
        markdown_cell(
            "After retraining the roster (rerun section 8 per artist), rerun the same batch with "
            "`tag=\"after\"` and diff the two tables. Cheaper variants, if you only need the "
            "numbers or want to grade what is actually published:\n\n"
            "```python\n"
            "# scores only, no validation grids (much faster):\n"
            'revalidate_all(ARTISTS_TO_SCORE, work_dir=WORK_DIR, validate=False, tag="after")\n'
            "# grade the PUBLISHED Hub LoRAs instead of the local snapshots:\n"
            'revalidate_all(ARTISTS_TO_SCORE, work_dir=WORK_DIR, grade_published=True, tag="published")\n'
            "```\n\n"
            "**Negative examples belong in the write-up too.** The thin-dataset artists whose "
            "checkpoints are on Drive but never shipped — **Van Gogh** (19 images), **Cassatt** "
            "(old 2400-step run) — are worth scoring precisely *because* they should score badly: "
            "a WEAK/OVERCOOKED contact sheet next to a SHIP one is the evidence that the quality "
            "gate works, not just an assertion that it does. Add their ids to the list to capture "
            "them (they will be skipped cleanly if their Drive folder isn't present)."
        )
    )

    return notebook(cells)


def _hf_username() -> str:
    """Read HF_USERNAME straight from training/config.py without importing torch etc."""
    with open(os.path.join(TRAINING_DIR, "config.py"), encoding="utf-8") as f:
        match = re.search(r'^HF_USERNAME\s*=\s*"([^"]+)"', f.read(), re.MULTILINE)
    return match.group(1) if match else "<your-hf-username>"


def write_notebook(nb: dict, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)
    print(f"Wrote {path}")


def main() -> None:
    write_notebook(build_app_notebook(), APP_OUTPUT_PATH)
    write_notebook(build_training_notebook(), TRAINING_OUTPUT_PATH)


if __name__ == "__main__":
    main()
