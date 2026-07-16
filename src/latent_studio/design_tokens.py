"""Design tokens: theme, the copy shown in the UI, and the Compare tab's sweep
ranges. Pure data — no gr.Blocks calls — so tweaking a range or a wording never
touches app logic."""

from dataclasses import dataclass

import gradio as gr

WIDTH = HEIGHT = 512
MAX_SEED = 2**32 - 1
MAX_IMAGES = 100  # a comparison beyond this is a runaway, not a study

# The Prompt box starts pre-filled with a random example. Content only, no medium/
# style words (no "35mm film", "painterly", "cyberpunk") — a style LoRA supplies the
# look, so the wording is chosen to give it clean subjects to work with.
EXAMPLE_PROMPTS = [
    "a great wave breaking over small fishing boats, seen from the shore",
    "a lone mountain rising above layered clouds at sunrise",
    "a wooden bridge crossing a river in a rain shower",
    "travellers resting under a large pine tree beside a dirt road",
    "a small boat drifting on a calm lake surrounded by lily pads",
    "a field of poppies stretching toward a distant windmill",
    "a rabbit sitting still in short grass, seen up close",
    "an old man's face lit from one side, deep shadow behind him",
    "a ship battling a storm at sea, waves crashing over the deck",
    "a garden path lined with irises leading to a small footbridge",
    "a hand holding a bundle of wildflowers against a plain background",
    "a harbor at dusk with fishing boats returning to shore",
    "a quiet village street after snowfall, footprints in the snow",
    "a hawk perched on a bare branch against a pale sky",
    "an old stone watermill beside a rushing stream",
    "a woman reading a letter by a window, soft daylight",
]

CONTROLNET_CHOICES = [("Off", "off"), ("Follow the outlines", "canny"), ("Follow the depth", "depth")]

# DESC_* = the static Markdown shown under a control's title. STATUS_* = the live
# reason a control is greyed out, swapped into its info= by callbacks.gating_updates.
DESC_PROMPT = "What should be in the picture. Naming a subject, a light and a look works best."
DESC_AVOID = (
    "Anything to keep out of the picture — leave it empty if nothing bothers you. "
    "(This is the negative prompt.)"
)
DESC_MODEL = (
    "The engine that paints. **Stable Diffusion 1.5** is the plain original, a neutral "
    "all-rounder. **DreamShaper 8** is a community retrain blending stylized and photoreal "
    "looks. **epiCRealism** is a community retrain pushed hard toward photorealism. "
    "Switching reloads the model, which takes a moment."
)
DESC_STYLE = (
    "An artist's hand laid over the model. Each is a **LoRA** trained for this project on that "
    "artist's public-domain works from the Art Institute of Chicago, then scored against the "
    "real paintings before it was allowed in here."
)
DESC_STYLE_STRENGTH = (
    "How far the style is pushed. 0 is the plain model, 1 is the style as it was trained, and "
    "above ~1.2 it usually burns — colours go flat and shapes fall apart."
)
DESC_CFG = (
    "Called guidance scale (CFG). How literally the model takes your prompt. 1–5 wanders off "
    "and invents; 7–9 is the sweet spot; above ~14 it forces the prompt through and the image "
    "turns harsh and over-contrasted."
)
DESC_STEPS = (
    "Called inference steps. How many passes the model makes to sharpen the image. Under 15 it "
    "stays smudgy, 25–35 is plenty, past ~50 you mostly pay time for nothing. Doubling the "
    "steps doubles the wait."
)
DESC_SEED = (
    "The random starting point behind the image. A new one every click gives you variety; "
    "fix it and the same settings always return the exact same image — that is how a "
    "result is reproduced."
)
DESC_REFERENCE = (
    "Optional, and called ControlNet. Give it a picture and the result keeps that picture's "
    "composition — either its outlines, or how near and far its parts are. Your prompt still "
    "decides what things are made of."
)
DESC_REFERENCE_SCALE = (
    "How tightly the result sticks to the reference. Around 0.4 it is a loose suggestion, 1.0 "
    "follows the shapes closely, above ~1.5 it traces them and ignores your prompt."
)
DESC_COMPARE = (
    "Generate a grid instead of one image — the fastest way to see what a setting "
    "actually does, laid out side by side."
)
DESC_ADVANCED = "Adds a panel with the full settings of the current image, and a button to reuse them on a new one."
DESC_PRELOAD = (
    "Each model downloads the first time you use it. Pull **all** of them now — handy right "
    "before a live demo, so the first switch isn't a wait. It only fills the cache; nothing is "
    "kept in memory."
)

STATUS_SWEPT = "Being compared right now — set its range in the Compare tab."
STATUS_STYLE_OFF = "Pick a style first."
STATUS_SEED_RANDOM = "Switch to a fixed seed to type your own."
STATUS_REFERENCE_OFF = "Pick a reference type first."


@dataclass(frozen=True)
class SweepSpec:
    """One Compare-tab setting: its slider range, default sweep range, and image count."""

    label: str
    minimum: float
    maximum: float
    step: float
    start: float
    end: float
    count: int
    max_count: int
    info: str


# Chosen per setting — one global default range/count would be meaningless across
# four very differently-scaled fields (a CFG range of 3-15 vs. a seed range in the
# billions).
SWEEP_SPECS: dict[str, SweepSpec] = {
    "cfg_scale": SweepSpec(
        "Prompt strength", 1.0, 20.0, 0.05, 3.0, 15.0, 5, 8,
        "Compares how literally the prompt is taken. Low values drift, high values "
        "get harsh — the useful range to look at is roughly 3 to 15.",
    ),
    "steps": SweepSpec(
        "Detail", 1, 100, 1, 10, 50, 5, 8,
        "Compares how much refinement the image gets. The interesting part is the "
        "low end: the difference between 10 and 30 is large, between 50 and 100 tiny.",
    ),
    "lora_weight": SweepSpec(
        "Style strength", 0.0, 1.5, 0.05, 0.0, 1.0, 5, 8,
        "Compares how hard the style is pushed. Sweep 0 → 1 to see it take hold, or "
        "past 1.2 to find the point where it burns.",
    ),
    "seed": SweepSpec(
        "Seed", 0, MAX_SEED, 1, 1312, 1, 9, 25,
        "Compares different random starting points with everything else held still — "
        "this is what shows you the spread a prompt can produce.",
    ),
}
SWEEP_CHOICES = [(spec.label, key) for key, spec in SWEEP_SPECS.items()]

# Colour comes entirely from Gradio's hue system (primary_hue); the .set() below is
# structural only. Passed to launch(), not the Blocks constructor (Gradio 6 moved it).
THEME = gr.themes.Base(
    primary_hue="rose",
    secondary_hue="stone",
    neutral_hue="stone",
    text_size="lg",
    spacing_size="lg",
    radius_size="xxl",
    font=[gr.themes.GoogleFont("Space Grotesk"), "ui-sans-serif", "system-ui", "sans-serif"],
    font_mono=[gr.themes.GoogleFont("Space Mono"), "ui-monospace", "Consolas", "monospace"],
).set(
    # Zeros the border/background on plain component-level blocks (Radio/Slider/etc.
    # already go container=False in app.py, so this mostly matters for anything that
    # doesn't). Deliberately not used to fight gr.Group()'s own box — its background
    # comes from a separate, widely-shared Gradio CSS variable with no clean lever —
    # app.py doesn't use gr.Group() at all for that reason; see _heading()'s docstring.
    block_background_fill="transparent",
    block_background_fill_dark="transparent",
    block_border_width="0px",
    block_border_width_dark="0px",
)
