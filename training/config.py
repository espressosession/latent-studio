"""Config for the LoRA automation pipeline (artist name -> downloaded ->
preprocessed -> captioned -> trained -> validated -> pushed .safetensors).

Single source of truth shared by the pipeline modules and — for trigger words
and HF repo ids — by the app's registry.py, so what the app loads matches what
the pipeline produced. Tweak the ARTISTS table / hyperparameters here; nothing
downstream hard-codes an artist.
"""

from dataclasses import dataclass

# Hugging Face account the trained LoRAs are pushed to (and that the app's
# registry.py points at). The two repos are created on first push.
HF_USERNAME = "espressosession"

# Neutral, semantically-empty trigger shared by every project LoRA — the base
# model carries ~no prior for it, so any style at weight 0 is fully attributable
# to training. Keep in sync with registry.py.
TRIGGER_WORD = "sks"


@dataclass(frozen=True)
class Artist:
    id: str  # short slug, used in filenames + the HF repo name
    label: str  # human-readable, shown in the app's LoRA dropdown
    search_name: str  # queried against the Art Institute of Chicago `artist_title`
    artist_match: str  # lowercased substring the returned `artist_title` must contain
    trigger_word: str = TRIGGER_WORD  # neutral shared trigger; see TRIGGER_WORD above
    # A subject THIS artist actually worked on (content words only, no medium —
    # that's the trigger word's job). Falls back to VALIDATION_PROMPT.
    prompt: str = ""

    @property
    def hf_repo(self) -> str:
        return f"{HF_USERNAME}/latent-studio-{self.id}-lora"

    @property
    def validation_prompt(self) -> str:
        return self.prompt or VALIDATION_PROMPT


# The training roster — every artist is public domain (CC0) via the AIC API.
# AIC's public-domain *coverage* per artist is the real constraint on dataset size
# (see the notebook section for the measured yield table and the shipped/limitation
# split). Only TRAINED artists get a registry.LORAS entry. `artist_match` is the
# lowercased substring the returned `artist_title` must contain — it must match
# AIC's stored spelling exactly, and for a shared surname (Cameron, Rembrandt) needs
# the full name to avoid blending in a different artist's work.
ARTISTS: dict[str, Artist] = {
    "hokusai": Artist(
        id="hokusai",
        label="Hokusai (Ukiyo-e)",
        search_name="Katsushika Hokusai",
        artist_match="hokusai",
        prompt="a small boat crossing a river beneath a mountain, birds flying overhead",
    ),
    "turner": Artist(
        id="turner",
        label="Turner (Oil)",
        search_name="Joseph Mallord William Turner",
        artist_match="joseph mallord william turner",
        prompt="a sailing ship on a rough sea, the sun breaking through heavy haze",
    ),
    "cassatt": Artist(
        id="cassatt",
        label="Cassatt (Impressionism)",
        search_name="Mary Cassatt",
        artist_match="cassatt",
        prompt="a mother holding a small child beside a window",
    ),
    "vangogh": Artist(
        id="vangogh",
        label="Van Gogh (Post-Impressionism)",
        search_name="Vincent van Gogh",
        artist_match="van gogh",
        prompt="a wheat field under a swirling sky, cypress trees at the edge",
    ),
    "kollwitz": Artist(
        id="kollwitz",
        label="Kollwitz (Expressionism)",
        search_name="Käthe Kollwitz",
        artist_match="kollwitz",
        prompt="a woman holding her child close, her head bowed",
    ),
    "monet": Artist(
        id="monet",
        label="Monet (Impressionism)",
        search_name="Claude Monet",
        artist_match="monet",
        prompt="a pond with water lilies and a footbridge, morning light",
    ),
    "cezanne": Artist(
        id="cezanne",
        label="Cézanne (Post-Impressionism)",
        search_name="Paul Cezanne",
        artist_match="cezanne",  # AIC spells it without the accent here
        prompt="apples and a jug on a draped table",
    ),
    "duerer": Artist(
        id="duerer",
        label="Dürer (Renaissance)",
        search_name="Albrecht Dürer",
        artist_match="dürer",  # AIC keeps the umlaut — the search handles it fine
        prompt="a bearded man in a wide hat holding a book",
    ),
    "lautrec": Artist(
        id="lautrec",
        label="Toulouse-Lautrec (Art Nouveau)",
        search_name="Henri de Toulouse-Lautrec",
        artist_match="toulouse-lautrec",
        prompt="a dancer on a stage under bright lights",
    ),
    "hiroshige": Artist(
        id="hiroshige",
        label="Hiroshige (Ukiyo-e)",
        search_name="Utagawa Hiroshige",
        artist_match="utagawa hiroshige",
        prompt="travellers on a coastal road in the rain, a village in the distance",
    ),
    "rembrandt": Artist(
        id="rembrandt",
        label="Rembrandt (Etching)",
        search_name="Rembrandt van Rijn",
        artist_match="rembrandt van rijn",  # full name: guards against Rembrandt Peale
        prompt="an old man's face lit from one side, deep shadow behind him",
    ),
    "renoir": Artist(
        id="renoir",
        label="Renoir (Impressionism)",
        search_name="Pierre-Auguste Renoir",
        artist_match="renoir",
        prompt="people at a table in a sunlit garden",
    ),
    "daumier": Artist(
        id="daumier",
        label="Daumier (Lithograph)",
        search_name="Honoré Daumier",
        artist_match="daumier",  # AIC titles him "Honoré-Victorin Daumier"
        prompt="two men in coats and tall hats arguing in the street",
    ),
    "utamaro": Artist(
        id="utamaro",
        label="Utamaro (Ukiyo-e)",
        search_name="Kitagawa Utamaro",
        artist_match="utamaro",
        prompt="a woman with an elaborate hairstyle holding a fan",
    ),
    "goya": Artist(
        id="goya",
        label="Goya (Etching)",
        search_name="Francisco Goya",
        artist_match="goya",  # AIC titles him "Francisco José de Goya y Lucientes"
        prompt="a crowd of figures in the dark, one of them holding a lantern",
    ),
    "cameron": Artist(
        id="cameron",
        label="Cameron (Photography)",
        search_name="Julia Margaret Cameron",
        # MUST be the full name: AIC also holds David Young Cameron (a different
        # artist, ~18 works). A bare "cameron" would blend his etchings into her style.
        artist_match="julia margaret cameron",
        prompt="a woman with long loose hair, her head turned toward the light",
    ),
}

# Base checkpoint the LoRAs train on and are validated against.
BASE_MODEL = "stable-diffusion-v1-5/stable-diffusion-v1-5"

# --- dataset targets (step 1-2) ----------------------------------------------
TARGET_IMAGE_COUNT = 150  # stop once this many usable images survive preprocessing
DOWNLOAD_WIDTH = 1686  # IIIF request width — 2x margin so border-trimmed plates stay above MIN_IMAGE_SIZE
MIN_IMAGE_SIZE = 384  # discard images whose shorter side is smaller than this
MAX_ASPECT_RATIO = 1.7  # discard images more elongated than this (long/short side)
RESOLUTION = 512  # SD 1.5 training resolution

# --- training hyperparameters (step 4) ---------------------------------------
RANK = 24  # LoRA rank — higher captures more style detail (8-16 is a minimum for a style)
LEARNING_RATE = 1e-4
TRAIN_SEED = 1337

# Training length scales with the dataset (steps per image, not a fixed total) —
# see pipeline._training_steps(). A fixed total overtrains a small set and
# undertrains a large one.
STEPS_PER_IMAGE = 40
MIN_TRAIN_STEPS = 800  # floor, so a small set still converges
MAX_TRAIN_STEPS = 6000  # cap, so a huge set stays inside a sane Colab runtime
CHECKPOINT_COUNT = 8  # intermediate snapshots for the validation sweep -> steps // 8

# --- validation (step 5) -----------------------------------------------------
# Fallback on-domain prompt for artists that don't set their own `prompt` (see
# Artist.validation_prompt). Prefer the per-artist one: it is what makes the
# on-domain grids actually on-domain.
VALIDATION_PROMPT = "a small boat crossing a river beneath a mountain, birds flying overhead"
VALIDATION_SEEDS = [1000, 1001, 1002]
VALIDATION_CFG = 7.5
VALIDATION_STEPS = 30
# Swept at fixed seeds to surface burn-in — spans the app's Style-strength range;
# 0.0 is the neutral baseline, >1.0 is where a style typically starts "cooking".
VALIDATION_WEIGHTS = [0.0, 0.5, 1.0, 1.5]
# Deliberately MODERN subjects no pre-1900 painter could have depicted — if the
# style still shows, that's generalisation, not a memorised training motif (also
# what makes scorecard.py's style_gain metric airtight).
VALIDATION_OFFDOMAIN_PROMPTS = [
    "a cat sitting on a windowsill",
    "a busy city street with cars and traffic lights",
    "a plate of food on a kitchen table",
    "a person riding a bicycle",
]

# --- scorecard (step 5b) -----------------------------------------------------
# The scorecard answers the one question validation never asks: does the LoRA look
# like the ARTIST'S ACTUAL WORK? It compares generations against the real corpus
# instead of only against a no-LoRA baseline. See scorecard.py.
SCORECARD_CLIP_MODEL = "openai/clip-vit-base-patch32"
SCORECARD_REFERENCE_COUNT = 24  # real artworks embedded to form the style centroid
SCORECARD_WEIGHTS = [0.5, 1.0]  # LoRA strengths scored (0.0 is the implicit baseline)
SCORECARD_SEEDS = [1000, 1001]  # two seeds per prompt, so a metric isn't one lucky draw
# Thresholds are heuristics calibrated by eye, not laws — they sort LoRAs into
# "look at this one first", they don't ship anything on their own. The contact
# sheet is still the arbiter.
SCORECARD_MIN_STYLE_GAIN = 0.02  # below this, the adapter barely registers as a style
SCORECARD_MAX_PROMPT_DROP = 0.03  # above this, the LoRA stopped following the prompt
SCORECARD_MAX_BURN_IN = 1.25  # saturation ratio vs. baseline above which it's cooked
