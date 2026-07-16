"""Curated checkpoint/LoRA registry — single source of truth for the Gradio
selectors. Keep display names short (they go directly into the UI labels)."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Checkpoint:
    id: str
    label: str
    repo_id: str
    license: str
    note: str = ""


@dataclass(frozen=True)
class LoRA:
    id: str
    label: str
    path: str | None  # local path or HF repo id; None for the "no LoRA" entry
    trigger_word: str = ""
    license: str = ""
    note: str = ""


CHECKPOINTS: list[Checkpoint] = [
    Checkpoint(
        id="sd15-base",
        label="Stable Diffusion 1.5",
        repo_id="stable-diffusion-v1-5/stable-diffusion-v1-5",
        license="CreativeML OpenRAIL-M",
    ),
    Checkpoint(
        id="dreamshaper8",
        label="DreamShaper 8",
        repo_id="Lykon/dreamshaper-8",
        license="CreativeML OpenRAIL-M",
        note="Community SD1.5 checkpoint, stylized/photoreal blend.",
    ),
    Checkpoint(
        id="epicrealism",
        label="epiCRealism",
        repo_id="emilianJR/epiCRealism",
        license="CreativeML OpenRAIL-M",
        note="Community SD1.5 checkpoint, photoreal focus.",
    ),
]

DEFAULT_CHECKPOINT_ID = CHECKPOINTS[0].id

# The project LoRAs, produced by training/ and published to the Hugging Face Hub —
# `path` is the HF repo id, downloaded on first use. Repo ids + trigger words must
# match training/config.py. This is the SHIPPED roster (six of the nine trained
# LoRAs; Cézanne/Cassatt/Van Gogh stay documented limitations, not shipped).
LORAS: list[LoRA] = [
    LoRA(id="none", label="No style", path=None),
    LoRA(
        id="hokusai",
        label="Katsushika Hokusai",
        path="espressosession/latent-studio-hokusai-lora",
        trigger_word="sks",  # shared neutral trigger — matches config.TRIGGER_WORD
        license="CreativeML OpenRAIL-M (LoRA); training data CC0 (Art Institute of Chicago)",
        note="Style LoRA — Katsushika Hokusai woodblock prints, public domain.",
    ),
    LoRA(
        id="turner",
        label="J.M.W. Turner",
        path="espressosession/latent-studio-turner-lora",
        trigger_word="sks",  # shared neutral trigger — matches config.TRIGGER_WORD
        license="CreativeML OpenRAIL-M (LoRA); training data CC0 (Art Institute of Chicago)",
        note="Style LoRA — J.M.W. Turner oil landscapes/seascapes, public domain.",
    ),
    LoRA(
        id="monet",
        label="Claude Monet",
        path="espressosession/latent-studio-monet-lora",
        trigger_word="sks",
        license="CreativeML OpenRAIL-M (LoRA); training data CC0 (Art Institute of Chicago)",
        note="Style LoRA — Claude Monet Impressionist landscapes, public domain.",
    ),
    LoRA(
        id="duerer",
        label="Albrecht Dürer",
        path="espressosession/latent-studio-duerer-lora",
        trigger_word="sks",
        license="CreativeML OpenRAIL-M (LoRA); training data CC0 (Art Institute of Chicago)",
        note="Style LoRA — Albrecht Dürer Renaissance engravings/woodcuts, public domain.",
    ),
    LoRA(
        id="hiroshige",
        label="Utagawa Hiroshige",
        path="espressosession/latent-studio-hiroshige-lora",
        trigger_word="sks",
        license="CreativeML OpenRAIL-M (LoRA); training data CC0 (Art Institute of Chicago)",
        note="Style LoRA — Utagawa Hiroshige ukiyo-e landscapes, public domain.",
    ),
    LoRA(
        id="rembrandt",
        label="Rembrandt van Rijn",
        path="espressosession/latent-studio-rembrandt-lora",
        trigger_word="sks",
        license="CreativeML OpenRAIL-M (LoRA); training data CC0 (Art Institute of Chicago)",
        note="Style LoRA — Rembrandt etchings, chiaroscuro, public domain.",
    ),
]

DEFAULT_LORA_ID = "none"


def get_checkpoint(checkpoint_id: str) -> Checkpoint:
    for cp in CHECKPOINTS:
        if cp.id == checkpoint_id:
            return cp
    raise KeyError(f"Unknown checkpoint id: {checkpoint_id}")


def get_lora(lora_id: str) -> LoRA:
    for lora in LORAS:
        if lora.id == lora_id:
            return lora
    raise KeyError(f"Unknown LoRA id: {lora_id}")
