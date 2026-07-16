"""Step 3 of the LoRA pipeline: caption each image with BLIP (local, no API key) and
write a diffusers `metadata.jsonl`. Each caption is `"<trigger>, <content>"` with
medium/style words (painting, drawing, engraving, sepia, …) stripped from the
content, so the trigger monopolises the *style* and the words carry only the
*content* — the decoupling a style LoRA needs."""

import glob
import json
import os
import re

_processor = None
_model = None

# Words naming the medium/style rather than the content — removed from captions so
# they don't leak into content tokens (they should live on the trigger word only).
_MEDIUM_WORDS = [
    "black and white", "black-and-white", "oil painting", "watercolour", "watercolor",
    "woodblock print", "woodblock", "engraving", "etching", "mezzotint", "lithograph",
    "painting", "drawing", "sketch", "illustration", "print", "artwork", "photograph",
    "photo", "picture", "image", "poster", "sepia", "monochrome", "grayscale", "greyscale",
]
_MEDIUM_RE = re.compile(r"\b(" + "|".join(re.escape(w) for w in _MEDIUM_WORDS) + r")\b", re.IGNORECASE)


def _clean_caption(text: str) -> str:
    """Strip medium/style words and tidy the leftovers (BLIP's "a painting of X"
    becomes "a X"). Falls back to a neutral phrase if nothing content-y remains."""
    text = _MEDIUM_RE.sub(" ", text.lower())
    text = re.sub(r"\b(a|an|the)\s+of\b", r"\1", text)  # "a of mountain" -> "a mountain"
    text = re.sub(r"\b(a|an|the)\s+(a|an|the)\b", r"\2", text)  # collapse "a a" / "an a"
    text = re.sub(r"\s+", " ", text).strip(" ,.")
    text = re.sub(r"^(of|with|in|on)\b", "", text).strip(" ,.")
    return text or "a scene"


def _load_blip():
    global _processor, _model
    if _model is None:
        from transformers import BlipForConditionalGeneration, BlipProcessor

        name = "Salesforce/blip-image-captioning-large"
        _processor = BlipProcessor.from_pretrained(name)
        _model = BlipForConditionalGeneration.from_pretrained(name)
    return _processor, _model


def caption_folder(image_dir: str, trigger_word: str, max_new_tokens: int = 30) -> str:
    """Captions every PNG in `image_dir` and writes `metadata.jsonl` alongside
    them (the format diffusers' `--train_data_dir` + `--caption_column=text`
    expects). Returns the metadata path."""
    import torch
    from PIL import Image

    processor, model = _load_blip()
    paths = sorted(glob.glob(os.path.join(image_dir, "*.png")))
    if not paths:
        raise SystemExit(f"No images to caption in {image_dir}")

    meta_path = os.path.join(image_dir, "metadata.jsonl")
    with open(meta_path, "w", encoding="utf-8") as f:
        for path in paths:
            image = Image.open(path).convert("RGB")
            inputs = processor(image, return_tensors="pt")
            with torch.no_grad():
                output = model.generate(**inputs, max_new_tokens=max_new_tokens)
            description = processor.decode(output[0], skip_special_tokens=True).strip()
            caption = f"{trigger_word}, {_clean_caption(description)}"
            f.write(json.dumps({"file_name": os.path.basename(path), "text": caption}) + "\n")

    print(f"captioned {len(paths)} images -> {meta_path}")
    return meta_path
