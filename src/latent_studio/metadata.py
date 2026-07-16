"""Embed/read generation metadata on PNGs. This is what makes the mandatory
"reproduce an example image from its own metadata" demo step possible: the
settings travel with the file, not just in a separate log."""

import json

from PIL import Image
from PIL.PngImagePlugin import PngInfo

METADATA_KEY = "latent_studio"


def save_with_metadata(image: Image.Image, metadata: dict, path: str) -> None:
    png_info = PngInfo()
    png_info.add_text(METADATA_KEY, json.dumps(metadata))
    image.save(path, pnginfo=png_info)


def read_metadata(path: str) -> dict | None:
    with Image.open(path) as image:
        image.load()  # tEXt chunks after IDAT are only parsed once the image is fully read
        raw = image.info.get(METADATA_KEY)
    if raw is None:
        return None
    return json.loads(raw)


def format_metadata(metadata: dict) -> str:
    """Human-readable block for display under a generated image in the Gradio UI."""
    lines = [f"**{key}:** {value}" for key, value in metadata.items()]
    return "\n".join(lines)
