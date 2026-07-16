"""Step 1 of the LoRA pipeline: download an artist's public-domain (CC0) artworks
from the Art Institute of Chicago Open Access API (no key). AIC over the Met for its
real `artist_title` filter + `is_public_domain` flag (the Met's full-text search left
only ~2 genuinely-Hokusai PD images); each result's name is verified against
`artist_match` so a loose match never slips in."""

import os
import random
import time

import requests

SEARCH_URL = "https://api.artic.edu/api/v1/artworks/search"
IIIF_BASE = "https://www.artic.edu/iiif/2"
# AIC asks API users to identify themselves via this header.
HEADERS = {"AIC-User-Agent": "Latent Studio capstone (github.com/espressosession/latent-studio)"}


def _search_page(search_name: str, page: int, per_page: int) -> list[dict]:
    params = {
        "query[bool][must][][match][artist_title]": search_name,
        "query[bool][must][][term][is_public_domain]": "true",
        "fields": "id,artist_title,image_id",
        "limit": per_page,
        "page": page,
    }
    resp = requests.get(SEARCH_URL, params=params, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    return resp.json().get("data", [])


def download_artist(
    search_name: str,
    out_dir: str,
    artist_match: str,
    target: int = 60,
    per_page: int = 100,
    request_delay: float = 0.1,
    pool_factor: int = 4,
    seed: int = 0,
    image_width: int = 1686,
) -> int:
    """Downloads up to `target` public-domain artworks whose `artist_title` contains
    `artist_match` into `out_dir`. Gathers a larger candidate pool first, then takes
    a deterministically shuffled subset so files spread across the artist's whole
    collection rather than clustering on the first few series. Returns the count
    downloaded."""
    os.makedirs(out_dir, exist_ok=True)

    candidates: list[dict] = []
    pool_cap = target * pool_factor
    page = 1
    while len(candidates) < pool_cap:
        try:
            results = _search_page(search_name, page, per_page)
        except Exception as exc:
            print(f"AIC search failed on page {page}: {exc}")
            break
        if not results:
            break
        for art in results:
            if artist_match not in (art.get("artist_title") or "").lower():
                continue  # loose match for a different artist — skip
            if art.get("image_id"):
                candidates.append(art)
        page += 1

    random.Random(seed).shuffle(candidates)  # spread across the collection, reproducibly
    selected = candidates[:target]

    saved = 0
    for i, art in enumerate(selected):
        url = f"{IIIF_BASE}/{art['image_id']}/full/{image_width},/0/default.jpg"
        try:
            content = requests.get(url, headers=HEADERS, timeout=60).content
        except Exception:
            continue
        with open(os.path.join(out_dir, f"{i:04d}_{art['id']}.jpg"), "wb") as f:
            f.write(content)
        saved += 1
        time.sleep(request_delay)  # be polite to the API

    print(f"'{search_name}': downloaded {saved} public-domain images (from {len(candidates)} candidates) -> {out_dir}")
    return saved
