"""Step 6 of the LoRA pipeline: ship the validated LoRA to a local `app_loras/` copy
and to its Hugging Face repo (what registry.py points at), under the standard
`pytorch_lora_weights.safetensors` filename diffusers expects when loading by repo id,
together with a model card (trigger word, base model, licence/source)."""

import glob
import os
import shutil


def _find_weights(snapshot_dir: str) -> str:
    preferred = os.path.join(snapshot_dir, "pytorch_lora_weights.safetensors")
    if os.path.exists(preferred):
        return preferred
    files = glob.glob(os.path.join(snapshot_dir, "*.safetensors"))
    if not files:
        raise SystemExit(f"No .safetensors weights found in {snapshot_dir}")
    return files[0]


def stage_local(snapshot_dir: str, artist_id: str, app_loras_dir: str = "app_loras") -> str:
    os.makedirs(app_loras_dir, exist_ok=True)
    src = _find_weights(snapshot_dir)
    dst = os.path.join(app_loras_dir, f"{artist_id}-lora.safetensors")
    shutil.copyfile(src, dst)
    print(f"staged {src} -> {dst}")
    return dst


def _model_card(artist_label: str, trigger_word: str, repo_id: str) -> str:
    return f"""---
license: creativeml-openrail-m
base_model: stable-diffusion-v1-5/stable-diffusion-v1-5
tags:
- stable-diffusion
- stable-diffusion-diffusers
- lora
- text-to-image
---

# {artist_label} — SD 1.5 style LoRA

Style LoRA for Stable Diffusion 1.5, produced for the **Latent Studio** capstone
(Creative Coding Advanced, TH Nürnberg) by an automated artist→LoRA pipeline
(download → preprocess → caption → train → validate → export).

- **Trigger word:** `{trigger_word}`
- **Base model:** `stable-diffusion-v1-5/stable-diffusion-v1-5`
- **Training data:** public-domain (CC0) artworks from the Art Institute of
  Chicago Open Access collection, filtered to this artist.

```python
pipe.load_lora_weights("{repo_id}")
image = pipe("{trigger_word}, a small boat crossing a river beneath a mountain").images[0]
```
"""


def push_to_hub(weights_path: str, repo_id: str, artist_label: str, trigger_word: str, private: bool = False) -> str:
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo_id, repo_type="model", exist_ok=True, private=private)
    api.upload_file(
        path_or_fileobj=weights_path,
        path_in_repo="pytorch_lora_weights.safetensors",
        repo_id=repo_id,
    )
    api.upload_file(
        path_or_fileobj=_model_card(artist_label, trigger_word, repo_id).encode("utf-8"),
        path_in_repo="README.md",
        repo_id=repo_id,
    )
    print(f"pushed LoRA -> https://huggingface.co/{repo_id}")
    return repo_id


def export(
    snapshot_dir: str,
    artist_id: str,
    hf_repo: str,
    artist_label: str,
    trigger_word: str,
    app_loras_dir: str = "app_loras",
    push: bool = True,
    private: bool = False,
) -> str:
    """Stages the chosen snapshot locally and (unless push=False) pushes it to
    `hf_repo`. Returns the local staged path."""
    weights_path = stage_local(snapshot_dir, artist_id, app_loras_dir)
    if push:
        try:
            push_to_hub(weights_path, hf_repo, artist_label, trigger_word, private)
        except Exception as exc:
            raise RuntimeError(
                f"Hugging Face push to '{hf_repo}' failed — but the weights are safely staged "
                f"locally at {weights_path}, so no work is lost.\n"
                "This is almost always a token-permission issue: your HF token needs WRITE access. "
                "Create one at https://huggingface.co/settings/tokens (role 'Write'), re-run the "
                "login cell, then just call export_artist(...) again (no need to retrain).\n"
                f"Original error: {exc}"
            ) from exc
    else:
        print("push=False — skipped Hugging Face upload (local staging only).")
    return weights_path
