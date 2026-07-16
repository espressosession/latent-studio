"""Step 4 of the LoRA pipeline: the training run — a thin subprocess wrapper around
diffusers' examples/text_to_image/train_text_to_image_lora.py (peft on the UNet, text
encoder frozen). Needs a cloned diffusers checkout (the notebook clones it) and reads
captions from metadata.jsonl if captioning.py wrote one, else a trivial fallback.

    run_training(...)                                                          # from pipeline.py
    python training/train_lora.py --dataset_dir ... --output_dir ... --trigger_word ...   # CLI
"""

import argparse
import glob
import json
import os
import subprocess
import sys

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")


def prepare_dataset(dataset_dir: str, trigger_word: str) -> str:
    """Fallback metadata.jsonl builder — only used if captioning.py hasn't
    already written one. Caption per image: a sibling <name>.txt if present,
    else just the trigger word."""
    meta_path = os.path.join(dataset_dir, "metadata.jsonl")
    if os.path.exists(meta_path):
        print(f"Using existing captions at {meta_path}")
        return meta_path

    images = [
        f for f in glob.glob(os.path.join(dataset_dir, "*")) if f.lower().endswith(IMAGE_EXTENSIONS)
    ]
    if not images:
        raise SystemExit(f"No images found in {dataset_dir}")

    with open(meta_path, "w", encoding="utf-8") as f:
        for image_path in images:
            caption_path = os.path.splitext(image_path)[0] + ".txt"
            if os.path.exists(caption_path):
                caption = open(caption_path, encoding="utf-8").read().strip()
            else:
                caption = trigger_word
            f.write(json.dumps({"file_name": os.path.basename(image_path), "text": caption}) + "\n")

    print(f"{len(images)} images -> {meta_path} (fallback captions)")
    return meta_path


def _ensure_diffusers_repo(diffusers_repo: str) -> str:
    """Ensure the diffusers example training script is available locally and
    return its path — clones the release tag matching the installed diffusers
    version (falls back to `main` for a dev/source install)."""
    script = os.path.join(diffusers_repo, "examples", "text_to_image", "train_text_to_image_lora.py")
    if os.path.exists(script):
        return script

    import diffusers

    version = diffusers.__version__.split("+")[0]
    if ".dev" in version:
        clone = ["git", "clone", "--depth", "1", "https://github.com/huggingface/diffusers.git", diffusers_repo]
        print(f"Cloning diffusers main (installed {version} is a dev build) for the training script...")
    else:
        clone = ["git", "clone", "--depth", "1", "--branch", f"v{version}",
                 "https://github.com/huggingface/diffusers.git", diffusers_repo]
        print(f"Cloning diffusers v{version} (matching the installed version) for the training script...")
    subprocess.run(clone, check=True)
    if not os.path.exists(script):
        raise SystemExit(f"Cloned diffusers but the training script is still missing at {script}")
    return script


def run_training(
    dataset_dir: str,
    output_dir: str,
    trigger_word: str,
    model_name: str = "stable-diffusion-v1-5/stable-diffusion-v1-5",
    diffusers_repo: str = "diffusers",
    max_train_steps: int = 1200,
    learning_rate: float = 1e-4,
    rank: int = 16,
    resolution: int = 512,
    train_batch_size: int = 1,
    gradient_accumulation_steps: int = 1,
    seed: int = 1337,
    checkpointing_steps: int = 300,
    resume_from_checkpoint: str | None = None,
) -> str:
    """Runs LoRA training and returns `output_dir` (final weights + checkpoint-*).
    `resume_from_checkpoint="latest"` continues an interrupted run instead of
    restarting from step 0 — only meaningful when `output_dir` persists (Drive)."""
    prepare_dataset(dataset_dir, trigger_word)

    import torch
    from accelerate.utils import write_basic_config

    # 8-bit Adam (bitsandbytes) and fp16 mixed precision both need CUDA; on any
    # other device fall back so the invocation stays valid instead of crashing on
    # a bitsandbytes/CUDA error. Real training still wants a CUDA GPU — this
    # pipeline is meant to run in Colab (T4), not locally on a Mac.
    on_cuda = torch.cuda.is_available()
    precision = "fp16" if on_cuda else "no"
    write_basic_config(mixed_precision=precision)

    # Let unsupported ops fall back to CPU when training on Apple Silicon (MPS).
    if not on_cuda and torch.backends.mps.is_available():
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

    training_script = _ensure_diffusers_repo(diffusers_repo)

    os.makedirs(output_dir, exist_ok=True)

    cmd = [
        sys.executable, "-m", "accelerate.commands.launch",
        f"--mixed_precision={precision}",
        training_script,
        f"--pretrained_model_name_or_path={model_name}",
        f"--train_data_dir={dataset_dir}",
        "--caption_column=text",
        f"--resolution={resolution}",
        "--center_crop",
        "--random_flip",
        f"--train_batch_size={train_batch_size}",
        f"--gradient_accumulation_steps={gradient_accumulation_steps}",
        "--gradient_checkpointing",
        *(["--use_8bit_adam"] if on_cuda else []),
        f"--mixed_precision={precision}",
        f"--max_train_steps={max_train_steps}",
        f"--learning_rate={learning_rate}",
        "--lr_scheduler=cosine",
        "--lr_warmup_steps=0",
        f"--rank={rank}",
        f"--seed={seed}",
        f"--checkpointing_steps={checkpointing_steps}",
        *([f"--resume_from_checkpoint={resume_from_checkpoint}"] if resume_from_checkpoint else []),
        f"--validation_prompt={trigger_word}",
        "--validation_epochs=1",
        f"--output_dir={output_dir}",
        # DataLoader workers spawn subprocesses that must pickle the script's local
        # transform fn — fine on Colab (Linux/fork), but breaks on macOS/Python 3.13
        # (spawn: "Can't get local object 'preprocess_train'"). Use workers only on CUDA.
        f"--dataloader_num_workers={2 if on_cuda else 0}",
    ]
    print("Running:", " ".join(cmd))
    # Stream the child's output line-by-line into the notebook. subprocess.run()
    # would let the training script write to the OS-level stdout/stderr, which
    # Jupyter/Colab does NOT capture into the cell — so real errors would vanish
    # into the runtime log and only an opaque CalledProcessError would surface.
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in process.stdout:
        print(line, end="")
    process.wait()
    if process.returncode != 0:
        raise RuntimeError(
            f"LoRA training failed (exit code {process.returncode}). The diffusers/accelerate "
            "error is in the streamed output just above this message."
        )

    lora_files = glob.glob(os.path.join(output_dir, "*.safetensors"))
    print("Trained LoRA weights:", lora_files)
    return output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_dir", required=True, help="Folder of training images (+ metadata.jsonl / .txt captions)")
    parser.add_argument("--output_dir", required=True, help="Where the trained .safetensors LoRA is written")
    parser.add_argument("--trigger_word", required=True, help="Caption fallback + validation prompt, e.g. 'sks'")
    parser.add_argument("--model_name", default="stable-diffusion-v1-5/stable-diffusion-v1-5")
    parser.add_argument("--diffusers_repo", default="diffusers", help="Path to a cloned huggingface/diffusers checkout")
    parser.add_argument("--max_train_steps", type=int, default=1200)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--checkpointing_steps", type=int, default=300)
    parser.add_argument("--resume_from_checkpoint", default=None, help="'latest' or a checkpoint-N dir to continue an interrupted run")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_training(
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        trigger_word=args.trigger_word,
        model_name=args.model_name,
        diffusers_repo=args.diffusers_repo,
        max_train_steps=args.max_train_steps,
        learning_rate=args.learning_rate,
        rank=args.rank,
        resolution=args.resolution,
        train_batch_size=args.train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        seed=args.seed,
        checkpointing_steps=args.checkpointing_steps,
        resume_from_checkpoint=args.resume_from_checkpoint,
    )
