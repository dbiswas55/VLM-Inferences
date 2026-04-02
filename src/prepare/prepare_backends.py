# =============================================================================
# Backend Preparation Guide
# Covers: Python environment · Ollama · MLX-VLM · vLLM · HuggingFace
# =============================================================================


# ── 1. Python Virtual Environment ────────────────────────────────────────────
#
# Create and activate:
#   python3 -m venv venv312
#   source venv312/bin/activate                  # macOS / Linux
#   venv312\Scripts\activate                     # Windows
#
# Install all dependencies:
#   pip install mlx mlx-vlm torch torchvision Pillow transformers accelerate\
#               huggingface_hub python-dotenv openai google-genai
#
# Configure environment — create a .env file at the project root:
#   HF_TOKEN=hf_your_token_here    # huggingface.co/settings/tokens
#   HF_HOME=.cache/huggingface     # optional; use absolute path on a cluster
#   GEMINI_API_KEY=...
#   OPENAI_API_KEY=...
#   ANTHROPIC_API_KEY=...


# ── 2. Ollama ─────────────────────────────────────────────────────────────────
#
# Install (macOS):  brew install ollama
# Install (Linux):  curl -fsSL https://ollama.com/install.sh | sh
#
# Pull a model and start the server:
#   ollama pull gemma3:4b
#   ollama serve                    # API base: http://localhost:11434/v1
#
# Useful commands:
#   ollama list                     # list downloaded models
#   ollama rm gemma3:4b             # remove a model
#
# Browse models: https://ollama.com/library


# ── 3. MLX-VLM (macOS / Apple Silicon only) ──────────────────────────────────
#
# Download a model into the HF cache (see HuggingFace section below), then serve:
#   Make sure to activate the Python environment where you installed mlx-vlm, then run: source venv312/bin/activate
#   python -m mlx_vlm.server --model mlx-community/gemma-3-4b-it-qat-4bit --port 8080
#   API base: http://localhost:8080/v1    (OpenAI-compatible)
#
# Browse available MLX models:
#   https://huggingface.co/mlx-community
#   Gemma3:   https://huggingface.co/collections/mlx-community/gemma-3-qat
#   Qwen3-VL: https://huggingface.co/collections/mlx-community/qwen3-vl


# ── 4. vLLM (CUDA / Linux — not yet applicable on macOS) ─────────────────────
#
# Install:  pip install vllm
#
# Serve a model:
#   vllm serve Qwen/Qwen3-VL-4B-Instruct --port 8000
#   API base: http://localhost:8000/v1    (OpenAI-compatible)
#
# Note: vLLM requires a CUDA-capable GPU. macOS support is not yet stable.


# ── 5. HuggingFace Model Management ──────────────────────────────────────────
#
# The Python functions below handle:
#   download_model()                — download a repo into the HF cache
#   list_cached_models()            — list all cached models with sizes
#   delete_cached_model()           — delete a specific model by repo id
#   delete_cached_model_interactive() — interactive prompt to pick and delete
#
# Run this script directly to download + inspect the cache:
#   python src/prepare/prepare_backends.py
#
# =============================================================================

import os
from dotenv import load_dotenv
from huggingface_hub import login, snapshot_download, scan_cache_dir

# Load HF_TOKEN and HF_HOME from .env before any huggingface_hub calls
load_dotenv()


def hf_login():
    """Interactive HF login — only needed for gated models not covered by HF_TOKEN in .env."""
    login()


def download_model(model_id: str):
    """Download all files for a model repo into the local HF cache (HF_HOME)."""
    print(f"Downloading {model_id} ...")
    snapshot_download(repo_id=model_id, token=os.getenv("HF_TOKEN"))
    print(f"Done: {model_id}")


def list_cached_models():
    """List all models in the HF cache with their sizes."""
    cache_info = scan_cache_dir()
    repos = sorted(cache_info.repos, key=lambda r: r.size_on_disk, reverse=True)

    if not repos:
        print("No models found in cache.")
        return

    total = sum(r.size_on_disk for r in repos)
    print(f"\n{'#':<4} {'Model':<55} {'Size':>10}  Revisions")
    print("-" * 85)
    for i, repo in enumerate(repos, 1):
        size_gb = repo.size_on_disk / 1024 ** 3
        revisions = len(repo.revisions)
        print(f"{i:<4} {repo.repo_id:<55} {size_gb:>8.2f} GB  {revisions}")
    print("-" * 85)
    print(f"{'Total:':<60} {total / 1024 ** 3:>8.2f} GB\n")
    return repos


def delete_cached_model(model_id: str):
    """Delete a specific model from the HF cache by repo id."""
    cache_info = scan_cache_dir()
    matches = [r for r in cache_info.repos if r.repo_id == model_id]

    if not matches:
        print(f"Model '{model_id}' not found in cache.")
        return

    repo = matches[0]
    size_gb = repo.size_on_disk / 1024 ** 3
    confirm = input(f"Delete '{model_id}' ({size_gb:.2f} GB)? [y/N]: ").strip().lower()
    if confirm != "y":
        print("Aborted.")
        return

    # Mark all revisions for deletion and commit
    delete_strategy = cache_info.delete_revisions(
        *[rev.commit_hash for rev in repo.revisions]
    )
    print(f"Freeing {delete_strategy.expected_freed_size_str} ...")
    delete_strategy.execute()
    print(f"Deleted '{model_id}' from cache.")


def delete_cached_model_interactive():
    """Show cached models and prompt the user to pick one to delete, repeating until cancelled."""
    while True:
        repos = list_cached_models()
        if not repos:
            break
        choice = input("Enter the # of the model to delete (or 0 to cancel): ").strip()
        if not choice.isdigit() or int(choice) == 0:
            print("Cancelled.")
            break
        if 1 <= int(choice) <= len(repos):
            delete_cached_model(repos[int(choice) - 1].repo_id)
        else:
            print(f"Invalid choice, pick a number between 1 and {len(repos)}.")


if __name__ == "__main__":
    # hf_login()  # Uncomment if you need interactive authentication

    # ── Download ──────────────────────────────────────────────────────────────
    # Uncomment or add the models you want to download, then run: python prepare_backends.py
    models = [
        # "mlx-community/Qwen3-VL-8B-Instruct-8bit",
        # "mlx-community/gemma-3-12b-it-qat-4bit",
        # "mlx-community/Qwen3-VL-4B-Instruct-8bit",
        "mlx-community/gemma-3-4b-it-qat-4bit",
    ]
    for model in models:
        download_model(model)

    # ── Cache management ──────────────────────────────────────────────────────
    list_cached_models()              # show all cached models + sizes
    # delete_cached_model_interactive() # interactive picker to delete one
