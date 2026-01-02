"""
Download GLM-Z1-9B-0414 from HuggingFace

Three methods provided - use whichever works for your setup.
"""

# ==============================================================================
# METHOD 1: huggingface_hub (Recommended)
# ==============================================================================
# First install: pip install huggingface_hub

def download_with_hub():
    from huggingface_hub import snapshot_download

    # Download entire repo
    local_dir = snapshot_download(
        repo_id="zai-org/GLM-Z1-9B-0414",
        local_dir="./models/GLM-Z1-9B-0414",
        local_dir_use_symlinks=False,  # Windows compatibility
        resume_download=True,  # Resume if interrupted
    )
    print(f"Downloaded to: {local_dir}")
    return local_dir


# ==============================================================================
# METHOD 2: Command line (simplest)
# ==============================================================================
# Run in terminal:
#
#   pip install huggingface_hub
#   huggingface-cli download zai-org/GLM-Z1-9B-0414 --local-dir ./models/GLM-Z1-9B-0414
#
# Or with authentication (if needed):
#   huggingface-cli login
#   huggingface-cli download zai-org/GLM-Z1-9B-0414 --local-dir ./models/GLM-Z1-9B-0414


# ==============================================================================
# METHOD 3: Git LFS (if above fails)
# ==============================================================================
# First install Git LFS: https://git-lfs.github.com/
#
#   git lfs install
#   git clone https://huggingface.co/zai-org/GLM-Z1-9B-0414 ./models/GLM-Z1-9B-0414


# ==============================================================================
# METHOD 4: Transformers auto-download (for direct use)
# ==============================================================================
def load_directly():
    """Load model directly - will auto-download to cache"""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_id = "zai-org/GLM-Z1-9B-0414"

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        device_map="auto",  # Automatic GPU placement
        torch_dtype="auto",  # Use model's native dtype
    )

    return model, tokenizer


if __name__ == "__main__":
    import sys

    print("GLM-Z1-9B-0414 Downloader")
    print("=" * 50)
    print("\nOptions:")
    print("1. Run this script (uses huggingface_hub)")
    print("2. CLI: huggingface-cli download zai-org/GLM-Z1-9B-0414 --local-dir ./models/GLM-Z1-9B-0414")
    print("3. Git: git clone https://huggingface.co/zai-org/GLM-Z1-9B-0414")
    print()

    if len(sys.argv) > 1 and sys.argv[1] == "--download":
        print("Downloading with huggingface_hub...")
        try:
            download_with_hub()
        except ImportError:
            print("\nhuggingface_hub not installed. Run:")
            print("  pip install huggingface_hub")
    else:
        print("Run with --download to start download")
        print("Or use the CLI method above")
