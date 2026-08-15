import modal

app = modal.App("qwen38-27b-download")

MODEL_REPO = "cyankiwi/Qwen3.8-27B-AWQ-INT4"
VOLUME_NAME = "qwen38-27b-awq"
MODELS_DIR = "/models"

image = modal.Image.debian_slim(python_version="3.12").pip_install(
    "huggingface_hub[hf_transfer]",
).env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})

volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)


@app.function(
    image=image,
    cpu=8,
    memory=16384,
    timeout=60 * 60,
    volumes={MODELS_DIR: volume},
)
def download():
    from pathlib import Path
    from huggingface_hub import snapshot_download

    target = Path(MODELS_DIR) / MODEL_REPO.split("/")[-1]
    marker = target / ".download_complete"

    if marker.exists():
        print(f"Already downloaded at {target}, skipping.")
        return

    # Public apache-2.0 repo — no HF token needed.
    print(f"Downloading {MODEL_REPO} -> {target}")
    snapshot_download(repo_id=MODEL_REPO, local_dir=str(target), max_workers=8)
    marker.touch()
    volume.commit()
    print("Download complete and volume committed.")


@app.local_entrypoint()
def main():
    download.remote()
