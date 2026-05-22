import logging
import os
import tarfile
from pathlib import Path

import requests
from huggingface_hub import snapshot_download

HUB_CHECKPOINTS = {
    "hf_character_bert": "helboukkouri/character-bert",
    "hf_character_bert_medical": "helboukkouri/character-bert-medical",
    "general": "helboukkouri/character-bert",
    "medical": "helboukkouri/character-bert-medical",
    "helboukkouri/character-bert": "helboukkouri/character-bert",
    "helboukkouri/character-bert-medical": "helboukkouri/character-bert-medical",
}

DRIVE_CHECKPOINTS = {
    "general_character_bert": (
        "https://drive.google.com/uc?id=11-kSfIwSWrPno6A4VuNFWuQVYD8Bg_aZ"
    ),
    "medical_character_bert": "https://drive.google.com/uc?id=1LEnQHAqP9GxDYa0I3UrZ9YV2QhHKOh2m",
    "general_bert": "https://drive.google.com/uc?id=1fwgKG2BziBZr7aQMK58zkbpI0OxWRsof",
    "medical_bert": "https://drive.google.com/uc?id=1GmnXJFntcEfrRY4pVZpJpg7FH62m47HS",
}

BERT_BASE_UNCASED_FILES = {
    "pytorch_model.bin": "https://huggingface.co/bert-base-uncased/resolve/main/pytorch_model.bin",
    "vocab.txt": "https://huggingface.co/bert-base-uncased/resolve/main/vocab.txt",
    "config.json": "https://huggingface.co/bert-base-uncased/resolve/main/config.json",
}

LOGGER = logging.getLogger(__name__)


def checkpoint_exists(path: str | Path) -> bool:
    checkpoint_dir = Path(path)
    return all(
        (checkpoint_dir / filename).exists()
        for filename in ("config.json", "pytorch_model.bin")
    )


def download_checkpoint(model: str, output_dir: str | Path | None = None) -> Path:
    destination = (
        Path(output_dir)
        if output_dir is not None
        else Path("pretrained-models") / model.replace("/", "__")
    )
    if checkpoint_exists(destination):
        LOGGER.info("Checkpoint already exists at %s", destination)
        return destination

    destination.parent.mkdir(parents=True, exist_ok=True)

    if model in DRIVE_CHECKPOINTS:
        _download_drive_checkpoint(DRIVE_CHECKPOINTS[model], destination)
    elif model == "bert-base-uncased":
        _download_bert_base_uncased(destination)
    else:
        repo_id = HUB_CHECKPOINTS.get(model, model)
        snapshot_download(
            repo_id=repo_id,
            local_dir=destination,
            local_dir_use_symlinks=False,
        )
    return destination


def read_mlm_vocab(checkpoint_dir: str | Path) -> list[str]:
    vocab_path = Path(checkpoint_dir) / "mlm_vocab.txt"
    with vocab_path.open(encoding="utf-8") as vocab_file:
        return [line.strip() for line in vocab_file if line.strip()]


def _download_bert_base_uncased(destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for filename, url in BERT_BASE_UNCASED_FILES.items():
        output_path = destination / filename
        if output_path.exists():
            continue
        _download_url(url, output_path)


def _download_drive_checkpoint(url: str, destination: Path) -> None:
    try:
        import gdown
    except ImportError as error:
        raise ImportError(
            "Downloading legacy Google Drive checkpoints requires `gdown`. "
            "Install with `uv sync --extra legacy-downloads`."
        ) from error

    destination.mkdir(parents=True, exist_ok=True)
    archive_path = destination.parent / "model.tar.xz"
    output = gdown.download(url, str(archive_path), quiet=False)
    if output is None or not archive_path.exists():
        raise RuntimeError(f"Failed to download archive from {url}")

    with tarfile.open(archive_path, "r:xz") as archive:
        _extract_tar_safely(archive, destination.parent)
    archive_path.unlink()


def _download_url(url: str, destination: Path) -> None:
    with requests.get(url, stream=True, timeout=30) as response:
        response.raise_for_status()
        with destination.open("wb") as output_file:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    output_file.write(chunk)


def _extract_tar_safely(archive: tarfile.TarFile, destination: Path) -> None:
    resolved_destination = destination.resolve()
    for member in archive.getmembers():
        member_path = (resolved_destination / member.name).resolve()
        if os.path.commonpath([resolved_destination, member_path]) != str(resolved_destination):
            raise RuntimeError(f"Archive member escapes destination: {member.name}")
    archive.extractall(path=destination)
