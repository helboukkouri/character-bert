from pathlib import Path

from huggingface_hub import snapshot_download


HUB_CHECKPOINTS = {
    "general": "helboukkouri/character-bert",
    "medical": "helboukkouri/character-bert-medical",
    "helboukkouri/character-bert": "helboukkouri/character-bert",
    "helboukkouri/character-bert-medical": "helboukkouri/character-bert-medical",
}


def download_checkpoint(model: str, output_dir: str | Path | None = None) -> Path:
    repo_id = HUB_CHECKPOINTS.get(model, model)
    destination = (
        Path(output_dir)
        if output_dir is not None
        else Path("pretrained-models") / repo_id.split("/")[-1]
    )
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
