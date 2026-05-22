# CharacterBERT

This branch is a refresh of the original COLING 2020 CharacterBERT repository.
The historical code has been moved to [`achived/`](./achived/) so we can rebuild the
package around a modern Python layout while keeping the old implementation close at hand.

## Goals

- Keep the CharacterBERT model code importable as a normal Python package.
- Support encoder-only checkpoints and Hugging Face Hub checkpoints with MLM/NSP heads.
- Add clean entry points for future fine-tuning and pretraining work.
- Use `pyproject.toml`, `uv`, and recent Python/package versions.

## Layout

```text
src/character_bert/
  modeling/       CharacterCNN, CharacterBERT encoder, and pretraining heads
  data/           token/character indexing utilities
  finetuning/     fine-tuning entry points and task adapters
  pretraining/    pretraining entry points and data builders
  training/       shared training utilities
tests/            regression and architecture tests for the refreshed package
achived/          previous repository state, kept for reference
```

## Setup

```bash
uv venv --python 3.12
uv sync --extra dev --extra finetuning --extra pretraining --extra legacy-downloads
```

For a minimal runtime install:

```bash
uv sync
```

## Checkpoints

The original Google Drive encoder checkpoints are still supported, and the Hugging Face Hub
checkpoints add MLM/NSP heads:

- `helboukkouri/character-bert`
- `helboukkouri/character-bert-medical`

Download them with:

```bash
uv run character-bert-download helboukkouri/character-bert --output-dir pretrained-models/hf_character_bert
uv run character-bert-download helboukkouri/character-bert-medical --output-dir pretrained-models/hf_character_bert_medical
```

## Tests

Fast architecture tests do not require checkpoints:

```bash
uv run pytest
```

Checkpoint-backed tests skip automatically when the corresponding model files are not present.

## Smoke Test

```bash
uv run character-bert-smoke-test
```
