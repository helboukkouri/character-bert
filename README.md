# CharacterBERT

This branch is a refresh of the original COLING 2020 CharacterBERT repository.
The package is organized around a modern Python layout while preserving the
modeling behavior and checkpoint compatibility of the original implementation.

## Goals

- Keep the CharacterBERT model code importable as a normal Python package.
- Support encoder-only checkpoints and Hugging Face Hub checkpoints with MLM/NSP heads.
- Add clean entry points for future fine-tuning and pretraining work.
- Use `pyproject.toml`, `uv`, and recent Python/package versions.

## Layout

```text
src/character_bert/
  modeling/       reusable library: config, CharacterCNN, encoder, heads, indexer
  finetuning/     fine-tuning app: CLI, task code, app-local data/utils
  pretraining/    pretraining app: CLI and app-local data/utils
tests/            regression and architecture tests for the refreshed package
```

The `modeling` package is the reusable part of the project. User code should be able to
import it without pulling in training applications:

```python
from character_bert.modeling import CharacterBertModel, CharacterIndexer
```

Fine-tuning and pretraining are organized as app namespaces that depend on `modeling`.
Their data processing and workflow helpers should stay local to each app unless a helper
is genuinely shared.

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
uv run character-bert-download \
  helboukkouri/character-bert \
  --output-dir pretrained-models/hf_character_bert

uv run character-bert-download \
  helboukkouri/character-bert-medical \
  --output-dir pretrained-models/hf_character_bert_medical
```

## Tests

Fast architecture tests do not require checkpoints:

```bash
uv run pytest
```

Checkpoint-backed tests skip automatically when the corresponding model files are not present.

Runtime folders such as `pretrained-models/` and `results/` are intentionally not tracked.
Download and training commands create them when needed.

## Fine-Tuning

The fine-tuning app supports the original classification and sequence-labeling file
formats:

```bash
uv run character-bert-finetune \
  --task classification \
  --embedding general_character_bert \
  --train-file path/to/train.txt \
  --test-file path/to/test.txt \
  --do-train \
  --do-predict
```

## Smoke Test

```bash
uv run character-bert-smoke-test
```
