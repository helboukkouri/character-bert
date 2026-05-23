# CharacterBERT

[paper]: https://aclanthology.org/2020.coling-main.609/

This is the refreshed code repository for the paper
"[CharacterBERT: Reconciling ELMo and BERT for Word-Level Open-Vocabulary
Representations From Characters][paper]" from COLING 2020.

The original research code has been reorganized into a modern Python package while
preserving CharacterBERT's modeling behavior and checkpoint compatibility. The
goal of this branch is to make the model easy to import, test, fine-tune, and
extend for future pretraining work.

## Paper Summary

### TL;DR

`CharacterBERT` is a variant of [BERT](https://arxiv.org/abs/1810.04805) that
produces contextual representations at the word level.

Instead of looking up predefined WordPiece embeddings, CharacterBERT attends to
the characters of each input token and dynamically builds token representations
with a CharacterCNN module inspired by [ELMo](https://arxiv.org/abs/1802.05365).
That means the model can represent arbitrary input tokens, including words that
would otherwise be split into several wordpieces or treated poorly by a fixed
vocabulary.

### Motivations

CharacterBERT has two main motivations:

- Domain adaptation often reuses the original general-domain BERT WordPiece
  vocabulary even when the final model is targeted at a specialized domain such
  as medicine, law, or science. Training a new BERT vocabulary and model from
  scratch for each domain is expensive, and reusing a mismatched vocabulary is
  not always ideal.
- WordPiece tokenization is powerful, but it makes many downstream workflows
  less direct. Sequence labeling and word-level representation tasks often need
  extra decisions about how to combine or select subword representations.

CharacterBERT keeps BERT's contextual Transformer encoder while replacing the
WordPiece embedding matrix with a character-based token encoder. The result is a
word-level, open-vocabulary model that can be re-adapted to new domains without
changing vocabularies and is more robust to typos and misspellings.

## Repository Layout

```text
src/character_bert/
  modeling/       reusable library: config, CharacterCNN, encoder, heads, indexer
  finetuning/     fine-tuning app: CLI, task code, app-local data/utils
  pretraining/    pretraining app skeleton for future work
tests/            architecture, checkpoint, and fine-tuning regression tests
Makefile          common GLUE fine-tuning and submission commands
```

The `modeling` package is the reusable library layer:

```python
from character_bert.modeling import CharacterBertModel, CharacterIndexer
```

Fine-tuning and pretraining are organized as app namespaces that import the
modeling library. Their data processing and workflow helpers should stay local to
each app unless a helper is genuinely shared.

Runtime folders such as `pretrained-models/` and `results/` are intentionally not
tracked. Download, inference, and training commands create them when needed.

## Setup

This repo uses `uv` and a `pyproject.toml`-based package layout.

```bash
uv venv --python 3.14
uv sync --extra dev --extra finetuning --extra pretraining --extra legacy-downloads
```

The package supports Python 3.12 and newer. Development currently targets the
latest stable Python 3.14 line so the refreshed repo stays ahead of near-term
deprecations.

For a minimal runtime install:

```bash
uv sync
```

## Checkpoints

The original encoder checkpoints are still supported. The Hugging Face Hub
checkpoints add MLM/NSP heads and are the preferred source for pretrained models:

- `helboukkouri/character-bert`
- `helboukkouri/character-bert-medical`

Download checkpoints with:

```bash
uv run character-bert-download \
  helboukkouri/character-bert \
  --output-dir pretrained-models/hf_character_bert

uv run character-bert-download \
  helboukkouri/character-bert-medical \
  --output-dir pretrained-models/hf_character_bert_medical
```

The legacy model aliases used by the original code are still recognized where
possible:

```text
general_character_bert
medical_character_bert
bert-base-uncased
```

## Using CharacterBERT

CharacterBERT accepts token-level character ids rather than WordPiece ids:

```python
from transformers import BasicTokenizer

from character_bert.modeling import CharacterBertModel, CharacterIndexer

tokenizer = BasicTokenizer(do_lower_case=True)
tokens = ["[CLS]", *tokenizer.tokenize("CharacterBERT handles new words."), "[SEP]"]

indexer = CharacterIndexer()
input_ids = indexer.as_padded_tensor([tokens])

model = CharacterBertModel.from_pretrained("pretrained-models/general_character_bert")
outputs = model(input_ids, return_dict=True)

print(outputs.last_hidden_state.shape)
```

Because the encoder follows the BERT module structure, it can also be used inside
standard BERT task heads:

```python
from transformers import BertForSequenceClassification

from character_bert.modeling import CharacterBertModel, CharacterBertConfig

config = CharacterBertConfig.from_pretrained(
    "pretrained-models/general_character_bert",
    num_labels=2,
)
model = BertForSequenceClassification(config)
model.bert = CharacterBertModel.from_pretrained(
    "pretrained-models/general_character_bert",
    config=config,
)
```

## Tests

Fast architecture tests do not require checkpoints:

```bash
uv run pytest
```

Checkpoint-backed tests skip automatically when the corresponding model files are
not present. For development, run:

```bash
uv run --extra dev --extra finetuning --frozen ruff check src tests
uv run --extra dev --extra finetuning --frozen python -m unittest discover -v
```

The test suite covers the CharacterBERT architecture, checkpoint loading,
MLM/NSP behavior for Hub checkpoints, fine-tuning features, and GLUE submission
label formatting.

## Fine-Tuning

The fine-tuning app uses `datasets` presets by default. GLUE tasks are
first-class because they cover the main sentence and sentence-pair workflows and
can emit leaderboard-style test TSV files.

```bash
uv run --extra finetuning character-bert-finetune \
  --dataset sst2 \
  --embedding bert-base-uncased \
  --num-train-epochs 3 \
  --do-train \
  --do-predict \
  --write-glue-submission
```

Supported GLUE presets:

```text
cola, sst2, mrpc, stsb, qqp, mnli, qnli, rte, wnli
```

`--write-glue-submission` writes files under `results/.../glue_submission/`.
For MNLI it writes `MNLI-m.tsv`, `MNLI-mm.tsv`, and the diagnostic `AX.tsv`.
Submission labels follow the GLUE upload format: CoLA, SST-2, MRPC, QQP, and
WNLI use `0`/`1`; MNLI, AX, QNLI, and RTE use entailment-style labels; STS-B
uses clipped floating-point scores.

Fine-tuning writes TensorBoard logs to `results/.../tensorboard/` by default:

```bash
uv run --extra finetuning tensorboard --logdir results
```

## GLUE Commands

The Makefile wraps common GLUE runs for BERT and CharacterBERT.

Run one task:

```bash
make finetune-bert-sst2
make finetune-characterbert-sst2
```

Run every GLUE task for one model and produce one submission-ready zip:

```bash
make glue-submission-bert
make glue-submission-characterbert
```

The output zips are written to:

```text
results/submissions/bert-base-uncased-glue.zip
results/submissions/general-character-bert-glue.zip
```

Run both models:

```bash
make glue-submissions
```

You can override training settings without editing the Makefile:

```bash
make glue-submission-characterbert \
  EPOCHS=3 \
  TRAIN_BATCH_SIZE=32 \
  EVAL_BATCH_SIZE=32 \
  LEARNING_RATE=2e-5 \
  MAX_SEQ_LENGTH=128
```

For a tiny end-to-end check that still predicts the full GLUE test splits and
produces upload-shaped zip files:

```bash
make smoke-glue-submission-bert
make smoke-glue-submission-characterbert
make smoke-glue-submissions
```

## Pretraining

The pretraining namespace is present as the place for modernized pretraining
work, but the refreshed implementation is not complete yet. The historical
pretraining code was originally released separately at:

```text
https://github.com/helboukkouri/character-bert-pretraining
```

## References

Please cite our paper if you use CharacterBERT in your work:

```bibtex
@inproceedings{el-boukkouri-etal-2020-characterbert,
    title = "{C}haracter{BERT}: Reconciling {ELM}o and {BERT} for Word-Level Open-Vocabulary Representations From Characters",
    author = "El Boukkouri, Hicham  and
      Ferret, Olivier  and
      Lavergne, Thomas  and
      Noji, Hiroshi  and
      Zweigenbaum, Pierre  and
      Tsujii, Jun{'}ichi",
    booktitle = "Proceedings of the 28th International Conference on Computational Linguistics",
    month = dec,
    year = "2020",
    address = "Barcelona, Spain (Online)",
    publisher = "International Committee on Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.coling-main.609",
    doi = "10.18653/v1/2020.coling-main.609",
    pages = "6903--6915",
    abstract = "Due to the compelling improvements brought by BERT, many recent representation models adopted the Transformer architecture as their main building block, consequently inheriting the wordpiece tokenization system despite it not being intrinsically linked to the notion of Transformers. While this system is thought to achieve a good balance between the flexibility of characters and the efficiency of full words, using predefined wordpiece vocabularies from the general domain is not always suitable, especially when building models for specialized domains (e.g., the medical domain). Moreover, adopting a wordpiece tokenization shifts the focus from the word level to the subword level, making the models conceptually more complex and arguably less convenient in practice. For these reasons, we propose CharacterBERT, a new variant of BERT that drops the wordpiece system altogether and uses a Character-CNN module instead to represent entire words by consulting their characters. We show that this new model improves the performance of BERT on a variety of medical domain tasks while at the same time producing robust, word-level, and open-vocabulary representations.",
}
```
