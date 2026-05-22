from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import TensorDataset

from character_bert.finetuning.tasks import ClassificationExample, SequenceLabelingExample
from character_bert.modeling import CharacterIndexer


@dataclass(frozen=True)
class InputFeatures:
    input_ids: list[int] | torch.Tensor
    attention_mask: list[int]
    token_type_ids: list[int]
    label_ids: int | float | list[int]


def classification_features(
    examples: list[ClassificationExample],
    *,
    tokenizer,
    labels: list[str] | None,
    max_seq_length: int,
    is_character_model: bool,
    pad_token_id: int = 0,
    regression: bool = False,
) -> list[InputFeatures]:
    label_map = {} if labels is None else {label: index for index, label in enumerate(labels)}
    features = []
    for example in examples:
        tokens_a = list(example.tokens_a)
        tokens_b = list(example.tokens_b) if example.tokens_b else None

        if tokens_b:
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        elif len(tokens_a) > max_seq_length - 2:
            tokens_a = tokens_a[: max_seq_length - 2]

        tokens = ["[CLS]", *tokens_a, "[SEP]"]
        token_type_ids = [0] * len(tokens)
        if tokens_b:
            tokens += [*tokens_b, "[SEP]"]
            token_type_ids += [1] * (len(tokens_b) + 1)

        input_ids = _tokens_to_model_ids(tokens, tokenizer, is_character_model, max_seq_length)
        attention_mask = [1] * len(tokens)
        padding_length = max_seq_length - len(attention_mask)
        if not is_character_model:
            input_ids += [pad_token_id] * padding_length
        attention_mask += [0] * padding_length
        token_type_ids += [0] * padding_length

        features.append(
            InputFeatures(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                label_ids=_classification_label_id(example.label, label_map, regression),
            )
        )
    return features


def sequence_labeling_features(
    examples: list[SequenceLabelingExample],
    *,
    tokenizer,
    labels: list[str],
    max_seq_length: int,
    is_character_model: bool,
    pad_token_label_id: int,
    pad_token_id: int = 0,
) -> list[InputFeatures]:
    label_map = {label: index for index, label in enumerate(labels)}
    features = []
    for example in examples:
        tokens = list(example.token_sequence)
        label_ids = [
            pad_token_label_id if token.startswith("##") else label_map[label]
            for token, label in zip(tokens, example.label_sequence, strict=True)
        ]

        if len(tokens) > max_seq_length - 2:
            tokens = tokens[: max_seq_length - 2]
            label_ids = label_ids[: max_seq_length - 2]

        tokens = ["[CLS]", *tokens, "[SEP]"]
        label_ids = [pad_token_label_id, *label_ids, pad_token_label_id]
        token_type_ids = [0] * len(tokens)

        input_ids = _tokens_to_model_ids(tokens, tokenizer, is_character_model, max_seq_length)
        attention_mask = [1] * len(tokens)
        padding_length = max_seq_length - len(attention_mask)
        if not is_character_model:
            input_ids += [pad_token_id] * padding_length
        attention_mask += [0] * padding_length
        token_type_ids += [0] * padding_length
        label_ids += [pad_token_label_id] * padding_length

        features.append(
            InputFeatures(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                label_ids=label_ids,
            )
        )
    return features


def features_to_dataset(
    features: list[InputFeatures],
    *,
    task: str,
    is_character_model: bool,
) -> TensorDataset:
    if is_character_model:
        all_input_ids = torch.tensor(
            [feature.input_ids.tolist() for feature in features],
            dtype=torch.long,
        )
    else:
        all_input_ids = torch.tensor([feature.input_ids for feature in features], dtype=torch.long)

    all_attention_mask = torch.tensor(
        [feature.attention_mask for feature in features],
        dtype=torch.long,
    )
    all_token_type_ids = torch.tensor(
        [feature.token_type_ids for feature in features],
        dtype=torch.long,
    )
    if task == "regression":
        all_label_ids = torch.tensor([feature.label_ids for feature in features], dtype=torch.float)
    elif task == "sequence_labeling":
        all_label_ids = torch.tensor(
            [feature.label_ids for feature in features],
            dtype=torch.long,
        )
    else:
        all_label_ids = torch.tensor([feature.label_ids for feature in features], dtype=torch.long)

    return TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_label_ids)


def _classification_label_id(
    label: str | float | None,
    label_map: dict[str, int],
    regression: bool,
) -> int | float:
    if label is None:
        return 0.0 if regression else 0
    if regression:
        return float(label)
    return label_map[str(label)]


def _tokens_to_model_ids(
    tokens: list[str],
    tokenizer,
    is_character_model: bool,
    max_seq_length: int,
) -> list[int] | torch.Tensor:
    if is_character_model:
        indexer = tokenizer if isinstance(tokenizer, CharacterIndexer) else CharacterIndexer()
        return indexer.as_padded_tensor([tokens], max_length=max_seq_length)[0]
    return tokenizer.convert_tokens_to_ids(tokens)


def _truncate_seq_pair(tokens_a: list[str], tokens_b: list[str], max_length: int) -> None:
    while len(tokens_a) + len(tokens_b) > max_length:
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()
