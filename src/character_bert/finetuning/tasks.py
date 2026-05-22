from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from transformers import BasicTokenizer


@dataclass(frozen=True)
class ClassificationExample:
    id: int
    tokens_a: list[str]
    tokens_b: list[str] | None
    label: str


@dataclass(frozen=True)
class SequenceLabelingExample:
    id: int
    token_sequence: list[str]
    label_sequence: list[str]


@dataclass(frozen=True)
class FineTuningData:
    task: str
    train_examples: list[ClassificationExample] | list[SequenceLabelingExample]
    validation_examples: list[ClassificationExample] | list[SequenceLabelingExample]
    test_examples: list[ClassificationExample] | list[SequenceLabelingExample]
    labels: list[str]


def load_classification_dataset(
    path: str | Path,
    *,
    do_lower_case: bool,
) -> list[ClassificationExample]:
    tokenizer = BasicTokenizer(do_lower_case=do_lower_case)
    examples = []
    with Path(path).open(encoding="utf-8") as data_file:
        for index, line in enumerate(data_file):
            split_line = line.strip().split()
            if not split_line:
                continue
            label = split_line[0].split("__label__")[-1]
            text = " ".join(split_line[1:])
            examples.append(
                ClassificationExample(
                    id=index,
                    tokens_a=tokenizer.tokenize(text),
                    tokens_b=None,
                    label=label,
                )
            )
    return examples


def load_classification_hf_dataset(
    dataset,
    *,
    text_column: str,
    label_column: str,
    labels: list[str],
    do_lower_case: bool,
) -> list[ClassificationExample]:
    tokenizer = BasicTokenizer(do_lower_case=do_lower_case)
    examples = []
    for index, row in enumerate(dataset):
        label = row[label_column]
        if isinstance(label, int):
            label = labels[label]
        examples.append(
            ClassificationExample(
                id=index,
                tokens_a=tokenizer.tokenize(row[text_column]),
                tokens_b=None,
                label=str(label),
            )
        )
    return examples


def load_sequence_labeling_dataset(
    path: str | Path,
    *,
    do_lower_case: bool,
) -> list[SequenceLabelingExample]:
    examples = []
    tokenizer = BasicTokenizer(do_lower_case=do_lower_case)
    token_sequence: list[str] = []
    label_sequence: list[str] = []

    with Path(path).open(encoding="utf-8") as data_file:
        for line in data_file:
            split_line = line.strip().split()
            if split_line:
                token, label = split_line
                token_sequence.append(token)
                label_sequence.append(label)
            elif token_sequence:
                examples.append(
                    SequenceLabelingExample(
                        id=len(examples),
                        token_sequence=token_sequence,
                        label_sequence=label_sequence,
                    )
                )
                token_sequence = []
                label_sequence = []

    if token_sequence:
        examples.append(
            SequenceLabelingExample(
                id=len(examples),
                token_sequence=token_sequence,
                label_sequence=label_sequence,
            )
        )

    return retokenize_sequence_labeling_examples(examples, tokenizer.tokenize)


def load_sequence_labeling_hf_dataset(
    dataset,
    *,
    tokens_column: str,
    labels_column: str,
    labels: list[str],
) -> list[SequenceLabelingExample]:
    examples = []
    for index, row in enumerate(dataset):
        label_sequence = [
            labels[label] if isinstance(label, int) else str(label)
            for label in row[labels_column]
        ]
        examples.append(
            SequenceLabelingExample(
                id=index,
                token_sequence=list(row[tokens_column]),
                label_sequence=label_sequence,
            )
        )
    return examples


def retokenize_classification_examples(
    examples: list[ClassificationExample],
    tokenize,
) -> list[ClassificationExample]:
    retokenized = []
    for example in examples:
        tokens_a = [piece for token in example.tokens_a for piece in tokenize(token)]
        tokens_b = None
        if example.tokens_b is not None:
            tokens_b = [piece for token in example.tokens_b for piece in tokenize(token)]
        retokenized.append(
            ClassificationExample(
                id=example.id,
                tokens_a=tokens_a or [""],
                tokens_b=tokens_b or None,
                label=example.label,
            )
        )
    return retokenized


def retokenize_sequence_labeling_examples(
    examples: list[SequenceLabelingExample],
    tokenize,
) -> list[SequenceLabelingExample]:
    retokenized = []
    for example in examples:
        tokens: list[str] = []
        labels: list[str] = []
        for token, label in zip(example.token_sequence, example.label_sequence, strict=True):
            token_pieces = tokenize(token)
            if not token_pieces:
                continue
            tokens.extend(token_pieces)
            labels.extend(_expand_label(label, len(token_pieces)))
        retokenized.append(
            SequenceLabelingExample(
                id=example.id,
                token_sequence=tokens or [""],
                label_sequence=labels or ["O"],
            )
        )
    return retokenized


def _expand_label(label: str, count: int) -> list[str]:
    if label == "O":
        return [label] * count

    label_position = label[:2]
    label_type = label.split("-")[-1]
    if label_position == "B-":
        return [label] + ["I-" + label_type] * (count - 1)
    return [label] * count
