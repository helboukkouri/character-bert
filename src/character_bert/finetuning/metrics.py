from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def classification_metrics(labels: Sequence[int], predictions: Sequence[int]) -> dict[str, float]:
    labels_array = np.asarray(labels)
    predictions_array = np.asarray(predictions)
    if labels_array.shape != predictions_array.shape:
        raise ValueError("labels and predictions must have the same shape")
    if labels_array.size == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "accuracy": 0.0}

    accuracy = float(np.mean(labels_array == predictions_array))
    return {
        "precision": accuracy,
        "recall": accuracy,
        "f1": accuracy,
        "accuracy": accuracy,
    }


def sequence_labeling_metrics(
    labels: Sequence[Sequence[str]],
    predictions: Sequence[Sequence[str]],
) -> dict[str, float]:
    return {
        "precision": precision_score(labels, predictions),
        "recall": recall_score(labels, predictions),
        "f1": f1_score(labels, predictions),
        "accuracy": sequence_accuracy(labels, predictions),
    }


def get_entities(sequence: Sequence[str] | Sequence[Sequence[str]]) -> list[tuple[str, int, int]]:
    if any(isinstance(label, list) for label in sequence):
        flattened = []
        for sentence in sequence:
            flattened.extend(sentence)
            flattened.append("O")
        sequence = flattened

    previous_tag = "O"
    previous_type = ""
    begin_offset = 0
    chunks = []
    for index, chunk in enumerate([*sequence, "O"]):
        tag = chunk[0]
        chunk_type = chunk.split("-")[-1]

        if _ends_chunk(previous_tag, tag, previous_type, chunk_type):
            chunks.append((previous_type, begin_offset, index - 1))
        if _starts_chunk(previous_tag, tag, previous_type, chunk_type):
            begin_offset = index
        previous_tag = tag
        previous_type = chunk_type

    return chunks


def precision_score(
    labels: Sequence[Sequence[str]],
    predictions: Sequence[Sequence[str]],
) -> float:
    true_entities = set(get_entities(labels))
    predicted_entities = set(get_entities(predictions))
    if not predicted_entities:
        return 0.0
    return len(true_entities & predicted_entities) / len(predicted_entities)


def recall_score(
    labels: Sequence[Sequence[str]],
    predictions: Sequence[Sequence[str]],
) -> float:
    true_entities = set(get_entities(labels))
    predicted_entities = set(get_entities(predictions))
    if not true_entities:
        return 0.0
    return len(true_entities & predicted_entities) / len(true_entities)


def f1_score(
    labels: Sequence[Sequence[str]],
    predictions: Sequence[Sequence[str]],
) -> float:
    precision = precision_score(labels, predictions)
    recall = recall_score(labels, predictions)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def sequence_accuracy(
    labels: Sequence[Sequence[str]],
    predictions: Sequence[Sequence[str]],
) -> float:
    flat_labels = [label for sequence in labels for label in sequence]
    flat_predictions = [prediction for sequence in predictions for prediction in sequence]
    if len(flat_labels) != len(flat_predictions):
        raise ValueError("labels and predictions must contain the same number of tokens")
    if not flat_labels:
        return 0.0
    correct = sum(label == prediction for label, prediction in zip(flat_labels, flat_predictions))
    return correct / len(flat_labels)


def _ends_chunk(previous_tag: str, tag: str, previous_type: str, chunk_type: str) -> bool:
    if previous_tag in {"E", "S"}:
        return True
    if previous_tag in {"B", "I"} and tag in {"B", "S", "O"}:
        return True
    return previous_tag not in {"O", "."} and previous_type != chunk_type


def _starts_chunk(previous_tag: str, tag: str, previous_type: str, chunk_type: str) -> bool:
    if tag in {"B", "S"}:
        return True
    if previous_tag in {"E", "S", "O"} and tag in {"E", "I"}:
        return True
    return tag not in {"O", "."} and previous_type != chunk_type
