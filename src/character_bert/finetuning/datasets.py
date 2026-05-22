from __future__ import annotations

from collections.abc import Mapping

from datasets import Dataset, DatasetDict, load_dataset

from character_bert.finetuning.tasks import (
    FineTuningData,
    load_classification_hf_dataset,
    load_sequence_labeling_hf_dataset,
)

DATASET_PRESETS = {
    "sst2": {
        "path": "nyu-mll/glue",
        "name": "sst2",
        "task": "classification",
        "text_column": "sentence",
        "label_column": "label",
        "train_split": "train",
        "validation_split": "validation",
        "test_split": "validation",
    },
    "conll2003": {
        "path": "tomaarsen/conll2003",
        "name": None,
        "task": "sequence_labeling",
        "tokens_column": "tokens",
        "labels_column": "ner_tags",
        "train_split": "train",
        "validation_split": "validation",
        "test_split": "test",
    },
}


def load_finetuning_dataset(
    dataset_name: str,
    *,
    do_lower_case: bool,
    max_train_examples: int | None = None,
    max_validation_examples: int | None = None,
    max_test_examples: int | None = None,
) -> FineTuningData:
    preset = DATASET_PRESETS[dataset_name]
    dataset = _load_dataset(preset)
    task = preset["task"]

    train_split = _limit(dataset[preset["train_split"]], max_train_examples)
    validation_split = _limit(dataset[preset["validation_split"]], max_validation_examples)
    test_split = _limit(dataset[preset["test_split"]], max_test_examples)

    if task == "classification":
        labels = _class_labels(train_split, preset["label_column"])
        return FineTuningData(
            task=task,
            train_examples=load_classification_hf_dataset(
                train_split,
                text_column=preset["text_column"],
                label_column=preset["label_column"],
                labels=labels,
                do_lower_case=do_lower_case,
            ),
            validation_examples=load_classification_hf_dataset(
                validation_split,
                text_column=preset["text_column"],
                label_column=preset["label_column"],
                labels=labels,
                do_lower_case=do_lower_case,
            ),
            test_examples=load_classification_hf_dataset(
                test_split,
                text_column=preset["text_column"],
                label_column=preset["label_column"],
                labels=labels,
                do_lower_case=do_lower_case,
            ),
            labels=labels,
        )

    labels = _sequence_labels(train_split, preset["labels_column"])
    return FineTuningData(
        task=task,
        train_examples=load_sequence_labeling_hf_dataset(
            train_split,
            tokens_column=preset["tokens_column"],
            labels_column=preset["labels_column"],
            labels=labels,
        ),
        validation_examples=load_sequence_labeling_hf_dataset(
            validation_split,
            tokens_column=preset["tokens_column"],
            labels_column=preset["labels_column"],
            labels=labels,
        ),
        test_examples=load_sequence_labeling_hf_dataset(
            test_split,
            tokens_column=preset["tokens_column"],
            labels_column=preset["labels_column"],
            labels=labels,
        ),
        labels=labels,
    )


def _load_dataset(preset: Mapping[str, str | None]) -> DatasetDict:
    if preset["name"] is None:
        return load_dataset(preset["path"])
    return load_dataset(preset["path"], preset["name"])


def _limit(dataset: Dataset, max_examples: int | None) -> Dataset:
    if max_examples is None:
        return dataset
    return dataset.select(range(min(max_examples, len(dataset))))


def _class_labels(dataset: Dataset, label_column: str) -> list[str]:
    feature = dataset.features[label_column]
    if hasattr(feature, "names"):
        return list(feature.names)
    return sorted({str(row[label_column]) for row in dataset})


def _sequence_labels(dataset: Dataset, labels_column: str) -> list[str]:
    feature = dataset.features[labels_column].feature
    if hasattr(feature, "names"):
        return list(feature.names)
    return sorted({str(label) for row in dataset for label in row[labels_column]})
