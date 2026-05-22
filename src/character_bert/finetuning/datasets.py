from __future__ import annotations

from dataclasses import dataclass

from datasets import Dataset, DatasetDict, load_dataset

from character_bert.finetuning.tasks import (
    FineTuningData,
    load_classification_hf_dataset,
    load_sequence_labeling_hf_dataset,
)


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    path: str
    config: str | None
    task: str
    text_column: str | None = None
    text_pair_column: str | None = None
    label_column: str = "label"
    tokens_column: str | None = None
    labels_column: str | None = None
    train_split: str = "train"
    validation_split: str = "validation"
    test_split: str = "test"
    submission_file: str | None = None
    extra_test_splits: dict[str, str] | None = None


GLUE_TASKS = {
    "cola": DatasetSpec(
        "cola", "nyu-mll/glue", "cola", "classification", "sentence", submission_file="CoLA.tsv"
    ),
    "sst2": DatasetSpec(
        "sst2", "nyu-mll/glue", "sst2", "classification", "sentence", submission_file="SST-2.tsv"
    ),
    "mrpc": DatasetSpec(
        "mrpc",
        "nyu-mll/glue",
        "mrpc",
        "classification",
        "sentence1",
        "sentence2",
        submission_file="MRPC.tsv",
    ),
    "stsb": DatasetSpec(
        "stsb",
        "nyu-mll/glue",
        "stsb",
        "regression",
        "sentence1",
        "sentence2",
        submission_file="STS-B.tsv",
    ),
    "qqp": DatasetSpec(
        "qqp",
        "nyu-mll/glue",
        "qqp",
        "classification",
        "question1",
        "question2",
        submission_file="QQP.tsv",
    ),
    "mnli": DatasetSpec(
        "mnli",
        "nyu-mll/glue",
        "mnli",
        "classification",
        "premise",
        "hypothesis",
        validation_split="validation_matched",
        test_split="test_matched",
        submission_file="MNLI-m.tsv",
        extra_test_splits={"test_mismatched": "MNLI-mm.tsv"},
    ),
    "qnli": DatasetSpec(
        "qnli",
        "nyu-mll/glue",
        "qnli",
        "classification",
        "question",
        "sentence",
        submission_file="QNLI.tsv",
    ),
    "rte": DatasetSpec(
        "rte",
        "nyu-mll/glue",
        "rte",
        "classification",
        "sentence1",
        "sentence2",
        submission_file="RTE.tsv",
    ),
    "wnli": DatasetSpec(
        "wnli",
        "nyu-mll/glue",
        "wnli",
        "classification",
        "sentence1",
        "sentence2",
        submission_file="WNLI.tsv",
    ),
}

DATASET_PRESETS = {
    **GLUE_TASKS,
    "conll2003": DatasetSpec(
        "conll2003",
        "tomaarsen/conll2003",
        None,
        "sequence_labeling",
        tokens_column="tokens",
        labels_column="ner_tags",
    ),
}


def load_finetuning_dataset(
    dataset_name: str,
    *,
    do_lower_case: bool,
    max_train_examples: int | None = None,
    max_validation_examples: int | None = None,
    max_test_examples: int | None = None,
) -> FineTuningData:
    spec = DATASET_PRESETS[dataset_name]
    dataset = load_dataset_splits(spec)

    train_split = _limit(dataset[spec.train_split], max_train_examples)
    validation_split = _limit(dataset[spec.validation_split], max_validation_examples)
    test_split = _limit(dataset[spec.test_split], max_test_examples)

    if spec.task in {"classification", "regression"}:
        labels = (
            None
            if spec.task == "regression"
            else _class_labels(train_split, spec.label_column)
        )
        return FineTuningData(
            name=dataset_name,
            task=spec.task,
            train_examples=_sequence_examples(train_split, spec, labels, do_lower_case),
            validation_examples=_sequence_examples(validation_split, spec, labels, do_lower_case),
            test_examples=_sequence_examples(test_split, spec, labels, do_lower_case),
            labels=labels,
        )

    labels = _sequence_labels(train_split, spec.labels_column)
    return FineTuningData(
        name=dataset_name,
        task=spec.task,
        train_examples=load_sequence_labeling_hf_dataset(
            train_split,
            tokens_column=spec.tokens_column,
            labels_column=spec.labels_column,
            labels=labels,
        ),
        validation_examples=load_sequence_labeling_hf_dataset(
            validation_split,
            tokens_column=spec.tokens_column,
            labels_column=spec.labels_column,
            labels=labels,
        ),
        test_examples=load_sequence_labeling_hf_dataset(
            test_split,
            tokens_column=spec.tokens_column,
            labels_column=spec.labels_column,
            labels=labels,
        ),
        labels=labels,
    )


def load_dataset_splits(spec: DatasetSpec) -> DatasetDict:
    if spec.config is None:
        return load_dataset(spec.path)
    return load_dataset(spec.path, spec.config)


def load_test_examples(
    spec: DatasetSpec,
    split_name: str,
    *,
    labels: list[str] | None,
    do_lower_case: bool,
    max_examples: int | None,
):
    dataset = _limit(load_dataset_splits(spec)[split_name], max_examples)
    return _sequence_examples(dataset, spec, labels, do_lower_case)


def glue_submission_splits(spec: DatasetSpec) -> dict[str, str]:
    if spec.submission_file is None:
        return {}
    splits = {spec.test_split: spec.submission_file}
    if spec.extra_test_splits:
        splits.update(spec.extra_test_splits)
    return splits


def _sequence_examples(
    dataset: Dataset,
    spec: DatasetSpec,
    labels: list[str] | None,
    do_lower_case: bool,
):
    return load_classification_hf_dataset(
        dataset,
        text_column=spec.text_column,
        text_pair_column=spec.text_pair_column,
        label_column=spec.label_column,
        labels=labels,
        do_lower_case=do_lower_case,
        regression=spec.task == "regression",
    )


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
