from __future__ import annotations

import argparse
import datetime as dt
import logging
from pathlib import Path

import torch
from safetensors.torch import load_file as load_safetensors_file
from torch.nn import CrossEntropyLoss
from transformers import (
    BasicTokenizer,
    BertConfig,
    BertForSequenceClassification,
    BertForTokenClassification,
    BertTokenizer,
)

from character_bert.finetuning.datasets import (
    DATASET_PRESETS,
    GLUE_TASKS,
    glue_submission_splits,
    load_finetuning_dataset,
    load_test_examples,
)
from character_bert.finetuning.engine import TrainingConfig, evaluate, predict, train
from character_bert.finetuning.features import (
    classification_features,
    features_to_dataset,
    sequence_labeling_features,
)
from character_bert.finetuning.tasks import (
    ClassificationExample,
    FineTuningData,
    SequenceLabelingExample,
    load_classification_dataset,
    load_sequence_labeling_dataset,
    retokenize_classification_examples,
    retokenize_sequence_labeling_examples,
)
from character_bert.finetuning.utils.paths import ensure_output_dir
from character_bert.finetuning.utils.seed import set_seed
from character_bert.modeling import CharacterBertConfig, CharacterBertModel, CharacterIndexer


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
    run(args)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune BERT or CharacterBERT.")
    parser.add_argument(
        "--task",
        default=None,
        choices=["classification", "sequence_labeling", "sequence_labelling"],
    )
    parser.add_argument("--embedding", required=True, help="Checkpoint name or path.")
    parser.add_argument(
        "--dataset",
        choices=sorted(DATASET_PRESETS),
        default=None,
        help="Classic datasets-library preset. Currently: sst2, conll2003.",
    )
    parser.add_argument(
        "--train-file",
        default=None,
        help="Training file in the legacy task format.",
    )
    parser.add_argument("--test-file", default=None, help="Test file in the legacy task format.")
    parser.add_argument("--pretrained-dir", default="pretrained-models")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--do-lower-case", action="store_true")
    parser.add_argument("--do-train", action="store_true")
    parser.add_argument("--do-predict", action="store_true")
    parser.add_argument(
        "--write-glue-submission",
        action="store_true",
        help="Write GLUE leaderboard TSV files for the dataset test split.",
    )
    parser.add_argument("--train-batch-size", type=int, default=1)
    parser.add_argument("--eval-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--num-train-epochs", type=int, default=3)
    parser.add_argument("--validation-ratio", type=float, default=0.5)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--adam-epsilon", type=float, default=1e-8)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--max-seq-length", type=int, default=None)
    parser.add_argument("--max-train-examples", type=int, default=None)
    parser.add_argument("--max-validation-examples", type=int, default=None)
    parser.add_argument("--max-test-examples", type=int, default=None)
    parser.add_argument(
        "--tensorboard-log-dir",
        default=None,
        help="TensorBoard log directory. Defaults to output-dir/tensorboard.",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def run(args: argparse.Namespace) -> dict[str, float] | None:
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info("Using device: %s", device)
    data = _load_data(args)
    output_dir = _resolve_output_dir(args, data.task)

    checkpoint_dir = _resolve_checkpoint_dir(args.embedding, args.pretrained_dir)
    is_character_model = "character" in checkpoint_dir.name or "character" in args.embedding
    tokenizer, feature_tokenizer = _load_tokenizers(
        checkpoint_dir=checkpoint_dir,
        pretrained_dir=Path(args.pretrained_dir),
        do_lower_case=args.do_lower_case,
        is_character_model=is_character_model,
    )

    train_examples = _retokenize_examples(
        data.train_examples,
        feature_tokenizer.tokenize,
        data.task,
    )
    validation_examples = _retokenize_examples(
        data.validation_examples,
        feature_tokenizer.tokenize,
        data.task,
    )
    test_examples = _retokenize_examples(
        data.test_examples,
        feature_tokenizer.tokenize,
        data.task,
    )
    labels = data.labels
    max_seq_length = args.max_seq_length or _max_sequence_length(
        data.task,
        train_examples,
        validation_examples,
        test_examples,
    )

    pad_token_label_id = CrossEntropyLoss().ignore_index
    pad_token_id = tokenizer.pad_token_id if tokenizer is not None else 0
    indexer = CharacterIndexer() if is_character_model else tokenizer
    datasets = {
        "train": _build_dataset(
            data.task,
            train_examples,
            indexer,
            labels,
            max_seq_length,
            is_character_model,
            pad_token_id,
            pad_token_label_id,
        ),
        "validation": _build_dataset(
            data.task,
            validation_examples,
            indexer,
            labels,
            max_seq_length,
            is_character_model,
            pad_token_id,
            pad_token_label_id,
        ),
        "test": _build_dataset(
            data.task,
            test_examples,
            indexer,
            labels,
            max_seq_length,
            is_character_model,
            pad_token_id,
            pad_token_label_id,
        ),
    }

    num_labels = len(labels) if labels is not None else 1
    model = _load_model(data.task, checkpoint_dir, num_labels, is_character_model)
    model.to(device)

    config = TrainingConfig(
        task=data.task,
        output_dir=output_dir,
        device=device,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        adam_epsilon=args.adam_epsilon,
        max_grad_norm=args.max_grad_norm,
        seed=args.seed,
        tensorboard_log_dir=Path(args.tensorboard_log_dir)
        if args.tensorboard_log_dir is not None
        else None,
    )

    best_metric = None
    best_epoch = None
    results = None
    if args.do_train:
        _, _, best_metric, best_epoch = train(
            config=config,
            train_dataset=datasets["train"],
            validation_dataset=datasets["validation"],
            model=model,
            tokenizer=None if is_character_model else tokenizer,
            labels=labels,
            pad_token_label_id=pad_token_label_id,
        )

    if args.do_predict:
        if args.do_train:
            model = _load_model(data.task, output_dir, num_labels, is_character_model)
            model.to(device)
        results, _ = evaluate(
            config=config,
            eval_dataset=datasets["test"],
            model=model,
            labels=labels,
            pad_token_label_id=pad_token_label_id,
        )
        _write_results(output_dir, results, best_metric, best_epoch)

    if args.write_glue_submission:
        if args.do_train:
            model = _load_model(data.task, output_dir, num_labels, is_character_model)
            model.to(device)
        _write_glue_submission(
            args=args,
            data=data,
            model=model,
            config=config,
            tokenizer=indexer,
            feature_tokenizer=feature_tokenizer,
            labels=labels,
            max_seq_length=max_seq_length,
            is_character_model=is_character_model,
            pad_token_id=pad_token_id,
            pad_token_label_id=pad_token_label_id,
            output_dir=output_dir,
        )
        return results if args.do_predict else None

    return results if args.do_predict else None


def _resolve_output_dir(args: argparse.Namespace, task: str) -> Path:
    if args.output_dir:
        return ensure_output_dir(args.output_dir)
    timestamp = dt.datetime.now().strftime("%Y-%m-%d_%Hh%Mm%Ss")
    return ensure_output_dir(
        Path("results") / task / args.embedding / f"{timestamp}__seed-{args.seed}"
    )


def _resolve_checkpoint_dir(embedding: str, pretrained_dir: str | Path) -> Path:
    path = Path(embedding)
    if path.exists():
        return path
    return Path(pretrained_dir) / embedding


def _load_data(args: argparse.Namespace) -> FineTuningData:
    task = "sequence_labeling" if args.task == "sequence_labelling" else args.task
    if args.dataset:
        data = load_finetuning_dataset(
            args.dataset,
            do_lower_case=args.do_lower_case,
            max_train_examples=args.max_train_examples,
            max_validation_examples=args.max_validation_examples,
            max_test_examples=args.max_test_examples,
        )
        if task is not None and task != data.task:
            raise ValueError(f"Dataset preset {args.dataset!r} is for task {data.task!r}")
        return data

    if task is None:
        raise ValueError("--task is required when using --train-file/--test-file")
    if args.train_file is None or args.test_file is None:
        raise ValueError("Provide either --dataset or both --train-file and --test-file")

    train_examples, validation_examples, test_examples = _load_legacy_examples(args, task)
    labels = _labels_for_examples(task, train_examples, validation_examples, test_examples)
    return FineTuningData(
        task=task,
        train_examples=train_examples,
        validation_examples=validation_examples,
        test_examples=test_examples,
        labels=labels,
    )


def _load_tokenizers(
    *,
    checkpoint_dir: Path,
    pretrained_dir: Path,
    do_lower_case: bool,
    is_character_model: bool,
):
    if not is_character_model:
        tokenizer = BertTokenizer.from_pretrained(checkpoint_dir, do_lower_case=do_lower_case)
        return tokenizer, tokenizer

    tokenizer_source = pretrained_dir / "bert-base-uncased"
    if not tokenizer_source.exists():
        tokenizer_source = "bert-base-uncased"
    tokenizer = BertTokenizer.from_pretrained(tokenizer_source, do_lower_case=do_lower_case)
    return tokenizer, BasicTokenizer(do_lower_case=do_lower_case)


def _load_legacy_examples(args: argparse.Namespace, task: str):
    if task == "classification":
        train_examples = load_classification_dataset(
            args.train_file,
            do_lower_case=args.do_lower_case,
        )
        test_examples = load_classification_dataset(
            args.test_file,
            do_lower_case=args.do_lower_case,
        )
    else:
        train_examples = load_sequence_labeling_dataset(
            args.train_file,
            do_lower_case=args.do_lower_case,
        )
        test_examples = load_sequence_labeling_dataset(
            args.test_file,
            do_lower_case=args.do_lower_case,
        )

    split = int(args.validation_ratio * len(train_examples))
    validation_examples = train_examples[:split]
    train_examples = train_examples[split:]
    if not train_examples or not validation_examples:
        raise ValueError("validation-ratio leaves an empty train or validation split")
    return train_examples, validation_examples, test_examples


def _retokenize_examples(examples, tokenize, task: str):
    if task == "classification":
        return retokenize_classification_examples(examples, tokenize)
    if task == "regression":
        return retokenize_classification_examples(examples, tokenize)
    return retokenize_sequence_labeling_examples(examples, tokenize)


def _labels_for_examples(
    task: str,
    train_examples: list[ClassificationExample] | list[SequenceLabelingExample],
    validation_examples: list[ClassificationExample] | list[SequenceLabelingExample],
    test_examples: list[ClassificationExample] | list[SequenceLabelingExample],
) -> list[str]:
    examples = [*train_examples, *validation_examples, *test_examples]
    if task == "classification":
        return sorted({example.label for example in examples})
    if task == "regression":
        return None
    return sorted({label for example in examples for label in example.label_sequence})


def _max_sequence_length(
    task: str,
    train_examples: list[ClassificationExample] | list[SequenceLabelingExample],
    validation_examples: list[ClassificationExample] | list[SequenceLabelingExample],
    test_examples: list[ClassificationExample] | list[SequenceLabelingExample],
) -> int:
    examples = [*train_examples, *validation_examples, *test_examples]
    if task in {"classification", "regression"}:
        longest = max(
            len(example.tokens_a) + (len(example.tokens_b) if example.tokens_b else 0)
            for example in examples
        )
        return min(512, longest + 3)
    longest = max(len(example.token_sequence) for example in examples)
    return min(512, longest + 2)


def _build_dataset(
    task: str,
    examples,
    tokenizer,
    labels: list[str],
    max_seq_length: int,
    is_character_model: bool,
    pad_token_id: int,
    pad_token_label_id: int,
):
    if task in {"classification", "regression"}:
        features = classification_features(
            examples,
            tokenizer=tokenizer,
            labels=labels,
            max_seq_length=max_seq_length,
            is_character_model=is_character_model,
            pad_token_id=pad_token_id,
            regression=task == "regression",
        )
    else:
        features = sequence_labeling_features(
            examples,
            tokenizer=tokenizer,
            labels=labels,
            max_seq_length=max_seq_length,
            is_character_model=is_character_model,
            pad_token_label_id=pad_token_label_id,
            pad_token_id=pad_token_id,
        )
    return features_to_dataset(features, task=task, is_character_model=is_character_model)


def _load_model(task: str, checkpoint_dir: Path, num_labels: int, is_character_model: bool):
    model_class = (
        BertForSequenceClassification
        if task in {"classification", "regression"}
        else BertForTokenClassification
    )
    model_num_labels = 1 if task == "regression" else num_labels
    if not is_character_model:
        config = BertConfig.from_pretrained(checkpoint_dir, num_labels=model_num_labels)
        if task == "regression":
            config.problem_type = "regression"
        return model_class.from_pretrained(checkpoint_dir, config=config)

    config = CharacterBertConfig.from_pretrained(checkpoint_dir, num_labels=model_num_labels)
    if task == "regression":
        config.problem_type = "regression"
    model = model_class(config=config)
    model.bert = CharacterBertModel(config=config)

    state_dict = _load_state_dict(checkpoint_dir)
    if any(key.startswith("bert.") for key in state_dict):
        model.load_state_dict(state_dict, strict=True)
    else:
        model.bert = CharacterBertModel.from_pretrained(checkpoint_dir, config=config)
    return model


def _load_state_dict(checkpoint_dir: Path) -> dict[str, torch.Tensor]:
    safetensors_path = checkpoint_dir / "model.safetensors"
    if safetensors_path.exists():
        return load_safetensors_file(safetensors_path)
    return torch.load(checkpoint_dir / "pytorch_model.bin", map_location="cpu")


def _write_results(
    output_dir: Path,
    results: dict[str, float],
    best_metric: float | None,
    best_epoch: int | None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "performance_on_test_set.txt").open("w", encoding="utf-8") as output_file:
        if best_metric is not None:
            output_file.write(f"best validation score: {best_metric}\n")
        if best_epoch is not None:
            output_file.write(f"best validation epoch: {best_epoch}\n")
        output_file.write("--- Performance on test set ---\n")
        for key, value in sorted(results.items()):
            output_file.write(f"{key}: {value}\n")


def _write_glue_submission(
    *,
    args: argparse.Namespace,
    data: FineTuningData,
    model: torch.nn.Module,
    config: TrainingConfig,
    tokenizer,
    feature_tokenizer,
    labels: list[str] | None,
    max_seq_length: int,
    is_character_model: bool,
    pad_token_id: int,
    pad_token_label_id: int,
    output_dir: Path,
) -> None:
    if data.name not in GLUE_TASKS:
        raise ValueError("--write-glue-submission is only available for GLUE datasets")

    spec = GLUE_TASKS[data.name]
    submission_dir = output_dir / "glue_submission"
    submission_dir.mkdir(parents=True, exist_ok=True)

    for split_name, filename in glue_submission_splits(spec).items():
        examples = load_test_examples(
            spec,
            split_name,
            labels=labels,
            do_lower_case=args.do_lower_case,
            max_examples=args.max_test_examples,
        )
        examples = _retokenize_examples(examples, feature_tokenizer.tokenize, data.task)
        dataset = _build_dataset(
            data.task,
            examples,
            tokenizer,
            labels,
            max_seq_length,
            is_character_model,
            pad_token_id,
            pad_token_label_id,
        )
        logits = predict(config=config, dataset=dataset, model=model)
        predictions = _glue_predictions(data.task, logits, labels)
        with (submission_dir / filename).open("w", encoding="utf-8") as output_file:
            output_file.write("index\tprediction\n")
            for example, prediction_value in zip(examples, predictions, strict=True):
                output_file.write(f"{example.id}\t{prediction_value}\n")


def _glue_predictions(
    task: str,
    logits,
    labels: list[str] | None,
) -> list[str]:
    if task == "regression":
        return [f"{min(5.0, max(0.0, float(value))):.3f}" for value in logits.squeeze(-1)]
    return [labels[index] for index in logits.argmax(axis=1)]
