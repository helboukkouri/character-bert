from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from tqdm.auto import tqdm, trange
from transformers import PreTrainedTokenizerBase, get_linear_schedule_with_warmup

from character_bert.finetuning.metrics import (
    classification_metrics,
    regression_metrics,
    sequence_labeling_metrics,
)
from character_bert.finetuning.utils.seed import set_seed

LOGGER = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    task: str
    output_dir: Path
    device: torch.device
    train_batch_size: int = 8
    eval_batch_size: int = 8
    gradient_accumulation_steps: int = 1
    num_train_epochs: int = 3
    learning_rate: float = 5e-5
    weight_decay: float = 0.1
    warmup_ratio: float = 0.1
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0
    seed: int = 42


def train(
    *,
    config: TrainingConfig,
    train_dataset: TensorDataset,
    validation_dataset: TensorDataset,
    model: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase | None,
    labels: list[str],
    pad_token_label_id: int,
) -> tuple[int, float, float, int]:
    train_dataloader = DataLoader(
        train_dataset,
        sampler=RandomSampler(train_dataset),
        batch_size=config.train_batch_size,
    )
    steps_per_epoch = max(1, len(train_dataloader) // config.gradient_accumulation_steps)
    total_steps = steps_per_epoch * config.num_train_epochs

    no_decay = ["bias", "LayerNorm.weight"]
    grouped_parameters = [
        {
            "params": [
                parameter
                for name, parameter in model.named_parameters()
                if not any(no_decay_name in name for no_decay_name in no_decay)
            ],
            "weight_decay": config.weight_decay,
        },
        {
            "params": [
                parameter
                for name, parameter in model.named_parameters()
                if any(no_decay_name in name for no_decay_name in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(grouped_parameters, lr=config.learning_rate, eps=config.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(config.warmup_ratio * total_steps),
        num_training_steps=total_steps,
    )

    LOGGER.info(
        "Running training on %d examples for %d epochs",
        len(train_dataset),
        config.num_train_epochs,
    )
    global_step = 0
    total_loss = 0.0
    best_metric = -1.0
    best_epoch = -1

    model.zero_grad()
    set_seed(config.seed)
    for epoch in trange(config.num_train_epochs, desc="Epoch"):
        for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
            model.train()
            batch = tuple(tensor.to(config.device) for tensor in batch)
            outputs = model(
                input_ids=batch[0],
                attention_mask=batch[1],
                token_type_ids=batch[2],
                labels=batch[3],
                return_dict=False,
            )
            loss = outputs[0]
            if config.gradient_accumulation_steps > 1:
                loss = loss / config.gradient_accumulation_steps
            loss.backward()

            total_loss += loss.item()
            if (step + 1) % config.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                optimizer.step()
                scheduler.step()
                model.zero_grad()
                global_step += 1

        results, _ = evaluate(
            config=config,
            eval_dataset=validation_dataset,
            model=model,
            labels=labels,
            pad_token_label_id=pad_token_label_id,
        )
        metric = results["f1"]
        if metric > best_metric:
            best_metric = metric
            best_epoch = epoch
            config.output_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(config.output_dir)
            if tokenizer is not None:
                tokenizer.save_pretrained(config.output_dir)
            torch.save(config, config.output_dir / "training_args.bin")
            LOGGER.info("Saved best model checkpoint to %s", config.output_dir)

    average_loss = total_loss / max(global_step, 1)
    return global_step, average_loss, best_metric, best_epoch


def evaluate(
    *,
    config: TrainingConfig,
    eval_dataset: TensorDataset,
    model: torch.nn.Module,
    labels: list[str],
    pad_token_label_id: int,
) -> tuple[dict[str, float], list[int] | list[float] | list[list[str]]]:
    eval_dataloader = DataLoader(
        eval_dataset,
        sampler=SequentialSampler(eval_dataset),
        batch_size=config.eval_batch_size,
    )

    eval_loss = 0.0
    predictions = None
    label_ids = None
    model.eval()
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        batch = tuple(tensor.to(config.device) for tensor in batch)
        with torch.no_grad():
            outputs = model(
                input_ids=batch[0],
                attention_mask=batch[1],
                token_type_ids=batch[2],
                labels=batch[3],
                return_dict=False,
            )
            loss, logits = outputs[:2]

        eval_loss += loss.item()
        batch_predictions = logits.detach().cpu().numpy()
        batch_labels = batch[3].detach().cpu().numpy()
        predictions = (
            batch_predictions
            if predictions is None
            else np.append(predictions, batch_predictions, axis=0)
        )
        label_ids = (
            batch_labels if label_ids is None else np.append(label_ids, batch_labels, axis=0)
        )

    eval_loss = eval_loss / max(len(eval_dataloader), 1)
    if config.task == "classification":
        label_map = {index: label for index, label in enumerate(labels)}
        predicted_labels = np.argmax(predictions, axis=1)
        results = {"loss": eval_loss, **classification_metrics(label_ids, predicted_labels)}
        return results, predicted_labels.tolist()

    if config.task == "regression":
        predicted_scores = np.squeeze(predictions, axis=-1)
        results = {"loss": eval_loss, **regression_metrics(label_ids, predicted_scores)}
        return results, predicted_scores.tolist()

    label_map = {index: label for index, label in enumerate(labels)}
    predicted_label_ids = np.argmax(predictions, axis=2)
    true_label_list: list[list[str]] = []
    prediction_list: list[list[str]] = []
    for example_index in range(label_ids.shape[0]):
        true_sentence = []
        prediction_sentence = []
        for token_index in range(label_ids.shape[1]):
            if label_ids[example_index, token_index] == pad_token_label_id:
                continue
            true_sentence.append(label_map[label_ids[example_index, token_index]])
            prediction_sentence.append(label_map[predicted_label_ids[example_index, token_index]])
        true_label_list.append(true_sentence)
        prediction_list.append(prediction_sentence)

    results = {"loss": eval_loss, **sequence_labeling_metrics(true_label_list, prediction_list)}
    return results, prediction_list


def predict(
    *,
    config: TrainingConfig,
    dataset: TensorDataset,
    model: torch.nn.Module,
) -> np.ndarray:
    dataloader = DataLoader(
        dataset,
        sampler=SequentialSampler(dataset),
        batch_size=config.eval_batch_size,
    )
    predictions = None
    model.eval()
    for batch in tqdm(dataloader, desc="Predicting"):
        batch = tuple(tensor.to(config.device) for tensor in batch)
        with torch.no_grad():
            outputs = model(
                input_ids=batch[0],
                attention_mask=batch[1],
                token_type_ids=batch[2],
                return_dict=False,
            )
            logits = outputs[0]
        batch_predictions = logits.detach().cpu().numpy()
        predictions = (
            batch_predictions
            if predictions is None
            else np.append(predictions, batch_predictions, axis=0)
        )
    return predictions
