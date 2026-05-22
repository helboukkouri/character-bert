import argparse

import torch
from transformers import BertConfig

from character_bert.checkpoints import download_checkpoint
from character_bert.modeling import CharacterBertModel, CharacterIndexer


def download() -> None:
    parser = argparse.ArgumentParser(description="Download a CharacterBERT checkpoint.")
    parser.add_argument("model", help="Hub repo id or alias: general, medical")
    parser.add_argument("--output-dir", default=None, help="Local destination directory")
    args = parser.parse_args()

    destination = download_checkpoint(args.model, args.output_dir)
    print(destination)


def smoke_test() -> None:
    indexer = CharacterIndexer()
    inputs = indexer.as_padded_tensor([["[CLS]", "characterbert", "works", "[SEP]"]])
    model = CharacterBertModel(
        BertConfig(
            hidden_size=32,
            intermediate_size=64,
            num_attention_heads=4,
            num_hidden_layers=2,
            hidden_dropout_prob=0.0,
            attention_probs_dropout_prob=0.0,
        )
    )
    model.eval()
    with torch.no_grad():
        sequence_output, pooled_output = model(inputs, return_dict=False)[:2]
    print(
        f"sequence_output={tuple(sequence_output.shape)} "
        f"pooled_output={tuple(pooled_output.shape)}"
    )
