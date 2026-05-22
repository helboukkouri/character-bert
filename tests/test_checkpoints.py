import os
import unittest

os.environ.setdefault("TRANSFORMERS_CACHE", "/tmp/hf-cache")
os.environ.setdefault("HF_HOME", "/tmp/hf-home")

import torch

from character_bert.checkpoints import read_mlm_vocab
from character_bert.modeling import (
    CharacterBertForPreTraining,
    CharacterBertModel,
    CharacterIndexer,
)

PRETRAINED_DIR = "pretrained-models"


def has_checkpoint(name: str, required_files=("config.json", "pytorch_model.bin")) -> bool:
    return all(os.path.exists(os.path.join(PRETRAINED_DIR, name, file)) for file in required_files)


def require_checkpoint(name: str, required_files=("config.json", "pytorch_model.bin")) -> None:
    if not has_checkpoint(name, required_files):
        raise unittest.SkipTest(f"Missing checkpoint: {name}")


def encode_tokens(tokens: list[str], indexer: CharacterIndexer):
    return {
        "input_ids": indexer.as_padded_tensor([tokens]),
        "attention_mask": torch.ones(1, len(tokens), dtype=torch.long),
        "token_type_ids": torch.zeros(1, len(tokens), dtype=torch.long),
    }


class CheckpointCompatibilityTest(unittest.TestCase):
    def test_drive_encoder_loads_with_refreshed_model_class(self):
        require_checkpoint("general_character_bert")
        indexer = CharacterIndexer()
        inputs = encode_tokens(["[CLS]", "hello", "world", "[SEP]"], indexer)

        model = CharacterBertModel.from_pretrained(
            os.path.join(PRETRAINED_DIR, "general_character_bert")
        )
        model.eval()

        with torch.no_grad():
            outputs = model(**inputs, return_dict=True)

        self.assertEqual(tuple(outputs.last_hidden_state.shape), (1, 4, 768))

    def test_hub_checkpoint_is_drive_encoder_plus_pretraining_heads(self):
        require_checkpoint("general_character_bert")
        require_checkpoint(
            "hf_character_bert",
            ("config.json", "pytorch_model.bin", "mlm_vocab.txt"),
        )

        drive_state = torch.load(
            os.path.join(PRETRAINED_DIR, "general_character_bert", "pytorch_model.bin"),
            map_location="cpu",
        )
        hub_state = torch.load(
            os.path.join(PRETRAINED_DIR, "hf_character_bert", "pytorch_model.bin"),
            map_location="cpu",
        )
        hub_encoder_state = {
            key.removeprefix("character_bert."): value
            for key, value in hub_state.items()
            if key.startswith("character_bert.")
        }
        hub_head_keys = {key for key in hub_state if key.startswith("cls.")}

        self.assertEqual(len(hub_head_keys), 9)
        self.assertFalse(set(drive_state) - set(hub_encoder_state))
        for key, expected_tensor in drive_state.items():
            self.assertTrue(torch.equal(expected_tensor, hub_encoder_state[key]), key)

    def test_hub_pretraining_head_predicts_masked_word(self):
        require_checkpoint(
            "hf_character_bert",
            ("config.json", "pytorch_model.bin", "mlm_vocab.txt"),
        )
        indexer = CharacterIndexer()
        tokens = ["[CLS]", "paris", "is", "the", "capital", "of", "[MASK]", ".", "[SEP]"]
        inputs = encode_tokens(tokens, indexer)
        vocab = read_mlm_vocab(os.path.join(PRETRAINED_DIR, "hf_character_bert"))

        model = CharacterBertForPreTraining.from_pretrained(
            os.path.join(PRETRAINED_DIR, "hf_character_bert")
        )
        model.eval()

        with torch.no_grad():
            outputs = model(**inputs, return_dict=True)

        top_id = outputs.prediction_logits[0, tokens.index("[MASK]")].argmax().item()
        self.assertEqual(vocab[top_id], "france")


if __name__ == "__main__":
    unittest.main()
