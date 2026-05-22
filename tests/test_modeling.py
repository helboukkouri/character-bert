import unittest

import torch
from transformers import BertConfig

from character_bert.modeling import (
    CharacterBertForPreTraining,
    CharacterBertModel,
    CharacterCNN,
    CharacterIndexer,
    CharacterMapper,
)


class CharacterIndexerTest(unittest.TestCase):
    def test_special_tokens_and_padding(self):
        indexer = CharacterIndexer()
        batch = [["[CLS]", "hello", "[SEP]"], ["[CLS]", "longer", "example", "[SEP]"]]

        tensor = indexer.as_padded_tensor(batch)

        self.assertEqual(tuple(tensor.shape), (2, 4, CharacterMapper.max_word_length))
        self.assertTrue(
            torch.equal(
                tensor[0, 0],
                torch.tensor(CharacterMapper.beginning_of_sentence_characters) + 1,
            )
        )
        self.assertTrue(
            torch.equal(
                tensor[0, 2],
                torch.tensor(CharacterMapper.end_of_sentence_characters) + 1,
            )
        )
        self.assertTrue(
            torch.equal(
                tensor[0, 3],
                torch.zeros(CharacterMapper.max_word_length, dtype=torch.long),
            )
        )


class CharacterBertArchitectureTest(unittest.TestCase):
    def small_config(self, **overrides):
        values = {
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_attention_heads": 4,
            "num_hidden_layers": 2,
            "max_position_embeddings": 32,
            "vocab_size": 101,
            "hidden_dropout_prob": 0.0,
            "attention_probs_dropout_prob": 0.0,
        }
        values.update(overrides)
        return BertConfig(**values)

    def test_character_cnn_shape_and_gradients(self):
        indexer = CharacterIndexer()
        inputs = indexer.as_padded_tensor([["[CLS]", "gradient", "[SEP]"]])
        model = CharacterCNN(output_dim=16)

        output = model(inputs)
        output.sum().backward()

        self.assertEqual(tuple(output.shape), (1, 3, 16))
        self.assertIsNotNone(model._projection.weight.grad)

    def test_encoder_output_contract(self):
        indexer = CharacterIndexer()
        inputs = indexer.as_padded_tensor([["[CLS]", "hello", "[SEP]"]])
        model = CharacterBertModel(
            self.small_config(output_hidden_states=True, output_attentions=True)
        )
        model.eval()

        with torch.no_grad():
            outputs = model(inputs, return_dict=True)

        self.assertEqual(tuple(outputs.last_hidden_state.shape), (1, 3, 32))
        self.assertEqual(tuple(outputs.pooler_output.shape), (1, 32))
        self.assertEqual(len(outputs.hidden_states), 3)
        self.assertEqual(len(outputs.attentions), 2)

    def test_pretraining_head_output_contract(self):
        indexer = CharacterIndexer()
        inputs = indexer.as_padded_tensor([["[CLS]", "hello", "[MASK]", "[SEP]"]])
        model = CharacterBertForPreTraining(self.small_config(vocab_size=99))
        model.eval()

        with torch.no_grad():
            outputs = model(inputs, return_dict=True)

        self.assertEqual(tuple(outputs.prediction_logits.shape), (1, 4, 99))
        self.assertEqual(tuple(outputs.seq_relationship_logits.shape), (1, 2))


if __name__ == "__main__":
    unittest.main()
