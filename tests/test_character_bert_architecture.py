import unittest

import torch
from transformers import BertConfig

from modeling.character_bert import BertCharacterEmbeddings, CharacterBertModel
from modeling.character_cnn import CharacterCNN
from utils.character_cnn import CharacterIndexer, CharacterMapper


class CharacterIndexerTest(unittest.TestCase):
    def test_indexer_pads_batches_and_preserves_special_tokens(self):
        indexer = CharacterIndexer()
        batch = [
            ["[CLS]", "hello", "[SEP]"],
            ["[CLS]", "a", "longer", "example", "[SEP]"],
        ]

        tensor = indexer.as_padded_tensor(batch)

        self.assertEqual(tuple(tensor.shape), (2, 5, CharacterMapper.max_word_length))
        self.assertTrue(torch.equal(tensor[0, 0], torch.tensor(CharacterMapper.beginning_of_sentence_characters) + 1))
        self.assertTrue(torch.equal(tensor[0, 2], torch.tensor(CharacterMapper.end_of_sentence_characters) + 1))
        self.assertTrue(torch.equal(tensor[0, 3], torch.zeros(CharacterMapper.max_word_length, dtype=torch.long)))
        self.assertEqual(tensor[1, 1, 0].item(), CharacterMapper.beginning_of_word_character + 1)
        self.assertEqual(tensor[1, 1, 1].item(), ord("a") + 1)
        self.assertEqual(tensor[1, 1, 2].item(), CharacterMapper.end_of_word_character + 1)


class CharacterCNNTest(unittest.TestCase):
    def test_character_cnn_returns_one_embedding_per_token(self):
        indexer = CharacterIndexer()
        inputs = indexer.as_padded_tensor([["[CLS]", "characterbert", "[SEP]"]])
        model = CharacterCNN(output_dim=32)

        output = model(inputs)

        self.assertEqual(tuple(output.shape), (1, 3, 32))
        self.assertFalse(torch.isnan(output).any())

    def test_character_cnn_supports_backpropagation(self):
        indexer = CharacterIndexer()
        inputs = indexer.as_padded_tensor([["[CLS]", "gradient", "[SEP]"]])
        model = CharacterCNN(output_dim=16)

        output = model(inputs)
        output.sum().backward()

        self.assertIsNotNone(model._projection.weight.grad)
        self.assertFalse(torch.isnan(model._projection.weight.grad).any())


class CharacterBertModelTest(unittest.TestCase):
    def _small_config(self):
        return BertConfig(
            hidden_size=32,
            intermediate_size=64,
            num_attention_heads=4,
            num_hidden_layers=2,
            max_position_embeddings=16,
            type_vocab_size=2,
            hidden_dropout_prob=0.0,
            attention_probs_dropout_prob=0.0,
        )

    def test_embedding_module_replaces_wordpiece_embeddings_with_character_cnn(self):
        embeddings = BertCharacterEmbeddings(self._small_config())

        self.assertIsInstance(embeddings.word_embeddings, CharacterCNN)
        self.assertEqual(embeddings.word_embeddings.get_output_dim(), 32)
        self.assertEqual(embeddings.position_embeddings.embedding_dim, 32)
        self.assertEqual(embeddings.token_type_embeddings.embedding_dim, 32)

    def test_model_forward_matches_bert_base_output_contract(self):
        indexer = CharacterIndexer()
        inputs = indexer.as_padded_tensor([
            ["[CLS]", "hello", "[SEP]"],
            ["[CLS]", "a", "longer", "test", "[SEP]"],
        ])
        attention_mask = torch.tensor([
            [1, 1, 1, 0, 0],
            [1, 1, 1, 1, 1],
        ])
        model = CharacterBertModel(self._small_config())
        model.eval()

        with torch.no_grad():
            sequence_output, pooled_output = model(inputs, attention_mask=attention_mask)[:2]

        self.assertEqual(tuple(sequence_output.shape), (2, 5, 32))
        self.assertEqual(tuple(pooled_output.shape), (2, 32))
        self.assertFalse(torch.isnan(sequence_output).any())
        self.assertFalse(torch.isnan(pooled_output).any())

    def test_model_can_return_hidden_states_and_attentions(self):
        config = self._small_config()
        config.output_hidden_states = True
        config.output_attentions = True
        indexer = CharacterIndexer()
        inputs = indexer.as_padded_tensor([["[CLS]", "inspect", "[SEP]"]])
        model = CharacterBertModel(config)
        model.eval()

        with torch.no_grad():
            outputs = model(inputs)

        sequence_output, pooled_output, hidden_states, attentions = outputs
        self.assertEqual(tuple(sequence_output.shape), (1, 3, 32))
        self.assertEqual(tuple(pooled_output.shape), (1, 32))
        self.assertEqual(len(hidden_states), config.num_hidden_layers + 1)
        self.assertEqual(len(attentions), config.num_hidden_layers)


if __name__ == "__main__":
    unittest.main()
