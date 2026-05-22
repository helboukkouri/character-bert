import os
import unittest

os.environ.setdefault("TRANSFORMERS_CACHE", "/tmp/hf-cache")
os.environ.setdefault("HF_HOME", "/tmp/hf-home")

import torch
from transformers import AutoModel, BertForPreTraining, BertModel, BertTokenizer

from modeling.character_bert import CharacterBertModel
from utils.character_cnn import CharacterIndexer


PRETRAINED_DIR = "pretrained-models"


def has_files(model_name, filenames):
    return all(os.path.exists(os.path.join(PRETRAINED_DIR, model_name, filename)) for filename in filenames)


def require_model(model_name, filenames=("config.json", "pytorch_model.bin")):
    if not has_files(model_name, filenames):
        raise unittest.SkipTest(
            f"Missing pretrained model `{model_name}`. Run `python download.py --model='{model_name}'` first."
        )


def read_mlm_vocab(model_name):
    with open(os.path.join(PRETRAINED_DIR, model_name, "mlm_vocab.txt"), encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def encode_character_tokens(tokens, indexer, token_type_ids=None):
    if token_type_ids is None:
        token_type_ids = torch.zeros(1, len(tokens), dtype=torch.long)
    input_ids = indexer.as_padded_tensor([tokens])
    attention_mask = torch.ones(1, len(tokens), dtype=torch.long)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "token_type_ids": token_type_ids,
    }


def encode_character_pair(sentence_a, sentence_b, tokenizer, indexer):
    tokens_a = tokenizer.basic_tokenizer.tokenize(sentence_a)
    tokens_b = tokenizer.basic_tokenizer.tokenize(sentence_b)
    tokens = ["[CLS]", *tokens_a, "[SEP]", *tokens_b, "[SEP]"]
    first_segment_length = len(tokens_a) + 2
    token_type_ids = torch.tensor(
        [[0] * first_segment_length + [1] * (len(tokens_b) + 1)],
        dtype=torch.long,
    )
    return encode_character_tokens(tokens, indexer, token_type_ids=token_type_ids)


class PretrainedEncoderRegressionTest(unittest.TestCase):
    maxDiff = None

    BERT_EXPECTED = {
        "bert-base-uncased": {
            "seq5": [-0.142413, 0.133537, -0.129071, -0.171648, -0.483229],
            "pool5": [-0.897565, -0.330402, -0.769421, 0.757993, 0.466782],
        },
        "general_bert": {
            "seq5": [0.27206, 1.100574, -0.585363, 0.324986, -0.252885],
            "pool5": [-0.534796, 0.563107, 0.80061, 0.123273, 0.58578],
        },
        "medical_bert": {
            "seq5": [0.284317, 0.212882, -0.227688, 0.047266, 0.436771],
            "pool5": [0.073973, 0.068475, 0.262793, -0.368525, 0.138402],
        },
    }

    CHARACTER_BERT_EXPECTED = {
        "general_character_bert": {
            "seq5": [-0.065009, 0.282772, -0.292036, -0.156121, 0.16317],
            "pool5": [0.04807, -0.224837, -0.168624, 0.362929, 0.836712],
        },
        "medical_character_bert": {
            "seq5": [-0.263793, 0.168549, -0.124364, -0.129445, 0.003586],
            "pool5": [-0.01892, -0.005544, -0.171266, 0.140695, 0.815971],
        },
    }

    def assert_close(self, actual, expected):
        torch.testing.assert_close(
            actual.cpu(),
            torch.tensor(expected, dtype=actual.dtype),
            rtol=0,
            atol=1e-5,
        )

    def test_pretrained_bert_encoder_outputs_are_stable(self):
        require_model("bert-base-uncased", ("config.json", "pytorch_model.bin", "vocab.txt"))
        tokenizer = BertTokenizer.from_pretrained(os.path.join(PRETRAINED_DIR, "bert-base-uncased"))
        inputs = tokenizer("Hello World!", return_tensors="pt")

        for model_name, expected in self.BERT_EXPECTED.items():
            with self.subTest(model=model_name):
                require_model(model_name)
                model = BertModel.from_pretrained(os.path.join(PRETRAINED_DIR, model_name))
                model.eval()

                with torch.no_grad():
                    sequence_output, pooled_output = model(**inputs, return_dict=False)[:2]

                self.assertEqual(tuple(sequence_output.shape), (1, 5, 768))
                self.assertEqual(tuple(pooled_output.shape), (1, 768))
                self.assert_close(sequence_output[0, 0, :5], expected["seq5"])
                self.assert_close(pooled_output[0, :5], expected["pool5"])

    def test_pretrained_character_bert_encoder_outputs_are_stable(self):
        require_model("bert-base-uncased", ("config.json", "pytorch_model.bin", "vocab.txt"))
        tokenizer = BertTokenizer.from_pretrained(os.path.join(PRETRAINED_DIR, "bert-base-uncased"))
        tokens = ["[CLS]", *tokenizer.basic_tokenizer.tokenize("Hello World!"), "[SEP]"]
        inputs = CharacterIndexer().as_padded_tensor([tokens])

        self.assertEqual(tokens, ["[CLS]", "hello", "world", "!", "[SEP]"])
        for model_name, expected in self.CHARACTER_BERT_EXPECTED.items():
            with self.subTest(model=model_name):
                require_model(model_name)
                model = CharacterBertModel.from_pretrained(os.path.join(PRETRAINED_DIR, model_name))
                model.eval()

                with torch.no_grad():
                    sequence_output, pooled_output = model(inputs)[:2]

                self.assertEqual(tuple(sequence_output.shape), (1, 5, 768))
                self.assertEqual(tuple(pooled_output.shape), (1, 768))
                self.assert_close(sequence_output[0, 0, :5], expected["seq5"])
                self.assert_close(pooled_output[0, :5], expected["pool5"])


class HuggingFaceCharacterBertComparisonTest(unittest.TestCase):
    HUB_TO_DRIVE = {
        "hf_character_bert": "general_character_bert",
        "hf_character_bert_medical": "medical_character_bert",
    }

    def test_hub_checkpoints_are_drive_encoders_plus_pretraining_heads(self):
        expected_head_keys = {
            "cls.predictions.bias",
            "cls.predictions.decoder.bias",
            "cls.predictions.decoder.weight",
            "cls.predictions.transform.LayerNorm.bias",
            "cls.predictions.transform.LayerNorm.weight",
            "cls.predictions.transform.dense.bias",
            "cls.predictions.transform.dense.weight",
            "cls.seq_relationship.bias",
            "cls.seq_relationship.weight",
        }

        for hub_model, drive_model in self.HUB_TO_DRIVE.items():
            with self.subTest(hub_model=hub_model, drive_model=drive_model):
                require_model(hub_model, ("config.json", "pytorch_model.bin", "mlm_vocab.txt"))
                require_model(drive_model)

                drive_state = torch.load(
                    os.path.join(PRETRAINED_DIR, drive_model, "pytorch_model.bin"),
                    map_location="cpu",
                )
                hub_state = torch.load(
                    os.path.join(PRETRAINED_DIR, hub_model, "pytorch_model.bin"),
                    map_location="cpu",
                )
                hub_encoder_state = {
                    key.removeprefix("character_bert."): value
                    for key, value in hub_state.items()
                    if key.startswith("character_bert.")
                }
                hub_head_keys = {key for key in hub_state if not key.startswith("character_bert.")}

                self.assertEqual(hub_head_keys, expected_head_keys)
                self.assertEqual(set(hub_encoder_state) - set(drive_state), {"embeddings.position_ids"})
                self.assertFalse(set(drive_state) - set(hub_encoder_state))
                for key, expected_tensor in drive_state.items():
                    self.assertTrue(torch.equal(expected_tensor, hub_encoder_state[key]), key)


class BertPretrainingHeadsTest(unittest.TestCase):
    def setUp(self):
        require_model("bert-base-uncased", ("config.json", "pytorch_model.bin", "vocab.txt"))
        self.tokenizer = BertTokenizer.from_pretrained(os.path.join(PRETRAINED_DIR, "bert-base-uncased"))
        self.model = BertForPreTraining.from_pretrained(os.path.join(PRETRAINED_DIR, "bert-base-uncased"))
        self.model.eval()

    def test_masked_language_model_predicts_missing_country(self):
        inputs = self.tokenizer("Paris is the capital of [MASK].", return_tensors="pt")
        mask_index = (inputs["input_ids"][0] == self.tokenizer.mask_token_id).nonzero(as_tuple=True)[0].item()

        with torch.no_grad():
            outputs = self.model(**inputs, return_dict=True)

        ranked_token_ids = torch.topk(outputs.prediction_logits[0, mask_index], k=5).indices.tolist()
        ranked_tokens = self.tokenizer.convert_ids_to_tokens(ranked_token_ids)

        self.assertEqual(ranked_tokens[0], "france")
        self.assertIn("morocco", ranked_tokens)

    def test_next_sentence_prediction_separates_related_and_unrelated_pairs(self):
        related = self.tokenizer(
            "The man went to the store.",
            "He bought a gallon of milk.",
            return_tensors="pt",
        )
        unrelated = self.tokenizer(
            "The man went to the store.",
            "Quantum fields are described by wave functions.",
            return_tensors="pt",
        )

        with torch.no_grad():
            related_probs = self.model(**related, return_dict=True).seq_relationship_logits.softmax(-1)[0]
            unrelated_probs = self.model(**unrelated, return_dict=True).seq_relationship_logits.softmax(-1)[0]

        self.assertGreater(related_probs[0].item(), 0.99)
        self.assertLess(related_probs[1].item(), 0.01)
        self.assertLess(unrelated_probs[0].item(), 0.01)
        self.assertGreater(unrelated_probs[1].item(), 0.99)


class CharacterBertPretrainingHeadsTest(unittest.TestCase):
    HUB_MODELS = {
        "hf_character_bert": {
            "mlm_top": "france",
            "related_threshold": 0.99,
            "unrelated_threshold": 0.99,
        },
        "hf_character_bert_medical": {
            "mlm_top": "france",
            "related_threshold": 0.99,
            "unrelated_threshold": 0.99,
        },
    }

    def setUp(self):
        require_model("bert-base-uncased", ("config.json", "pytorch_model.bin", "vocab.txt"))
        self.bert_tokenizer = BertTokenizer.from_pretrained(os.path.join(PRETRAINED_DIR, "bert-base-uncased"))
        self.character_indexer = CharacterIndexer()

    def test_hub_character_bert_models_predict_masked_words_and_next_sentences(self):
        masked_tokens = ["[CLS]", "paris", "is", "the", "capital", "of", "[MASK]", ".", "[SEP]"]
        masked_inputs = encode_character_tokens(masked_tokens, self.character_indexer)
        mask_index = masked_tokens.index("[MASK]")

        for model_name, expectations in self.HUB_MODELS.items():
            with self.subTest(model=model_name):
                require_model(model_name, ("config.json", "pytorch_model.bin", "mlm_vocab.txt"))
                mlm_vocab = read_mlm_vocab(model_name)
                model = AutoModel.from_pretrained(
                    os.path.join(PRETRAINED_DIR, model_name),
                    trust_remote_code=True,
                )
                model.eval()

                with torch.no_grad():
                    outputs = model(**masked_inputs, return_dict=True)
                ranked_token_ids = torch.topk(outputs.prediction_logits[0, mask_index], k=10).indices.tolist()
                ranked_tokens = [mlm_vocab[token_id] for token_id in ranked_token_ids]

                self.assertEqual(ranked_tokens[0], expectations["mlm_top"])

                related = encode_character_pair(
                    "The man went to the store.",
                    "He bought a gallon of milk.",
                    self.bert_tokenizer,
                    self.character_indexer,
                )
                unrelated = encode_character_pair(
                    "The man went to the store.",
                    "Quantum fields are described by wave functions.",
                    self.bert_tokenizer,
                    self.character_indexer,
                )

                with torch.no_grad():
                    related_probs = model(**related, return_dict=True).seq_relationship_logits.softmax(-1)[0]
                    unrelated_probs = model(**unrelated, return_dict=True).seq_relationship_logits.softmax(-1)[0]

                self.assertGreater(related_probs[0].item(), expectations["related_threshold"])
                self.assertGreater(unrelated_probs[1].item(), expectations["unrelated_threshold"])


if __name__ == "__main__":
    unittest.main()
