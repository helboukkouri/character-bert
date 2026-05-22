import tempfile
import unittest
from pathlib import Path

import torch
from datasets import ClassLabel, Dataset, Features, Sequence, Value
from transformers import BertTokenizer

from character_bert.finetuning.datasets import DATASET_PRESETS
from character_bert.finetuning.features import (
    classification_features,
    features_to_dataset,
    sequence_labeling_features,
)
from character_bert.finetuning.metrics import (
    classification_metrics,
    get_entities,
    sequence_labeling_metrics,
)
from character_bert.finetuning.tasks import (
    ClassificationExample,
    SequenceLabelingExample,
    load_classification_dataset,
    load_classification_hf_dataset,
    load_sequence_labeling_dataset,
    load_sequence_labeling_hf_dataset,
    retokenize_sequence_labeling_examples,
)
from character_bert.modeling import CharacterIndexer


class FineTuningDataTests(unittest.TestCase):
    def test_loads_legacy_classification_format(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "train.txt"
            path.write_text(
                "__label__pos This is lovely\n__label__neg Bad news\n",
                encoding="utf-8",
            )

            examples = load_classification_dataset(path, do_lower_case=True)

        self.assertEqual(len(examples), 2)
        self.assertEqual(examples[0].label, "pos")
        self.assertEqual(examples[0].tokens_a, ["this", "is", "lovely"])

    def test_loads_datasets_classification_rows(self):
        dataset = Dataset.from_dict(
            {"sentence": ["A warm movie."], "label": [1]},
            features=Features(
                {
                    "sentence": Value("string"),
                    "label": ClassLabel(names=["negative", "positive"]),
                }
            ),
        )

        examples = load_classification_hf_dataset(
            dataset,
            text_column="sentence",
            text_pair_column=None,
            label_column="label",
            labels=["negative", "positive"],
            do_lower_case=True,
        )

        self.assertEqual(examples[0].tokens_a, ["a", "warm", "movie", "."])
        self.assertEqual(examples[0].label, "positive")

    def test_loads_datasets_regression_rows(self):
        dataset = Dataset.from_dict(
            {
                "sentence1": ["A dog runs."],
                "sentence2": ["An animal moves."],
                "label": [4.2],
                "idx": [7],
            }
        )

        examples = load_classification_hf_dataset(
            dataset,
            text_column="sentence1",
            text_pair_column="sentence2",
            label_column="label",
            labels=None,
            do_lower_case=True,
            regression=True,
        )

        self.assertEqual(examples[0].id, 7)
        self.assertEqual(examples[0].tokens_b, ["an", "animal", "moves", "."])
        self.assertEqual(examples[0].label, 4.2)

    def test_loads_and_retokenizes_sequence_labeling_format(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "train.txt"
            path.write_text("John B-PER\nSmith I-PER\n\nParis B-LOC\n", encoding="utf-8")

            examples = load_sequence_labeling_dataset(path, do_lower_case=False)
            retokenized = retokenize_sequence_labeling_examples(
                [SequenceLabelingExample(0, ["unaffable"], ["B-MISC"])],
                lambda token: ["una", "##ffa", "##ble"],
            )

        self.assertEqual(len(examples), 2)
        self.assertEqual(examples[0].label_sequence, ["B-PER", "I-PER"])
        self.assertEqual(retokenized[0].label_sequence, ["B-MISC", "I-MISC", "I-MISC"])

    def test_loads_datasets_sequence_labeling_rows(self):
        labels = ["O", "B-PER", "I-PER"]
        dataset = Dataset.from_dict(
            {"tokens": [["John", "Smith"]], "ner_tags": [[1, 2]]},
            features=Features(
                {
                    "tokens": Sequence(Value("string")),
                    "ner_tags": Sequence(ClassLabel(names=labels)),
                }
            ),
        )

        examples = load_sequence_labeling_hf_dataset(
            dataset,
            tokens_column="tokens",
            labels_column="ner_tags",
            labels=labels,
        )

        self.assertEqual(examples[0].token_sequence, ["John", "Smith"])
        self.assertEqual(examples[0].label_sequence, ["B-PER", "I-PER"])

    def test_classic_dataset_presets_are_available(self):
        self.assertEqual(DATASET_PRESETS["sst2"].task, "classification")
        self.assertEqual(DATASET_PRESETS["stsb"].task, "regression")
        self.assertEqual(DATASET_PRESETS["conll2003"].task, "sequence_labeling")


class FineTuningFeatureTests(unittest.TestCase):
    def test_classification_features_support_character_inputs(self):
        examples = [ClassificationExample(0, ["hello"], None, "pos")]
        features = classification_features(
            examples,
            tokenizer=CharacterIndexer(),
            labels=["pos"],
            max_seq_length=5,
            is_character_model=True,
        )
        dataset = features_to_dataset(features, task="classification", is_character_model=True)

        self.assertEqual(dataset.tensors[0].shape, torch.Size([1, 5, 50]))
        self.assertEqual(dataset.tensors[1].tolist(), [[1, 1, 1, 0, 0]])

    def test_sequence_labeling_features_mask_wordpiece_labels(self):
        tokenizer = BertTokenizer.from_pretrained("pretrained-models/bert-base-uncased")
        examples = [SequenceLabelingExample(0, ["hello", "##s"], ["B-X", "I-X"])]
        features = sequence_labeling_features(
            examples,
            tokenizer=tokenizer,
            labels=["B-X", "I-X"],
            max_seq_length=5,
            is_character_model=False,
            pad_token_label_id=-100,
            pad_token_id=tokenizer.pad_token_id,
        )

        self.assertEqual(features[0].label_ids, [-100, 0, -100, -100, -100])


class FineTuningMetricTests(unittest.TestCase):
    def test_classification_metrics_match_micro_accuracy(self):
        metrics = classification_metrics([0, 1, 1, 0], [0, 1, 0, 0])
        self.assertEqual(metrics["accuracy"], 0.75)
        self.assertEqual(metrics["f1"], 0.75)

    def test_sequence_labeling_metrics_score_entities(self):
        labels = [["B-PER", "I-PER", "O", "B-LOC"]]
        predictions = [["B-PER", "I-PER", "O", "O"]]

        self.assertEqual(get_entities(labels), [("PER", 0, 1), ("LOC", 3, 3)])
        metrics = sequence_labeling_metrics(labels, predictions)
        self.assertEqual(metrics["precision"], 1.0)
        self.assertEqual(metrics["recall"], 0.5)
        self.assertAlmostEqual(metrics["f1"], 2 / 3)


if __name__ == "__main__":
    unittest.main()
