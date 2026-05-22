from collections.abc import Callable
from typing import Any

import torch

PADDING_VALUE = 0


def _make_bos_eos(
    character: int,
    padding_character: int,
    beginning_of_word_character: int,
    end_of_word_character: int,
    max_word_length: int,
) -> list[int]:
    char_ids = [padding_character] * max_word_length
    char_ids[0] = beginning_of_word_character
    char_ids[1] = character
    char_ids[2] = end_of_word_character
    return char_ids


def pad_sequence_to_length(
    sequence: list[Any],
    desired_length: int,
    default_value: Callable[[], Any] = lambda: 0,
    padding_on_right: bool = True,
) -> list[Any]:
    if padding_on_right:
        padded_sequence = sequence[:desired_length]
    else:
        padded_sequence = sequence[-desired_length:]

    values_to_pad = [default_value()] * (desired_length - len(padded_sequence))
    if padding_on_right:
        return padded_sequence + values_to_pad
    return values_to_pad + padded_sequence


class CharacterMapper:
    max_word_length = 50

    beginning_of_sentence_character = 256
    end_of_sentence_character = 257
    beginning_of_word_character = 258
    end_of_word_character = 259
    padding_character = 260
    mask_character = 261

    beginning_of_sentence_characters = _make_bos_eos(
        beginning_of_sentence_character,
        padding_character,
        beginning_of_word_character,
        end_of_word_character,
        max_word_length,
    )
    end_of_sentence_characters = _make_bos_eos(
        end_of_sentence_character,
        padding_character,
        beginning_of_word_character,
        end_of_word_character,
        max_word_length,
    )
    mask_characters = _make_bos_eos(
        mask_character,
        padding_character,
        beginning_of_word_character,
        end_of_word_character,
        max_word_length,
    )
    pad_characters = [PADDING_VALUE - 1] * max_word_length

    bos_token = "[CLS]"
    eos_token = "[SEP]"
    pad_token = "[PAD]"
    mask_token = "[MASK]"

    def __init__(self, tokens_to_add: dict[str, int] | None = None) -> None:
        self.tokens_to_add = tokens_to_add or {}

    def convert_word_to_char_ids(self, word: str) -> list[int]:
        if word in self.tokens_to_add:
            char_ids = [self.padding_character] * self.max_word_length
            char_ids[0] = self.beginning_of_word_character
            char_ids[1] = self.tokens_to_add[word]
            char_ids[2] = self.end_of_word_character
        elif word == self.bos_token:
            char_ids = self.beginning_of_sentence_characters
        elif word == self.eos_token:
            char_ids = self.end_of_sentence_characters
        elif word == self.mask_token:
            char_ids = self.mask_characters
        elif word == self.pad_token:
            char_ids = self.pad_characters
        else:
            word_encoded = word.encode("utf-8", "ignore")[: self.max_word_length - 2]
            char_ids = [self.padding_character] * self.max_word_length
            char_ids[0] = self.beginning_of_word_character
            for index, char_id in enumerate(word_encoded, start=1):
                char_ids[index] = char_id
            char_ids[len(word_encoded) + 1] = self.end_of_word_character

        return [char_id + 1 for char_id in char_ids]


class CharacterIndexer:
    def __init__(self, mapper: CharacterMapper | None = None) -> None:
        self.mapper = mapper or CharacterMapper()

    def tokens_to_indices(self, tokens: list[str]) -> list[list[int]]:
        return [self.mapper.convert_word_to_char_ids(token) for token in tokens]

    def as_padded_tensor(
        self,
        batch: list[list[str]],
        *,
        as_tensor: bool = True,
        max_length: int | None = None,
    ) -> torch.Tensor | list[list[list[int]]]:
        if max_length is None:
            max_length = max(map(len, batch))

        batch_indices = [self.tokens_to_indices(tokens) for tokens in batch]
        padded_batch = [
            pad_sequence_to_length(
                indices,
                max_length,
                default_value=lambda: [PADDING_VALUE] * self.mapper.max_word_length,
            )
            for indices in batch_indices
        ]
        if as_tensor:
            return torch.LongTensor(padded_batch)
        return padded_batch
