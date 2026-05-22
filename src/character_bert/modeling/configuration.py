from transformers import BertConfig


class CharacterBertConfig(BertConfig):
    model_type = "character_bert"

    def __init__(self, mlm_vocab_size: int | None = None, **kwargs):
        if mlm_vocab_size is not None:
            kwargs["vocab_size"] = mlm_vocab_size
        super().__init__(**kwargs)
        self.mlm_vocab_size = mlm_vocab_size or self.vocab_size
        self.tie_word_embeddings = False
