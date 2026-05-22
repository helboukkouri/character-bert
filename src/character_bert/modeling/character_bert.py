import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import BertConfig
from transformers.masking_utils import create_bidirectional_mask, create_causal_mask
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions
from transformers.models.bert.modeling_bert import (
    BertEncoder,
    BertForPreTrainingOutput,
    BertPooler,
    BertPreTrainedModel,
    BertPreTrainingHeads,
)
from transformers.pytorch_utils import apply_chunking_to_forward

from character_bert.modeling.character_cnn import CharacterCNN
from character_bert.modeling.configuration import CharacterBertConfig


class CharacterBertEmbeddings(nn.Module):
    def __init__(self, config: BertConfig) -> None:
        super().__init__()
        self.word_embeddings = CharacterCNN(output_dim=config.hidden_size)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.register_buffer(
            "position_ids",
            torch.arange(config.max_position_embeddings).expand((1, -1)),
            persistent=False,
        )

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        token_type_ids: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if input_ids is not None:
            input_shape = input_ids[:, :, 0].size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        sequence_length = input_shape[1]
        if position_ids is None:
            position_ids = torch.arange(
                sequence_length,
                device=self.position_ids.device,
            ).unsqueeze(0)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=position_ids.device)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)

        embeddings = (
            inputs_embeds
            + self.position_embeddings(position_ids)
            + self.token_type_embeddings(token_type_ids)
        )
        embeddings = self.LayerNorm(embeddings)
        return self.dropout(embeddings)


class CharacterBertModel(BertPreTrainedModel):
    config_class = CharacterBertConfig
    base_model_prefix = "character_bert"
    main_input_name = "input_ids"

    def __init__(self, config: BertConfig) -> None:
        super().__init__(config)
        self.config = config
        self.embeddings = CharacterBertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)
        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        token_type_ids: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        head_mask: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        encoder_hidden_states: torch.Tensor | None = None,
        encoder_attention_mask: torch.Tensor | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
    ):
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        if input_ids is not None:
            input_shape = input_ids[:, :, 0].size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        output_attentions = (
            output_attentions if output_attentions is not None else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )

        device = input_ids.device if input_ids is not None else inputs_embeds.device
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
        )

        if self.config.is_decoder:
            attention_mask = create_causal_mask(
                config=self.config,
                inputs_embeds=embedding_output,
                attention_mask=attention_mask,
                past_key_values=None,
            )
        else:
            attention_mask = create_bidirectional_mask(
                config=self.config,
                inputs_embeds=embedding_output,
                attention_mask=attention_mask,
            )

        if encoder_attention_mask is not None:
            encoder_attention_mask = create_bidirectional_mask(
                config=self.config,
                inputs_embeds=embedding_output,
                attention_mask=encoder_attention_mask,
                encoder_hidden_states=encoder_hidden_states,
            )

        sequence_output, hidden_states, attentions, cross_attentions = self._encode(
            embedding_output,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        pooled_output = self.pooler(sequence_output)

        if not return_dict:
            output = (sequence_output, pooled_output)
            if output_hidden_states:
                output += (hidden_states,)
            if output_attentions:
                output += (attentions,)
            return output

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            past_key_values=None,
            hidden_states=hidden_states,
            attentions=attentions,
            cross_attentions=cross_attentions,
        )

    def _encode(
        self,
        hidden_states: torch.Tensor,
        *,
        attention_mask: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None,
        encoder_attention_mask: torch.Tensor | None,
        output_attentions: bool,
        output_hidden_states: bool,
    ) -> tuple[
        torch.Tensor,
        tuple[torch.Tensor, ...] | None,
        tuple[torch.Tensor, ...] | None,
        tuple[torch.Tensor, ...] | None,
    ]:
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None

        for layer_module in self.encoder.layer:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if output_attentions:
                self_attention_output, attention_probs = layer_module.attention(
                    hidden_states,
                    attention_mask,
                )
                attention_output = self_attention_output
                all_self_attentions += (attention_probs,)

                if layer_module.is_decoder and encoder_hidden_states is not None:
                    if not hasattr(layer_module, "crossattention"):
                        raise ValueError(
                            "Decoder layers need cross-attention modules when "
                            "`encoder_hidden_states` is provided."
                        )
                    cross_attention_output, cross_attention_probs = layer_module.crossattention(
                        attention_output,
                        None,
                        encoder_hidden_states,
                        encoder_attention_mask,
                    )
                    attention_output = cross_attention_output
                    all_cross_attentions += (cross_attention_probs,)

                hidden_states = apply_chunking_to_forward(
                    layer_module.feed_forward_chunk,
                    layer_module.chunk_size_feed_forward,
                    layer_module.seq_len_dim,
                    attention_output,
                )
            else:
                hidden_states = layer_module(
                    hidden_states,
                    attention_mask,
                    encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                )

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        return hidden_states, all_hidden_states, all_self_attentions, all_cross_attentions


class CharacterBertForPreTraining(BertPreTrainedModel):
    config_class = CharacterBertConfig
    base_model_prefix = "character_bert"

    def __init__(self, config: BertConfig) -> None:
        super().__init__(config)
        if hasattr(config, "mlm_vocab_size"):
            config.vocab_size = config.mlm_vocab_size
        config.tie_word_embeddings = False
        self.character_bert = CharacterBertModel(config)
        self.cls = BertPreTrainingHeads(config)
        self.post_init()

    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    def set_output_embeddings(self, new_embeddings):
        self.cls.predictions.decoder = new_embeddings

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        token_type_ids: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        head_mask: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        next_sentence_label: torch.Tensor | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.character_bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output, pooled_output = outputs[:2]
        prediction_scores, seq_relationship_score = self.cls(sequence_output, pooled_output)

        total_loss = None
        if labels is not None and next_sentence_label is not None:
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(
                prediction_scores.view(-1, self.config.vocab_size),
                labels.view(-1),
            )
            next_sentence_loss = loss_fct(
                seq_relationship_score.view(-1, 2),
                next_sentence_label.view(-1),
            )
            total_loss = masked_lm_loss + next_sentence_loss

        if not return_dict:
            output = (prediction_scores, seq_relationship_score) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        return BertForPreTrainingOutput(
            loss=total_loss,
            prediction_logits=prediction_scores,
            seq_relationship_logits=seq_relationship_score,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
