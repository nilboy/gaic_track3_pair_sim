import copy
import json
import os
from typing import List

from transformers.configuration_utils import PretrainedConfig
from transformers.models.bert.modeling_bert import *
from transformers.utils import logging

from simpletransformers_addons.models.sim_text.transformer_models.bert_model import BertForSimText

logger = logging.get_logger(__name__)

class EnsembleBertConfig(PretrainedConfig):
    model_type = "ensemble_bert"

    def __init__(
        self,
        vocab_size=30522,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=2,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        pad_token_id=0,
        gradient_checkpointing=False,
        position_embedding_type="absolute",
        use_cache=True,
        num_bottom_hidden_layers=8,
        num_top_hidden_layers=4,
        num_models=5,
        num_labels=2,
        temperatures=None,
        **kwargs
    ):
        super().__init__(pad_token_id=pad_token_id, **kwargs)

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.gradient_checkpointing = gradient_checkpointing
        self.position_embedding_type = position_embedding_type
        self.use_cache = use_cache
        self.num_bottom_hidden_layers = num_bottom_hidden_layers
        self.num_top_hidden_layers = num_top_hidden_layers
        self.num_models = num_models
        self.num_labels = num_labels
        if temperatures is None:
            self.temperatures = [1.0] * self.num_models
        else:
            self.temperatures = temperatures


class EnsembleBertModel(BertPreTrainedModel):
    config_class = EnsembleBertConfig
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        # bottom bert
        bottom_config = copy.deepcopy(config)
        bottom_config.num_hidden_layers = config.num_bottom_hidden_layers
        top_config = copy.deepcopy(config)
        top_config.num_hidden_layers = config.num_top_hidden_layers
        self.embeddings = BertEmbeddings(config)
        self.bottom_encoder = BertEncoder(bottom_config)
        self.top_encoder = nn.ModuleList([BertEncoder(top_config) for _ in range(config.num_models)])
        self.pooler = nn.ModuleList([BertPooler(config) for _ in range(config.num_models)])
        self.dropout = nn.ModuleList([nn.Dropout(config.hidden_dropout_prob) for _ in range(config.num_models)])
        self.classifier = nn.ModuleList([nn.Linear(config.hidden_size, config.num_labels) for _ in range(config.num_models)])

    def forward_inner(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.config.is_decoder:
            use_cache = use_cache if use_cache is not None else self.config.use_cache
        else:
            use_cache = False

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            batch_size, seq_length = input_shape
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            batch_size, seq_length = input_shape
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if attention_mask is None:
            attention_mask = torch.ones(((batch_size, seq_length + past_key_values_length)), device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape, device)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
        )
        #
        bottom_outputs = self.bottom_encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        bottom_outputs = bottom_outputs[0]
        pooled_output_list = []
        for i in range(self.config.num_models):
            encoder_outputs = self.top_encoder[i](
                bottom_outputs,
                attention_mask=extended_attention_mask,
                head_mask=head_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_extended_attention_mask,
                past_key_values=past_key_values,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict)
            sequence_output = encoder_outputs[0]
            pooled_output = self.pooler[i](sequence_output)
            pooled_output_list.append(pooled_output)
        return pooled_output_list

    def forward(self,
                input_ids_a=None,
                attention_mask_a=None,
                token_type_ids_a=None,
                input_ids_b=None,
                attention_mask_b=None,
                token_type_ids_b=None):
        pooled_output_list_a = self.forward_inner(input_ids=input_ids_a,
                                                  attention_mask=attention_mask_a,
                                                  token_type_ids=token_type_ids_a)
        pooled_output_list_b = self.forward_inner(input_ids=input_ids_b,
                                                  attention_mask=attention_mask_b,
                                                  token_type_ids=token_type_ids_b)
        scores_list = []
        for i in range(self.config.num_models):
            pooled_output = (pooled_output_list_a[i] + pooled_output_list_b[i]) / 2.0
            pooled_output = self.dropout[i](pooled_output)
            logits = self.classifier[i](pooled_output)
            logits /= self.config.temperatures[i]
            scores = torch.softmax(logits, dim=1)
            scores_list.append(scores)
        mean_scores = torch.mean(torch.stack(scores_list, 0), 0)
        return mean_scores

    @staticmethod
    def init_from_models_path(models_path: List[str],
                              sub_model_class=BertForSimText,
                              num_bottom_hidden_layers=8,
                              num_top_hidden_layers=4,
                              num_labels=2,
                              temperatures=None):
        num_models = len(models_path)
        logger.info('load model_config')
        bert_config = BertConfig.from_pretrained(models_path[0])
        bert_config_dict = bert_config.to_dict()
        bert_config_dict = {k: v for k, v in bert_config_dict.items() if k not in ['architectures']}
        model_config = EnsembleBertConfig.from_dict(bert_config_dict)
        if temperatures is None:
            temperatures = []
            for model_path in models_path:
                model_args_file = os.path.join(model_path, 'model_args.json')
                if os.path.exists(model_args_file):
                    temperatures.append(json.load(open(model_args_file)).get('temperature', 1.0))
                else:
                    temperatures.append(1.0)
        model_config.num_bottom_hidden_layers = num_bottom_hidden_layers
        model_config.num_top_hidden_layers = num_top_hidden_layers
        model_config.num_models = num_models
        model_config.num_labels = num_labels
        model_config.temperatures = temperatures
        logger.info('build ensemble model...')
        ensemble_model = EnsembleBertModel(model_config)
        logger.info('load sub models...')
        sub_models = [sub_model_class.from_pretrained(model_path, config=BertConfig.from_pretrained(model_path), temperature=1.0)
                      for model_path in models_path]
        logger.info('load params...')
        ensemble_model.embeddings.load_state_dict(sub_models[0].bert.embeddings.state_dict())
        # set bottom_encoder
        for layer_id in range(model_config.num_bottom_hidden_layers):
            ensemble_model.bottom_encoder.layer[layer_id].load_state_dict(
                sub_models[0].bert.encoder.layer[layer_id].state_dict())
        # set top_models
        for model_id in range(num_models):
            for layer_id in range(model_config.num_top_hidden_layers):
                layer_state_dict = sub_models[model_id].bert.encoder.layer[layer_id + \
                                                                           model_config.num_bottom_hidden_layers].state_dict()
                ensemble_model.top_encoder[model_id].layer[layer_id].load_state_dict(layer_state_dict)
            ensemble_model.pooler[model_id].load_state_dict(sub_models[model_id].bert.pooler.state_dict())
            ensemble_model.classifier[model_id].load_state_dict(sub_models[model_id].classifier.state_dict())
        return ensemble_model

