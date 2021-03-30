from typing import List, Any, Dict
from transformers.configuration_utils import PretrainedConfig
from simpletransformers_addons.transformers.bert.modeling_bert import *
from simpletransformers_addons.transformers.bert.modeling_nezha import NeZhaConfig, NeZhaModel
from transformers.utils import logging

logger = logging.get_logger(__name__)

SUBMODEL_MAP = {
    "bert": (BertModel, BertConfig),
    "nezha": (NeZhaModel, NeZhaConfig)
}

class EnsembleModelConfig(PretrainedConfig):
    model_type = 'ensemble_model'

    def __init__(self, config_list: List[Dict]=None,
                 submodel_type_list: List[str]=None,
                 **kwargs):
        super(EnsembleModelConfig, self).__init__(**kwargs)
        self.config_list = config_list
        self.submodel_type_list = submodel_type_list

def ensemble_pooling(inputs: List[Any]):
    """pooling ensemble model"""
    # return torch.max(torch.stack(inputs, dim=-1), dim=-1)[0]
    return torch.cat(inputs, dim=-1)

class EnsembleModel(BertPreTrainedModel):
    config_class = EnsembleModelConfig
    def __init__(self, config):
        super().__init__(config)
        self.model_num = len(self.config.config_list)
        models = []
        for model_id in range(self.model_num):
            model_class = SUBMODEL_MAP[self.config.submodel_type_list[model_id]][0]
            config_class = SUBMODEL_MAP[self.config.submodel_type_list[model_id]][1]
            model_config = config_class.from_dict(self.config.config_list[model_id])
            models.append(model_class(model_config))
        self.models = nn.ModuleList(models)

    def forward(
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
        pooled_outputs_list, sequence_outputs_list = [], []
        for i in range(self.model_num):
            outputs = self.models[i].forward(input_ids=input_ids,
                                   attention_mask=attention_mask,
                                   token_type_ids=token_type_ids,
                                   position_ids=position_ids,
                                   head_mask=head_mask,
                                   inputs_embeds=inputs_embeds,
                                   encoder_hidden_states=encoder_hidden_states,
                                   encoder_attention_mask=encoder_attention_mask,
                                   past_key_values=past_key_values,
                                   use_cache=use_cache,
                                   output_attentions=output_attentions,
                                   output_hidden_states=output_hidden_states,
                                   return_dict=return_dict)
            pooled_outputs, sequence_outputs = outputs[1], outputs[0]
            pooled_outputs_list.append(pooled_outputs)
            sequence_outputs_list.append(sequence_outputs)
        pooled_outputs = ensemble_pooling(pooled_outputs_list)
        sequence_outputs = ensemble_pooling(sequence_outputs_list)
        return sequence_outputs, pooled_outputs

    @staticmethod
    def init_from_models_path(models_path: List[str],
                              submodel_type_list: List[str]):
        sub_config_list = []
        sub_models = []
        model_num = len(models_path)
        for model_id in range(model_num):
            model_path = models_path[model_id]
            config_class = SUBMODEL_MAP[submodel_type_list[model_id]][1]
            model_class = SUBMODEL_MAP[submodel_type_list[model_id]][0]
            sub_config_list.append(config_class.from_pretrained(model_path).to_dict())
            sub_models.append(model_class.from_pretrained(model_path))
        ensemble_config = EnsembleModelConfig(sub_config_list,
                                                           submodel_type_list)
        ensemble_config.update(sub_config_list[0])
        ensemble_config.hidden_size = 0
        for sub_config in sub_config_list:
            ensemble_config.hidden_size += sub_config['hidden_size']
        ensemble_model = EnsembleModel(ensemble_config)
        for model_id in range(model_num):
            ensemble_model.models[model_id].load_state_dict(sub_models[model_id].state_dict())
        return ensemble_model
