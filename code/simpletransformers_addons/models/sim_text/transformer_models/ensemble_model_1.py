import math
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
        self.device_num = torch.cuda.device_count()
        self.model_num = len(self.config.config_list)
        models = []
        for model_id in range(self.model_num):
            model_class = SUBMODEL_MAP[self.config.submodel_type_list[model_id]][0]
            config_class = SUBMODEL_MAP[self.config.submodel_type_list[model_id]][1]
            model_config = config_class.from_dict(self.config.config_list[model_id])
            models.append(model_class(model_config))
        self.models = nn.ModuleList(models)

    def get_model_device_id(self, model_id):
        if self.device_num > 1:
            # if model_id >= self.device_num:
            #     return 0
            # else:
            #     return model_id
            return self.device_num - (model_id % self.device_num) - 1
        else:
            return 0

    def move_model_to_device(self):
        if self.device_num > 1:
            for i in range(len(self.models)):
                self.models[i].to(f'cuda:{self.get_model_device_id(i)}')

    def split_run_forward(self, model, input_dict, split_num=2):
        return model.forward(**input_dict)
        # batch_size = input_dict['input_ids'].shape[0]
        # split_size = math.ceil(batch_size / split_num)
        # outputs_list = []
        # for i in range(split_num):
        #     input_dict_item = {k: v[i * split_size:(i+1)*split_size] for k, v in input_dict.items() if v is not None}
        #     outputs_list.append(model.forward(**input_dict_item))
        # sequence_outputs = torch.cat([item[0] for item in outputs_list], dim=0)
        # pooled_outputs = torch.cat([item[1] for item in outputs_list], dim=0)
        # return sequence_outputs, pooled_outputs

    def model_forward(self, model_id, input_dict):
        if self.device_num > 1:
            device_id = self.get_model_device_id(model_id)
            if device_id == 0:
                return self.split_run_forward(self.models[model_id], input_dict)
            else:
                input_dict = {k: v.to(f'cuda:{device_id}') for k, v in input_dict.items() if v is not None}
                outputs = self.split_run_forward(self.models[model_id], input_dict)
                return None, outputs[1]
        else:
            return self.models[model_id].forward(**input_dict)

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
        input_dict = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'token_type_ids': token_type_ids,
            'position_ids': position_ids
        }
        for i in range(self.model_num):
            outputs = self.model_forward(i, input_dict)
            pooled_outputs, sequence_outputs = outputs[1], outputs[0]
            pooled_outputs_list.append(pooled_outputs)
            sequence_outputs_list.append(sequence_outputs)
        if self.device_num > 1:
            pooled_outputs_list = [item.to('cuda:0') for item in pooled_outputs_list]
        pooled_outputs = ensemble_pooling(pooled_outputs_list)
        # sequence_outputs = ensemble_pooling(sequence_outputs_list)
        return None, pooled_outputs

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
