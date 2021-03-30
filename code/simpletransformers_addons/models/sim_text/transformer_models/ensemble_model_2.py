from typing import List, Any, Dict
from transformers.configuration_utils import PretrainedConfig
from simpletransformers_addons.transformers.bert.modeling_bert import *
from simpletransformers_addons.transformers.bert.modeling_nezha import NeZhaConfig, NeZhaModel
from transformers.utils import logging
import threading

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

    def move_model_to_device(self):
        if self.device_num > 1:
            for i in range(len(self.models)):
                self.models[i].to(f'cuda:{self.get_model_device_id(i)}')

    def get_model_device_id(self, model_id):
        if self.device_num > 1:
            return self.device_num - (model_id % self.device_num) - 1
        else:
            return 0

    def scatter(self, inputs):
        inputs_list, devices = [], []
        for model_id in range(self.model_num):
            device_id = self.get_model_device_id(model_id)
            device_inputs = {k: v.to(f'cuda:{device_id}') for k, v in inputs.items() if v is not None}
            inputs_list.append(device_inputs)
            devices.append(f'cuda:{device_id}')
        return tuple(inputs_list), devices

    def parallel_apply(self, models, input_dict_list, devices):
        lock = threading.Lock()
        results = {}

        def _worker(i, module, input_dict, device=None):
            # torch.set_grad_enabled(grad_enabled)
            try:
                with torch.cuda.device(device):
                    output = module(**input_dict)[1]
                output = output.to('cuda:0')
                with lock:
                    results[i] = output
            except Exception as e:
                with lock:
                    results[i] = e

        threads = [threading.Thread(target=_worker,
                                    args=(i, module, input_dict, device))
                   for i, (module, input_dict, device) in
                   enumerate(zip(models, input_dict_list, devices))]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        outputs = []
        for i in range(len(input_dict_list)):
            output = results[i]
            if isinstance(output, Exception):
                raise output
            outputs.append(output)
        return outputs


    def models_forward(self, input_dict):
        pooled_outputs_list = []
        if self.device_num <= 1:
            for i in range(self.model_num):
                outputs = self.models[i].forward(**input_dict)
                pooled_outputs, sequence_outputs = outputs[1], outputs[0]
                pooled_outputs_list.append(pooled_outputs)
        else:
            input_dict_list, devices = self.scatter(input_dict)
            outputs = self.parallel_apply(self.models, input_dict_list, devices)
            for item in outputs:
                pooled_outputs_list.append(item)
        return pooled_outputs_list


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
        input_dict = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'token_type_ids': token_type_ids,
            'position_ids': position_ids
        }
        pooled_outputs_list = self.models_forward(input_dict)
        pooled_outputs = ensemble_pooling(pooled_outputs_list)
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
