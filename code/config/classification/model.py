import math
from typing import Dict

from sklearn.metrics import accuracy_score
from simpletransformers_addons.models.sim_text.sim_text_model import SimTextArgs, SimTextModel
from simpletransformers_addons.model_wrappers.fgm_wrapper import FGMWrapper
from simpletransformers_addons.model_wrappers.pgd_wrapper import PGDWrapper

from sklearn.metrics import roc_curve, auc, accuracy_score

def manual_auc(labels, preds):
    labels = [1 if item >= 0.5 else 0 for item in labels]
    fpr, tpr, thresholds = roc_curve(labels, preds)
    auroc = auc(fpr, tpr)
    return auroc

def manual_accuracy(labels, preds):
    labels = [1 if item >= 0.5 else 0 for item in labels]
    preds = [1 if item >= 0.5 else 0 for item in preds]
    return accuracy_score(labels, preds)

def train(train_df, eval_df, model_args, aux_params=None):
    """单实例训练函数"""
    # model = SimTextModel("bert", aux_params.get("pretrain_model_path", "bert-base-chinese"),
    #                      num_labels=1, args=model_args)
    model = SimTextModel("bert", aux_params.get("pretrain_model_path", "bert-base-chinese"),
                         args=model_args)
    at_type = aux_params.get('at_type', 'pgd')
    if at_type != 'base':
        if at_type == 'fgm':
            wrapper = FGMWrapper
        elif at_type == 'pgd':
            wrapper = PGDWrapper
        model = wrapper(model, epsilon=aux_params.get('at_epsilon', 1.0),
                        alpha=aux_params.get('at_epsilon', 1.0)/3.3)

    model.train_model(train_df,
                      eval_df=eval_df)

def get_model_args(wandb_config: Dict,
                   original_model_args: Dict,
                   additional_config: Dict):
    """
        将wandb超参数转化为model_args形式参数.
    """
    model_args = SimTextArgs()
    model_args.update_from_dict(original_model_args)
    # Extracting the hyperparameter values
    cleaned_args = {}
    additional_args = {}
    layer_params = []
    param_groups = []
    for key, value in wandb_config.items():
        if key.startswith("layer_"):
            # These are layer parameters
            layer_keys = key.split("_")[-1]

            # Get the start and end layers
            start_layer = int(layer_keys.split("-")[0])
            end_layer = int(layer_keys.split("-")[-1])

            # Add each layer and its value to the list of layer parameters
            for layer_key in range(start_layer, end_layer):
                layer_params.append(
                    {"layer": layer_key, "lr": value,}
                )
        elif key.startswith("params_"):
            # These are parameter groups (classifier)
            params_key = "_".join(key.split("_")[1:])
            if params_key == 'embeddings':
                params_key_list = ["bert.embeddings.word_embeddings.weight",
                              "bert.embeddings.position_embeddings.weight",
                              "bert.embeddings.token_type_embeddings.weight",
                              "bert.embeddings.LayerNorm.weight",
                              "bert.embeddings.LayerNorm.bias"]
            elif params_key == 'pooler':
                params_key_list = ["bert.pooler.dense.weight",
                              "bert.pooler.dense.bias"]
            elif params_key == 'classifier':
                params_key_list = ["classifier.weight",
                              "classifier.bias"]
            else:
                raise ValueError(f'params_{params_key} not exist')
            for params_key in params_key_list:
                param_groups.append(
                    {
                        "params": [params_key],
                        "lr": value,
                        "weight_decay": model_args.weight_decay
                        if "bias" not in params_key
                        else 0.0,
                    }
                )
        elif key.startswith('additional_'):
            # 不直接添加到model_args的参数
            params_key = "_".join(key.split("_")[1:])
            additional_args[params_key] = value
        else:
            # Other hyperparameters (single value)
            cleaned_args[key] = value
    cleaned_args["custom_layer_parameters"] = layer_params
    cleaned_args["custom_parameter_groups"] = param_groups
    if 'batch_size' in additional_args:
        max_batch_size = additional_config['max_batch_size']
        batch_size = additional_args['batch_size']
        if batch_size > max_batch_size:
            cleaned_args['gradient_accumulation_steps'] = math.ceil(batch_size / max_batch_size)
            cleaned_args['train_batch_size'] = math.ceil(batch_size / cleaned_args['gradient_accumulation_steps'])
        else:
            cleaned_args['gradient_accumulation_steps'] = 1
            cleaned_args['train_batch_size'] = batch_size
        cleaned_args['evaluate_during_training_steps'] = int(original_model_args['evaluate_during_training_steps'] * max_batch_size / batch_size)
    aux_params = {}
    if 'at_type' in additional_args:
        aux_params['at_type'] = additional_args['at_type']
        aux_params['at_epsilon'] = additional_args['at_epsilon']
    if 'pretrain_model_path' in additional_args:
        aux_params['pretrain_model_path'] = additional_args['pretrain_model_path']
    model_args.update_from_dict(cleaned_args)
    return model_args, aux_params
