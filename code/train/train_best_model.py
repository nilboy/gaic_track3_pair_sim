import fire
import logging
import os
import pandas as pd
from ruamel import yaml

import os
import sys
from importlib import import_module

def load_module_from_path(path):
    """loads module from path.
    Args:
        path: module path.
    Returns:
        A Python module.
    """
    if not os.path.exists(path):
        raise ValueError(f"not found path: {path}")
    dirname, filename = os.path.split(path)
    module_name, _ = os.path.splitext(filename)
    sys.path.insert(0, os.path.abspath(dirname))
    module = import_module(module_name)
    sys.path.pop(0)
    return module

logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

def train(config_file,
          best_config_file,
          model_file,
          train_file,
          dev_file,
          output_dir):
    global_config = yaml.load(open(config_file))
    best_config = yaml.load(open(best_config_file))
    train_df = pd.read_json(train_file, lines=True)
    eval_df = pd.read_json(dev_file, lines=True)
    eval_df = eval_df.iloc[0:8000]
    model = load_module_from_path(model_file)
    train_fn, get_model_args = model.train, model.get_model_args
    logging.info('start training...')
    # Get sweep hyperparameters
    args = {
        key: value['value']
        for key, value in best_config.items()
        if key not in ["_wandb", "wandb_version"]
    }
    logging.info('construct model args...')
    model_args, aux_params = get_model_args(args, original_model_args=global_config['model_args'],
                                additional_config=global_config['additional_config'])
    os.makedirs(output_dir, exist_ok=True)
    model_args.output_dir = os.path.join(output_dir, 'outputs')
    model_args.tensorboard_dir = os.path.join(output_dir, 'runs')
    model_args.best_model_dir = os.path.join(output_dir, 'best')
    model_args.cache_dir = os.path.join(output_dir, 'cache')
    model_args.wandb_project = None
    model_args.no_save = False

    model_args.save_steps = 0
    model_args.save_eval_checkpoints = False
    model_args.save_model_every_epoch = False
    logging.info(f'random seed: {model_args.manual_seed}')
    train_fn(train_df, eval_df, model_args, aux_params=aux_params)

if __name__ == '__main__':
    fire.Fire(train)
