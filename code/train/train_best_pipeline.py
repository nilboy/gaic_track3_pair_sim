import fire
import os
import subprocess
import logging

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def train_pipeline(data_id,
                   task='classification',
                   train_file_mode='train',
                   model_names=None,
                   model_num=1):
    if model_names is None:
        model_names = os.listdir('../user_data/mlm')
    else:
        model_names = model_names.split(',')
    for model_name in model_names:
        for model_id in range(model_num):
            config_file = os.path.join('config', task, model_name, 'base.yaml')
            best_config_file = os.path.join('config', task, model_name, f'{model_id}.yaml')
            model_file = os.path.join('config', task, 'model.py')
            train_file = os.path.join('../user_data/data/train_data/kfold', str(data_id), f'{train_file_mode}.jsonl')
            dev_file = os.path.join('../user_data/data/train_data/kfold', str(data_id), 'dev.jsonl')
            output_dir = os.path.join('../user_data', task, model_name, f'{data_id}-{model_id}')
            os.makedirs(output_dir, exist_ok=True)
            run_command = ["python", "train/train_best_model.py",
                           "--config_file", config_file,
                           "--best_config_file", best_config_file,
                           "--model_file", model_file,
                           "--train_file", train_file,
                           "--dev_file", dev_file,
                           "--output_dir", output_dir]
            logger.info(" ".join(run_command))
            subprocess.run(run_command)


if __name__ == '__main__':
    fire.Fire(train_pipeline)
