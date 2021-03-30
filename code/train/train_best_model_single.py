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
                   model_name='bert-base',
                   model_id=0):
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
    logger.info("构造soft-label...")
    model_batch_size = {
        'bert-base': 1024,
        'nezha-base': 1024,
        'bert-large': 512,
        'macbert-large': 512
    }
    run_command = ["python", "data_processor/construct_soft_label_single.py",
                   "--model_name", model_name,
                   "--batch_size", str(model_batch_size[model_name]),
                   "--kfold_id", str(data_id),
                   "--model_id", str(model_id)]
    logger.info(" ".join(run_command))
    subprocess.run(run_command)


if __name__ == '__main__':
    fire.Fire(train_pipeline)
