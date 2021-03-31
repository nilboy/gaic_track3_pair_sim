import fire
import json
import logging
import numpy as np
import os
from tqdm.auto import tqdm
logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

def get_all_files_from_model_name():
    input_data_dir = '../user_data/soft_labels/outputs'
    file_name_list = []
    for file_name in os.listdir(input_data_dir):
        file_name_list.append(os.path.join(input_data_dir, file_name))
    return file_name_list

def convert_single_file(input_file,
                        output_file, merge_results, train_rate):
    with open(output_file, 'w') as fout:
        with open(input_file) as fin:
            for line in tqdm(fin):
                record = json.loads(line)
                if merge_results[record['rid']]['dev']:
                    dev_score = np.mean(merge_results[record['rid']]['dev'])
                else:
                    dev_score = np.mean(merge_results[record['rid']]['train'])
                train_score = np.mean(merge_results[record['rid']]['train'])
                score = train_rate * train_score + (1.0 - train_rate) * dev_score
                record['labels'] = score
                fout.write(json.dumps(record, ensure_ascii=False) + '\n')
                record['text_a'], record['text_b'] = record['text_b'], record['text_a']
                fout.write(json.dumps(record, ensure_ascii=False) + '\n')

def construct_model_data(train_rate: float=0.55,
                         kfold_num: int=5):
    logger.info(f'construct model data...')
    file_name_list = get_all_files_from_model_name()
    merge_results = {}
    for file_name in file_name_list:
        logger.info(file_name)
        with open(file_name) as f:
            for line in f:
                record = json.loads(line)
                if record['rid'] not in merge_results:
                    merge_results[record['rid']] = {
                        'train': [],
                        'dev': []
                    }
                merge_results[record['rid']][record['mode']].append(record['score'])

    for kfold_id in range(0, kfold_num):
        input_file = f"../user_data/data/train_data/kfold/{kfold_id}/train.jsonl"
        output_file = f"../user_data/data/train_data/kfold/{kfold_id}/train_regression.jsonl"
        convert_single_file(input_file, output_file, merge_results, train_rate)
    convert_single_file("../user_data/data/train_data/train.jsonl",
                        "../user_data/data/train_data/train_regression.jsonl", merge_results, train_rate)

def construct_data(train_rate: float=0.55, kfold_num: int=5):
    os.makedirs("../user_data/soft_labels/regression", exist_ok=True)
    construct_model_data(train_rate=train_rate,
                         kfold_num=kfold_num)

if __name__ == '__main__':
    fire.Fire(construct_data)
