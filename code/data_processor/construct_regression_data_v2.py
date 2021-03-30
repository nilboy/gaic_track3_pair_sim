import fire
import json
import logging
import numpy as np
import os
from tqdm.auto import tqdm
logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

def get_all_files_from_model_name(model_name: str):
    input_data_dir = '../user_data/soft_labels/outputs'
    file_name_list = []
    for file_name in os.listdir(input_data_dir):
        if model_name == 'all' or file_name[0:-len("-4-1.jsonl")] == model_name:
            file_name_list.append(os.path.join(input_data_dir, file_name))
    return file_name_list

def construct_model_data(model_name: str,
                         train_rate: float=0.8):
    logger.info(f'construct model data: {model_name}')
    file_name_list = get_all_files_from_model_name(model_name)
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
    output_file = os.path.join("../user_data/soft_labels/regression", f"{model_name}-{train_rate}.jsonl")
    with open(output_file, 'w') as fout:
        with open("../user_data/data/train_data/train.jsonl") as fin:
        #with open("../user_data/data/train_data/kfold/0/train.jsonl") as fin:
            for line in tqdm(fin):
                record = json.loads(line)
                if merge_results[record['rid']]['dev']:
                    dev_score = np.mean(merge_results[record['rid']]['dev'])
                else:
                    dev_score = np.mean(merge_results[record['rid']]['train'])
                train_score = np.mean(merge_results[record['rid']]['train'])
                score = train_rate * train_score + (1.0 - train_rate) * dev_score
                # score_list = merge_results[record['rid']]['train'] + merge_results[record['rid']]['dev']
                # score = np.mean(score_list)
                record['labels'] = score
                fout.write(json.dumps(record, ensure_ascii=False) + '\n')
                record['text_a'], record['text_b'] = record['text_b'], record['text_a']
                fout.write(json.dumps(record, ensure_ascii=False) + '\n')


def construct_data(model_names="bert_base,chinese-bert-wwm-ext,bert_large,chinese-roberta-wwm-ext-large",
                   train_rate: float=0.8):
    os.makedirs("../user_data/soft_labels/regression", exist_ok=True)
    model_names = model_names.split(',')
    model_names.append('all')
    for model_name in model_names:
        construct_model_data(model_name, train_rate=train_rate)

if __name__ == '__main__':
    fire.Fire(construct_data)
