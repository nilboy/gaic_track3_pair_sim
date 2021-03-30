import json
import numpy as np
import os
from tqdm.auto import tqdm

def process_data(data_base_dir,
                 include_dev=True,
                 kfold_num=5):
    with open(os.path.join(data_base_dir, 'merge_results.json')) as f:
        merge_results = json.load(f)
    for kfold_id in tqdm(range(kfold_num)):
        for mode in ['train_enhanced']:
            if include_dev:
                output_data_file = os.path.join(data_base_dir,
                                       'kfold', str(kfold_id),
                                       f'{mode}_regress_dev.jsonl')
            else:
                output_data_file = os.path.join(data_base_dir,
                                       'kfold', str(kfold_id),
                                       f'{mode}_regress.jsonl')
            input_data_file = os.path.join(data_base_dir,
                                   'kfold', str(kfold_id),
                                   f'{mode}.jsonl')
            with open(output_data_file, 'w') as fout:
                with open(input_data_file) as fin:
                    for line in tqdm(fin):
                        record = json.loads(line)
                        if include_dev:
                            score_list = merge_results[record['rid']]['train'] + merge_results[record['rid']]['dev']
                        else:
                            score_list = merge_results[record['rid']]['train']
                        score = np.mean(score_list)
                        record['labels'] = score
                        fout.write(json.dumps(record, ensure_ascii=False) + '\n')
    # construct full data
    input_data_file = os.path.join(data_base_dir,
                                   "train.jsonl")
    if include_dev:
        output_data_file = os.path.join(data_base_dir,
                                        "train_enhanced_regress_dev.jsonl")
    else:
        output_data_file = os.path.join(data_base_dir,
                                        "train_enhanced_regress.jsonl")

    with open(output_data_file, 'w') as fout:
        with open(input_data_file) as fin:
            for line in tqdm(fin):
                record = json.loads(line)
                if include_dev:
                    score_list = merge_results[record['rid']]['train'] + merge_results[record['rid']]['dev']
                else:
                    score_list = merge_results[record['rid']]['train']
                score = np.mean(score_list)
                record['labels'] = score
                fout.write(json.dumps(record, ensure_ascii=False) + '\n')
                record['text_a'], record['text_b'] = record['text_b'], record['text_a']
                fout.write(json.dumps(record, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    process_data('../user_data/data/train_data',
                 include_dev=True)
    # process_data('../user_data/data/train_data',
    #              include_dev=False)
