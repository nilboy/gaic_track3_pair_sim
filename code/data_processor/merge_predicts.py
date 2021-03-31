import json
import os
from tqdm.auto import tqdm

def merge_predicts(input_data_dir,
                   output_file):
    merge_results = {}
    for file_name in tqdm(os.listdir(input_data_dir)):
        with open(os.path.join(input_data_dir, file_name)) as f:
            for line in f:
                record = json.loads(line)
                if record['rid'] not in merge_results:
                    merge_results[record['rid']] = {
                        'train': [],
                        'dev': []
                    }
                merge_results[record['rid']][record['mode']].append(record['score'])
    with open(output_file, 'w') as fout:
        json.dump(merge_results, fout, ensure_ascii=False, indent=2)

if __name__ == '__main__':
    merge_predicts('../user_data/soft_labels/outputs', '../user_data/data/train_data/merge_results.json')
