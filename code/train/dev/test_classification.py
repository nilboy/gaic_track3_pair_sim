from data_processor.record_processor import RecordProcessor

record_processor = RecordProcessor(
    "../user_data/data/normal_vocab.json",
    "../user_data/data/train_data/vocab.json"
)

import os

from simpletransformers_addons.models.sim_text.sim_text_model import SimTextArgs, SimTextModel

import json
from tqdm.auto import tqdm
import pandas as pd
from scipy.special import softmax
records = []

df = pd.read_csv("../tcdata/oppo_breeno_round1_data/testB.tsv",
                 delimiter='\t',
                 header=None)
for _, row in df.iterrows():
    text_a, text_b = record_processor.process_record(row[0], row[1])
    records.append([text_a, text_b])

batch_size = len(records)
results = [{} for _ in range(len(records))]

base_path = "../user_data"
sub_paths = []
model_types = [
    #"bert-base",
    #"bert-large",
    "nezha-base",
]
model_num = 4
for model_type in model_types:
    for kfold_id in range(0, 5):
        for model_id in range(0, model_num):
            sub_paths.append(f"classification/{model_type}/{kfold_id}-{model_id}/best")

sub_paths = ["classification/nezha-large/0-3/best"]

model_names = [os.path.join(base_path, sub_path) for sub_path in sub_paths]

for model_name in tqdm(model_names):
    if model_name in results[0]:
        continue
    try:
        model = SimTextModel('bert', model_name,
                            args={"eval_batch_size": 512, 'cache_dir': 'cache',
                                  "max_seq_length": 96, 'use_symmetric': False})
        for i in tqdm(range(0, len(records), batch_size)):
            cur_results = model.predict(records[i:i+batch_size], no_cache=False)
            scores = softmax(cur_results[1], 1)
            for j in range(len(scores)):
                results[i+j][model_name] = scores[i+j, 1]
    except:
        import traceback
        traceback.print_exc()
        print(f'{model_name} error.')
import numpy as np
variance = np.mean(np.var([[v for k, v in item.items()]for item in results], axis=1))
print(f'方差: {variance}')
results_merge = [np.mean([v for k, v in item.items()]) for item in results]

with open('../prediction_result/result.tsv', 'w') as fout:
    for value in results_merge:
        fout.write(f'{value}' + '\n')
