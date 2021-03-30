import fire
import json
import logging
import os
from scipy.special import softmax

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

from simpletransformers_addons.models.sim_text.sim_text_model import SimTextArgs, SimTextModel

def predict_label(model_name,
                  batch_size,
                  kfold_id,
                  model_id,
                  data_base_dir,
                  model_base_dir,
                  output_dir,
                  task):
    logger.info(f'model_name: {model_name}; kfold_id: {kfold_id}; model_id {model_id}')
    os.makedirs(output_dir, exist_ok=True)
    model = SimTextModel('bert',
                         os.path.join(model_base_dir, model_name, f'{kfold_id}-{model_id}', 'best'),
                         args={"eval_batch_size": batch_size,
                               "max_seq_length": 32})
    records, record_meta_infos = [], []
    for mode in ['train', 'dev']:
        with open(os.path.join(data_base_dir, 'kfold', str(kfold_id),
                               f'{mode}.jsonl')) as fin:
            for line in fin:
                record = json.loads(line)
                records.append([record['text_a'], record['text_b']])
                record_meta_infos.append({
                    'rid': record['rid'],
                    'mode': mode
                })
    results = model.predict(records)
    if task == 'classification':
        scores = softmax(results[1], 1)
        for i in range(len(scores)):
            record_meta_infos[i]['score'] = scores[i][1]
    else:
        scores = results[0]
        for i in range(len(scores)):
            record_meta_infos[i]['score'] = scores[i]

    with open(os.path.join(output_dir, f'{model_name}-{kfold_id}-{model_id}.jsonl'), 'w') as fout:
        for item in record_meta_infos:
            fout.write(json.dumps(item, ensure_ascii=False) + '\n')

def construct_soft_label(model_name=None,
                         task="classification",
                         batch_size=256,
                         kfold_id=0,
                         model_id=0):
    data_base_dir = '../user_data/data/train_data'
    model_base_dir = f'../user_data/{task}'
    output_dir = f'../user_data/soft_labels/{task}'
    predict_label(model_name,
                  batch_size,
                  kfold_id,
                  model_id,
                  data_base_dir,
                  model_base_dir,
                  output_dir,
                  task=task)

if __name__ == '__main__':
    fire.Fire(construct_soft_label)
