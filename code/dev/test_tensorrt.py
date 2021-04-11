import sys
import os
import numpy as np
from transformers import BertTokenizer

def get_inputs_dict(text_a, text_b,
                    tokenizer,
                   max_length=128):
    model_inputs = tokenizer.batch_encode_plus([[text_a, text_b]],
                                               return_tensors="pt", padding="max_length", truncation=True,
                                               max_length=max_length)
    outputs = {k: v.detach().cpu().numpy() for k, v in model_inputs.items()}
    return {
        "input_ids": outputs['input_ids'].reshape(-1, 1),
        "input_mask": outputs['attention_mask'].reshape(-1, 1),
        "segment_ids":  outputs['token_type_ids'].reshape(-1, 1),
        "input_mask_fp32": np.asarray(outputs['attention_mask'].reshape(-1, 1), dtype=np.float32)
    }

from fast_inference.engine import Engine
from tqdm.auto import tqdm
engine_32 = Engine("../user_data/ensemble/tensorrt/model-128.engine")
tokenizer = BertTokenizer.from_pretrained('../user_data/ensemble/tensorrt')

import time
import json
t1 = time.time()
results = []
labels = []
for line in tqdm(open("../user_data/data/train_data/kfold/0/dev.jsonl")):
    record = json.loads(line)
    inputs_numpy = get_inputs_dict(record['text_a'], record['text_b'], tokenizer)
    outputs = engine_32.run(inputs_numpy)
    results.append(outputs[0][1, 0])
    labels.append(record['labels'])

t2 = time.time()
print(t2-t1)

from sklearn.metrics import roc_curve, auc, accuracy_score

def manual_auc(labels, preds):
    labels = [1 if item >= 0.5 else 0 for item in labels]
    fpr, tpr, thresholds = roc_curve(labels, preds)
    auroc = auc(fpr, tpr)
    return auroc

print(manual_auc(labels, results))
