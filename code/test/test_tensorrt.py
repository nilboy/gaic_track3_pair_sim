import time
from tqdm.auto import tqdm
from deploy.predictor import ModelPredictor

t1 = time.time()
model = ModelPredictor("../user_data/ensemble/tensorrt")
t2 = time.time()
print(f'initial time: {t2-t1}')
with open('../prediction_result/result.tsv', 'w') as fout:
    for line in tqdm(open('../tcdata/oppo_breeno_round1_data/testB.tsv')):
        text_a, text_b = line.strip().split('\t')
        score = model.predict(text_a, text_b)
        fout.write(f'{score}\n')
