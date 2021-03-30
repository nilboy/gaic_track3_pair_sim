import os
import numpy as np
from transformers import BertTokenizerFast as BertTokenizer
from fast_inference.engine import Engine

import json
import logging
import time
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger()
logger.setLevel(logging.INFO)
w_logger =logging.getLogger("werkzeug")
w_logger.setLevel(logging.WARNING)

from data_processor.record_processor import RecordProcessor

class ModelPredictor(object):
    def __init__(self, model_dir):
        self.engine_128 = Engine(os.path.join(model_dir,
                                             'model-128.engine'))
        self.tokenizer = BertTokenizer.from_pretrained(model_dir)

        self.record_processor = RecordProcessor(
            os.path.join(model_dir, 'normal_vocab.json'),
            os.path.join(model_dir, 'idmap.json'),
            os.path.join(model_dir, 'vocab.json'))

    def get_inputs_dict(self, text_a, text_b,
                        max_length=32):
        model_inputs = self.tokenizer.batch_encode_plus([[text_a, text_b]],
                                                   return_tensors="pt", padding="max_length", truncation=True,
                                                   max_length=max_length)
        model_inputs = {k: v.detach().cpu().numpy() for k, v in model_inputs.items()}
        return {
                "input_ids": model_inputs['input_ids'].reshape(-1, 1),
                "input_mask": model_inputs['attention_mask'].reshape(-1, 1),
                "segment_ids":  model_inputs['token_type_ids'].reshape(-1, 1),
                "input_mask_fp32": np.asarray(model_inputs['attention_mask'].reshape(-1, 1), dtype=np.float32)
        }

    def predict(self, text_a, text_b):
        text_a, text_b = self.record_processor.process_record(text_a, text_b)
        if len(text_a) + len(text_b) + 3 <= 32:
            inputs_dict = self.get_inputs_dict(text_a, text_b, max_length=32)
            engine = self.engine_128
        elif len(text_a) + len(text_b) + 3 > 32:
            text_a = text_a[0:16]
            text_b = text_b[0:29 - len(text_a)]
            inputs_dict = self.get_inputs_dict(text_a, text_b, max_length=32)
            engine = self.engine_128
        return float(engine.run(inputs_dict)[0][1, 0])


import json
from ai_hub import inferServer


class AIHubInfer(inferServer):
    def __init__(self, model):
        super().__init__(model)

    # 数据前处理
    def pre_process(self, req_data):
        input_batch = {}
        input_batch["input"] = req_data.form.getlist("input")
        input_batch["index"] = req_data.form.getlist("index")

        return input_batch

    # 数据后处理，如无，可空缺
    def post_process(self, predict_data):
        response = json.dumps(predict_data)
        return response

    # 如需自定义，可覆盖重写
    def predict(self, preprocessed_data):
        input_list = preprocessed_data["input"]
        index_list = preprocessed_data["index"]

        response_batch = {}
        response_batch["results"] = []
        for i in range(len(index_list)):
            index_str = index_list[i]

            response = {}
            try:
                input_sample = input_list[i].strip()
                elems = input_sample.strip().split("\t")
                query_A = elems[0].strip()
                query_B = elems[1].strip()
                predict = infer(model, query_A, query_B)
                response["predict"] = predict
                response["index"] = index_str
                response["ok"] = True
            except Exception as e:
                response["predict"] = 0
                response["index"] = index_str
                response["ok"] = False
            response_batch["results"].append(response)

        return response_batch


# 需要根据模型类型重写
def infer(model, query_A, query_B):
    return model.predict(query_A, query_B)
    #return 1.0 - model.predict(query_A, query_B)


# 需要根据模型类型重写
def init_model(model_path):
    import time
    t1 = time.time()
    os.system("bash ./deploy/build_tensorrt_engine.sh")
    model = ModelPredictor("../user_data/ensemble/tensorrt")
    t2 = time.time()
    #
    print(model.predict('12 2954 16', '12 32 126 5951 456 16'))
    print(model.predict('426 160 8 9 172', '14973 8 9 337 155 156'))
    print(model.predict("12 19 212 213 29 106 172 162", "29 431 432 72 12 1046 16"))
    print(model.predict("10330 252 415 29 128 23 470 113 114 76", "29 19 1686 23 470"))
    #
    logger.info(f'initialize time: {(t2-t1)}')
    return model


if __name__ == "__main__":
    model = init_model(None)
    aihub_infer = AIHubInfer(model)
    aihub_infer.run(debuge=False)
