import json
import logging

import os
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
    # return model.predict(query_A, query_B)
    return 0.5


# 需要根据模型类型重写
def init_model(model_path):
    logging.getLogger().setLevel(logging.WARNING)
    return {"empty_model": None}


if __name__ == "__main__":
    model = init_model(None)
    aihub_infer = AIHubInfer(model)
    aihub_infer.run(debuge=False)
