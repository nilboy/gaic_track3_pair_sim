import os
import onnxruntime
from transformers import BertTokenizer
from fast_inference.engine import Engine
from data_processor.record_processor import RecordProcessor

class ModelPredictor(object):
    def __init__(self, model_dir):
        #so = onnxruntime.SessionOptions()
        #so.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.engine_32 = onnxruntime.InferenceSession(os.path.join(model_dir, 'model-32', 'model'),
                                                      #so,
                                                      providers=["CUDAExecutionProvider"])
        # self.engine_128 = onnxruntime.InferenceSession(os.path.join(model_dir, 'model-128', 'model'),
        #                                               providers=["CUDAExecutionProvider"])
        self.tokenizer = BertTokenizer.from_pretrained(model_dir)

        self.record_processor = RecordProcessor(
            os.path.join(model_dir, 'normal_vocab.json'),
            os.path.join(model_dir, 'vocab.json'))

    def get_inputs_dict(self, text_a, text_b,
                        max_length=32):
        model_inputs = self.tokenizer.batch_encode_plus([[text_a, text_b]],
                                                   return_tensors="pt", padding="max_length", truncation=True,
                                                   max_length=max_length)
        return {k: v.detach().cpu().numpy() for k, v in model_inputs.items()}

    def predict(self, text_a, text_b):
        text_a, text_b = self.record_processor.process_record(text_a, text_b)
        if True or len(text_a) + len(text_b) + 3 <= 32:
            inputs_dict = self.get_inputs_dict(text_a, text_b, max_length=32)
            engine = self.engine_32
        elif len(text_a) + len(text_b) + 3 <= 128:
            inputs_dict = self.get_inputs_dict(text_a, text_b, max_length=128)
            engine = self.engine_128
        elif len(text_a) + len(text_b) + 3 > 128:
            text_a = text_a[0:62]
            text_b = text_b[0:125 - len(text_a)]
            inputs_dict = self.get_inputs_dict(text_a, text_b, max_length=128)
            engine = self.engine_128
        return engine.run(None, inputs_dict)[0][0, 0]
