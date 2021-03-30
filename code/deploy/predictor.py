import os
import numpy as np
from transformers import BertTokenizer
from fast_inference.engine import Engine
from data_processor.record_processor import RecordProcessor

class ModelPredictor(object):
    def __init__(self, model_dir):
        self.engine_128 = Engine(os.path.join(model_dir,
                                             'model-128.engine'))
        self.tokenizer = BertTokenizer.from_pretrained(model_dir)

        self.record_processor = RecordProcessor(
            os.path.join(model_dir, 'normal_vocab.json'),
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
        if len(text_a) + len(text_b) + 3 <= 128:
            inputs_dict = self.get_inputs_dict(text_a, text_b, max_length=128)
            engine = self.engine_128
        elif len(text_a) + len(text_b) + 3 > 128:
            text_a = text_a[0:62]
            text_b = text_b[0:125 - len(text_a)]
            inputs_dict = self.get_inputs_dict(text_a, text_b, max_length=128)
            engine = self.engine_128
        return float(engine.run(inputs_dict)[0][1, 0])
