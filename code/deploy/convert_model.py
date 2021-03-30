import fire
import os
import logging
from transformers import BertTokenizer
from simpletransformers_addons.models.sim_text.transformer_models.bert_model import BertForSimText
from simpletransformers_addons.models.sim_text.transformer_models.ensemble_model import EnsembleModelConfig

from fast_inference.convert_onnx_to_tensorrt import convert_onnx_to_tensorrt
from fast_inference.convert_pt_to_onnx import convert_pt_to_onnx

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def get_inputs_dict(tokenizer,
                   max_length=32):
    model_inputs = tokenizer.batch_encode_plus([['人见人爱好多好多个'*12, '花见花开'*12]],
                                               return_tensors="pt", padding="max_length", truncation=True,
                                               max_length=max_length)
    return {k: v.detach().cpu().numpy() for k, v in model_inputs.items()}

def convert_pytorch_to_onnx(pytorch_model_path,
                            onnx_model_path,
                            tokenizer,
                            max_length,
                            use_external_data_format=False):
    inputs_dict = get_inputs_dict(tokenizer, max_length=max_length)
    config = EnsembleModelConfig.from_pretrained(pytorch_model_path)
    model = BertForSimText.from_pretrained(pytorch_model_path,
                                           config=config,
                                           temperature=1.0,
                                           use_bimodel=False,
                                           submodel_type="ensemble")
    model.eval()
    convert_pt_to_onnx(model,
                       inputs_dict,
                       input_names=["input_ids", "attention_mask", "token_type_ids"],
                       output_names=["output"],
                       output_model_path=onnx_model_path,
                       use_external_data_format=use_external_data_format)

def convert_model(model_path='../user_data/regression-all/ensemble/outputs',
                  output_model_path='../user_data/deploy-model',
                  max_lengths="32"):
    tokenizer = BertTokenizer.from_pretrained(model_path)

    max_length_items = [int(item) for item in max_lengths.split(",")]
    for max_length in max_length_items:
        os.makedirs(os.path.join(output_model_path, 'onnx', f'model-{max_length}'), exist_ok=True)
        onnx_model_path = os.path.join(output_model_path, 'onnx', f'model-{max_length}', 'model')
        logger.info(f'convert pt to onnx {max_length}...')
        convert_pytorch_to_onnx(model_path,
                                onnx_model_path,
                                tokenizer,
                                max_length,
                                use_external_data_format=True)
        os.makedirs(os.path.join(output_model_path, 'tensorrt'), exist_ok=True)
        tensorrt_path = os.path.join(output_model_path, 'tensorrt', f"model-{max_length}.engine")
        logger.info(f'convert onnx to tensorrt {max_length}...')
        convert_onnx_to_tensorrt(onnx_model_path,
                                 tensorrt_path,
                                 'CUDA:0',
                                 max_batch_size=1)
    # save tokenizer
    tokenizer.save_pretrained(os.path.join(output_model_path, 'tensorrt'))
    os.system(f'cp ../user_data/data/normal_vocab.json {os.path.join(output_model_path, "tensorrt", "normal_vocab.json")}')
    os.system(f'cp ../user_data/data/train_data/vocab.json {os.path.join(output_model_path, "tensorrt", "vocab.json")}')

if __name__ == '__main__':
    fire.Fire(convert_model)


