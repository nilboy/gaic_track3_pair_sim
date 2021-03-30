import fire
import logging
import torch
import numpy as np
import random
from torch import nn
from transformers import BertTokenizer, BertForMaskedLM
from simpletransformers_addons.transformers.bert.modeling_nezha import NeZhaForMaskedLM
from transformers import DebertaForMaskedLM

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def set_seed(manual_seed):
    random.seed(manual_seed)
    np.random.seed(manual_seed)
    torch.manual_seed(manual_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(manual_seed)

def convert_model(input_model_name='/nfs/users/jiangxinghua/models/transformers/hugface/chinese-bert-wwm-ext',
                  output_model_name='../user_data/pretrained/demo',
                  model_type="bert",
                  tokenizer_model="../user_data/tokenizer"):
    model_map = {
        'bert': BertForMaskedLM,
        'nezha': NeZhaForMaskedLM,
        'deberta': DebertaForMaskedLM
    }
    set_seed(9527)
    tokenizer = BertTokenizer.from_pretrained(tokenizer_model)
    model = model_map[model_type].from_pretrained(input_model_name)
    new_embedding_weight = torch.normal(0.0, 1.0, size=(tokenizer.vocab_size, model.config.hidden_size))
    model.set_input_embeddings(nn.Embedding.from_pretrained(new_embedding_weight))
    model.cls.predictions.decoder = nn.Linear(in_features=model.config.hidden_size,
                                             out_features=tokenizer.vocab_size, bias=True)
    model.cls.predictions.bias = nn.Parameter(torch.normal(0, 1.0, size=(tokenizer.vocab_size,)))
    model.config.vocab_size = tokenizer.vocab_size
    model.save_pretrained(output_model_name)
    tokenizer.save_pretrained(output_model_name)

if __name__ == '__main__':
    fire.Fire(convert_model)
