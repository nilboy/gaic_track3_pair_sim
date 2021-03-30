import fire
import logging
import os
import torch
import numpy as np
import random
from torch import nn
from tqdm.auto import tqdm
from tokenizers import BertWordPieceTokenizer, ByteLevelBPETokenizer
from transformers import BertTokenizer, BertForMaskedLM

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def set_seed(manual_seed):
    random.seed(manual_seed)
    np.random.seed(manual_seed)
    torch.manual_seed(manual_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(manual_seed)

def train_tokenizer(train_file="../user_data/data/train_data/tokenizer_data.txt",
                    vocab_size=22000,
                    output_model_name="../user_data/tokenizer",
                    seed=9527):
    set_seed(seed)
    tokenizer = BertWordPieceTokenizer(
                clean_text=False,
                handle_chinese_chars=True,
                strip_accents=False,
                lowercase=False
            )
    special_tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
    # for i in range(100):
    #     special_tokens.append(f"[unused{i}]")

    tokenizer.train(
        files=[train_file],
        vocab_size=vocab_size,
        min_frequency=0,
        special_tokens=special_tokens,
        limit_alphabet=vocab_size,
        wordpieces_prefix="##"
    )
    os.makedirs(output_model_name, exist_ok=True)
    tokenizer.save_model(output_model_name)
    tokenizer = BertTokenizer.from_pretrained(output_model_name,
                                                  do_lower_case=False,
                                                  strip_accents=False)
    tokenizer.save_pretrained(output_model_name)
    logger.info(f'save tokenizer, with vocab_size: {tokenizer.vocab_size}')

if __name__ == '__main__':
    fire.Fire(train_tokenizer)
