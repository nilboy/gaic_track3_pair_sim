import logging
import json
import os
import pickle
import random
from multiprocessing import Pool
from typing import Tuple, Dict, List

import torch
from tokenizers.processors import BertProcessing
from torch.utils.data import Dataset
from tqdm.auto import tqdm
from transformers import PreTrainedTokenizer

from simpletransformers_addons.utils import get_diff_token_type_ids

logger = logging.getLogger(__name__)


def encode(data):
    tokenizer, line = data
    return tokenizer.encode(line)


def encode_sliding_window(data):
    tokenizer, line, max_seq_length, special_tokens_count, stride, no_padding = data

    tokens = tokenizer.tokenize(line)
    stride = int(max_seq_length * stride)
    token_sets = []
    if len(tokens) > max_seq_length - special_tokens_count:
        token_sets = [tokens[i : i + max_seq_length - special_tokens_count] for i in range(0, len(tokens), stride)]
    else:
        token_sets.append(tokens)

    features = []
    if not no_padding:
        sep_token = tokenizer.sep_token_id
        cls_token = tokenizer.cls_token_id
        pad_token = tokenizer.pad_token_id

        for tokens in token_sets:
            tokens = [cls_token] + tokens + [sep_token]

            input_ids = tokenizer.convert_tokens_to_ids(tokens)
            padding_length = max_seq_length - len(input_ids)
            input_ids = input_ids + ([pad_token] * padding_length)

            assert len(input_ids) == max_seq_length

            features.append(input_ids)
    else:
        for tokens in token_sets:
            input_ids = tokenizer.convert_tokens_to_ids(tokens)
            features.append(input_ids)

    return features


class SimpleDataset(Dataset):
    def __init__(self, tokenizer, args, file_path, mode, block_size=512, special_tokens_count=2, sliding_window=False):
        assert os.path.isfile(file_path)
        block_size = block_size - special_tokens_count
        directory, filename = os.path.split(file_path)
        cached_features_file = os.path.join(
            args.cache_dir, args.model_type + "_cached_lm_" + str(block_size) + "_" + filename
        )

        if os.path.exists(cached_features_file) and (
            (not args.reprocess_input_data and not args.no_cache)
            or (mode == "dev" and args.use_cached_eval_features and not args.no_cache)
        ):
            logger.info(" Loading features from cached file %s", cached_features_file)
            with open(cached_features_file, "rb") as handle:
                self.examples = pickle.load(handle)
        else:
            logger.info(" Creating features from dataset file at %s", args.cache_dir)

            if sliding_window:
                no_padding = True if args.model_type in ["gpt2", "openai-gpt"] else False
                with open(file_path, encoding="utf-8") as f:
                    lines = [
                        (tokenizer, line, args.max_seq_length, special_tokens_count, args.stride, no_padding)
                        for line in f.read().splitlines()
                        if (len(line) > 0 and not line.isspace())
                    ]

                if args.use_multiprocessing:
                    with Pool(args.process_count) as p:
                        self.examples = list(
                            tqdm(
                                p.imap(encode_sliding_window, lines, chunksize=args.multiprocessing_chunksize),
                                total=len(lines),
                                # disable=silent,
                            )
                        )
                else:
                    self.examples = [encode_sliding_window(line) for line in lines]

                self.examples = [example for example_set in self.examples for example in example_set]
            else:
                with open(file_path, encoding="utf-8") as f:
                    lines = [
                        (tokenizer, line) for line in f.read().splitlines() if (len(line) > 0 and not line.isspace())
                    ]

                if args.use_multiprocessing:
                    with Pool(args.process_count) as p:
                        self.examples = list(
                            tqdm(
                                p.imap(encode, lines, chunksize=args.multiprocessing_chunksize),
                                total=len(lines),
                                # disable=silent,
                            )
                        )
                else:
                    self.examples = [encode(line) for line in lines]

                self.examples = [token for tokens in self.examples for token in tokens]
                if len(self.examples) > block_size:
                    self.examples = [
                        tokenizer.build_inputs_with_special_tokens(self.examples[i : i + block_size])
                        for i in tqdm(range(0, len(self.examples) - block_size + 1, block_size))
                    ]
                else:
                    self.examples = [tokenizer.build_inputs_with_special_tokens(self.examples)]

            logger.info(" Saving features into cached file %s", cached_features_file)
            with open(cached_features_file, "wb") as handle:
                pickle.dump(self.examples, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return torch.tensor(self.examples[item], dtype=torch.long)

from simpletransformers_addons.utils import mask_tokens_enhanced
mask_tokens = mask_tokens_enhanced

def mask_tokens_v1(inputs: torch.Tensor, tokenizer: PreTrainedTokenizer, args) -> Tuple[torch.Tensor, torch.Tensor]:
    """ Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original. """

    if tokenizer.mask_token is None:
        raise ValueError(
            "This tokenizer does not have a mask token which is necessary for masked language modeling."
            "Set 'mlm' to False in args if you want to use this tokenizer."
        )

    labels = inputs.clone()
    # We sample a few tokens in each sequence for masked-LM training
    # (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)
    # probability_matrix = torch.full(labels.shape, args.mlm_probability)

    from simpletransformers_addons.utils import get_mask_probility_matrix
    probability_matrix = get_mask_probility_matrix(labels, sep_token_id=tokenizer.sep_token_id,
                                                   mlm_probability=args.mlm_probability)
    no_mask_tokens = [tokenizer.convert_tokens_to_ids(token) for token in args.no_mask_tokens + args.label_tokens] + [tokenizer.sep_token_id, tokenizer.cls_token_id, -100]
    special_tokens_mask = [
        list(map(lambda x: 1 if x in no_mask_tokens else 0, val)) for val in labels.tolist()
    ]
    probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
    if tokenizer._pad_token is not None:
        padding_mask = labels.eq(tokenizer.pad_token_id)
        probability_matrix.masked_fill_(padding_mask, value=0.0)
    masked_indices = torch.bernoulli(probability_matrix).bool()
    # ngram mask.
    try:
        args.mask_ngram
    except:
        args.mask_ngram = 1
    if args.mask_ngram > 1:
        # construct ngram mask
        for i in range(masked_indices.shape[0]):
            for j in range(masked_indices.shape[1]-1, -1, -1):
                if masked_indices[i, j]:
                    cur_ngram = random.randint(1, args.mask_ngram)
                    for ngram_id in range(cur_ngram):
                        if j + ngram_id < masked_indices.shape[1] and probability_matrix[i, j+ngram_id] != 0.0:
                            masked_indices[i, j+ngram_id] = True


    labels[~masked_indices] = -100  # We only compute loss on masked tokens

    if args.model_type == "electra":
        # For ELECTRA, we replace all masked input tokens with tokenizer.mask_token
        inputs[masked_indices] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)
    else:
        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
    # change -100 to 0
    inputs[inputs == -100] = 0
    return inputs, labels

class SentencePairTextDataset(Dataset):
    """
    This will be superseded by a framework-agnostic approach soon.
    """

    def __init__(self, tokenizer: PreTrainedTokenizer, file_path: str, block_size: int,
                 label_tokens: List[str]):
        assert os.path.isfile(file_path), f"Input file path {file_path} not found"
        # Here, we do not cache the features, operating under the assumption
        # that we will soon use fast multithreaded tokenizers from the
        # `tokenizers` repo everywhere =)
        logger.info("Creating features from dataset file at %s", file_path)
        label_token_ids = [tokenizer.convert_tokens_to_ids(token) for token in label_tokens]
        with open(file_path, encoding="utf-8") as f:
            lines, labels = [], []
            for line in f.read().splitlines():
                if len(line) > 0 and not line.isspace():
                    record = json.loads(line)
                    if 'text_a' in record:
                        lines.append([record['text_a'], record['text_b']])
                    else:
                        lines.append(record['text'])
                    labels.append(label_token_ids[int(record['labels'])])

        batch_encoding = tokenizer(lines, add_special_tokens=True, truncation=True, max_length=block_size - 1)
        self.examples = []
        for i in range(len(batch_encoding['input_ids'])):
            batch_encoding['input_ids'][i].append(labels[i])
            batch_encoding['token_type_ids'][i].append(0)
            batch_encoding['attention_mask'][i].append(1)
            self.examples.append({
                "input_ids": torch.tensor(batch_encoding['input_ids'][i], dtype=torch.long),
                "token_type_ids": torch.tensor(batch_encoding['token_type_ids'][i], dtype=torch.long),
                "attention_mask": torch.tensor(batch_encoding['attention_mask'][i], dtype=torch.long)
            })

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i) -> Dict[str, torch.tensor]:
        return self.examples[i]

class DiffSentencePairTextDataset(Dataset):
    """
    This will be superseded by a framework-agnostic approach soon.
    """

    def __init__(self, tokenizer: PreTrainedTokenizer, file_path: str, block_size: int,
                 label_tokens: List[str]):
        assert os.path.isfile(file_path), f"Input file path {file_path} not found"
        # Here, we do not cache the features, operating under the assumption
        # that we will soon use fast multithreaded tokenizers from the
        # `tokenizers` repo everywhere =)
        logger.info("Creating features from dataset file at %s", file_path)
        label_token_ids = [tokenizer.convert_tokens_to_ids(token) for token in label_tokens]
        with open(file_path, encoding="utf-8") as f:
            lines, labels = [], []
            for line in f.read().splitlines():
                if len(line) > 0 and not line.isspace():
                    record = json.loads(line)
                    if 'text_a' in record:
                        lines.append([record['text_a'], record['text_b']])
                    else:
                        lines.append(record['text'])
                    labels.append(label_token_ids[int(record['labels'])])

        batch_encoding = tokenizer(lines, add_special_tokens=True, truncation=True, max_length=block_size - 1)

        self.examples = []
        for i in range(len(batch_encoding['input_ids'])):
            # convert diff token_type_ids
            batch_encoding['token_type_ids'][i] = get_diff_token_type_ids(input_ids=batch_encoding['input_ids'][i],
                                                                          token_type_ids=batch_encoding['token_type_ids'][i],
                                                                          attention_mask=batch_encoding['attention_mask'][i])
            batch_encoding['input_ids'][i].append(labels[i])
            batch_encoding['token_type_ids'][i].append(0)
            batch_encoding['attention_mask'][i].append(1)
            self.examples.append({
                "input_ids": torch.tensor(batch_encoding['input_ids'][i], dtype=torch.long),
                "token_type_ids": torch.tensor(batch_encoding['token_type_ids'][i], dtype=torch.long),
                "attention_mask": torch.tensor(batch_encoding['attention_mask'][i], dtype=torch.long)
            })

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i) -> Dict[str, torch.tensor]:
        return self.examples[i]


class SymmetricSentencePairTextDataset(Dataset):
    """
    This will be superseded by a framework-agnostic approach soon.
    """

    def __init__(self, tokenizer: PreTrainedTokenizer, file_path: str, block_size: int,
                 label_tokens: List[str]):
        assert os.path.isfile(file_path), f"Input file path {file_path} not found"
        # Here, we do not cache the features, operating under the assumption
        # that we will soon use fast multithreaded tokenizers from the
        # `tokenizers` repo everywhere =)
        logger.info("Creating features from dataset file at %s", file_path)
        label_token_ids = [tokenizer.convert_tokens_to_ids(token) for token in label_tokens]
        with open(file_path, encoding="utf-8") as f:
            lines, labels = [], []
            for line in f.read().splitlines():
                if len(line) > 0 and not line.isspace():
                    record = json.loads(line)
                    if 'text_a' in record:
                        lines.append([record['text_a'], record['text_b']])
                    else:
                        lines.append(record['text'])
                    labels.append(label_token_ids[int(record['labels'])])
        batch_encoding = {
            'input_ids': [],
            'token_type_ids': [],
            'attention_mask': [],
            'position_ids': []
        }
        for line in lines:
            input_ids, token_type_ids, attention_mask, position_ids = [], [], [], []
            if isinstance(line, list):
                # [CLS]
                input_ids.append(tokenizer.cls_token_id)
                token_type_ids.append(0)
                attention_mask.append(1)
                position_ids.append(0)
                # [CLS]
                input_ids.append(tokenizer.cls_token_id)
                token_type_ids.append(1)
                attention_mask.append(1)
                position_ids.append(0)
                # append first sentence input
                sentence_encode = tokenizer(line[0], add_special_tokens=False)
                input_ids.extend(sentence_encode['input_ids'])
                token_type_ids.extend([0] * len(sentence_encode['input_ids']))
                attention_mask.extend([1] * len(sentence_encode['input_ids']))
                position_ids.extend(list(range(1, len(sentence_encode['input_ids'])+1)))
                # append second sentence input
                sentence_encode = tokenizer(line[1], add_special_tokens=False)
                input_ids.extend(sentence_encode['input_ids'])
                token_type_ids.extend([1] * len(sentence_encode['input_ids']))
                attention_mask.extend([1] * len(sentence_encode['input_ids']))
                position_ids.extend(list(range(1, len(sentence_encode['input_ids'])+1)))
            else:
                raise ValueError('SymmetricSentencePairTextDataset not support single string')
            input_ids = input_ids[0:block_size]
            token_type_ids = token_type_ids[0:block_size]
            attention_mask = attention_mask[0:block_size]
            position_ids = position_ids[0:block_size]
            batch_encoding['input_ids'].append(input_ids)
            batch_encoding['token_type_ids'].append(token_type_ids)
            batch_encoding['attention_mask'].append(attention_mask)
            batch_encoding['position_ids'].append(position_ids)

        self.examples = []
        for i in range(len(batch_encoding['input_ids'])):
            self.examples.append({
                "input_ids": torch.tensor(batch_encoding['input_ids'][i], dtype=torch.long),
                "token_type_ids": torch.tensor(batch_encoding['token_type_ids'][i], dtype=torch.long),
                "attention_mask": torch.tensor(batch_encoding['attention_mask'][i], dtype=torch.long),
                "position_ids": torch.tensor(batch_encoding['position_ids'][i], dtype=torch.long)
            })

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i) -> Dict[str, torch.tensor]:
        return self.examples[i]
