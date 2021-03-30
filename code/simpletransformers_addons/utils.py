from difflib import SequenceMatcher
import torch
from pathos.multiprocessing import ProcessingPool as Pool

def merge_text_pair(text_a, text_b):
    matches = SequenceMatcher(None,
                         text_a,
                         text_b
                         ).get_opcodes()
    output_str = []
    for item in matches:
        if item[0] == 'equal':
            output_str.append(text_a[item[1]:item[2]])
        elif item[0] == 'replace':
            output_str.append(f'({text_a[item[1]:item[2]]})[{text_b[item[3]:item[4]]}]')
        elif item[0] == 'insert':
            output_str.append(f'[{text_b[item[3]:item[4]]}]')
        elif item[0] == 'delete':
            output_str.append(f'({text_a[item[1]:item[2]]})')
    return ''.join(output_str)

def get_diff_token_type_ids(input_ids,
                            token_type_ids,
                            attention_mask):
    total_len = len(input_ids)
    KEEP_ID = 1
    DELETE_ID = 2
    ADD_ID = 3
    REPLACE_ID = 4
    seq_len = sum(attention_mask)
    input_a, input_b = [], []
    for i in range(seq_len):
        if token_type_ids[i] == 0:
            input_a.append(input_ids[i])
        elif token_type_ids[i] == 1:
            input_b.append(input_ids[i])
    input_a = input_a[1:-1]
    input_b = input_b[:-1]
    diff_token_type_ids_a = [0] * len(input_a)
    diff_token_type_ids_b = [0] * len(input_b)
    # matches = SequenceMatcher(None,
    #                      input_a,
    #                      input_b
    #                      ).get_opcodes()
    # for item in matches:
    #     if item[0] == 'equal':
    #         diff_token_type_ids_a[item[1]:item[2]] = [KEEP_ID] * (item[2] - item[1])
    #         diff_token_type_ids_b[item[3]:item[4]] = [KEEP_ID] * (item[4] - item[3])
    #     elif item[0] == 'replace':
    #         diff_token_type_ids_a[item[1]:item[2]] = [REPLACE_ID] * (item[2] - item[1])
    #         diff_token_type_ids_b[item[3]:item[4]] = [REPLACE_ID] * (item[4] - item[3])
    #     elif item[0] == 'insert':
    #         diff_token_type_ids_b[item[3]:item[4]] = [ADD_ID] * (item[4] - item[3])
    #     elif item[0] == 'delete':
    #         diff_token_type_ids_a[item[1]:item[2]] = [DELETE_ID] * (item[2] - item[1])
    tokens_set_a = set(input_a)
    tokens_set_b = set(input_b)
    for i in range(len(input_a)):
        if input_a[i] in tokens_set_b:
            diff_token_type_ids_a[i] = 1
        else:
            diff_token_type_ids_a[i] = 2
    for i in range(len(input_b)):
        if input_b[i] in tokens_set_a:
            diff_token_type_ids_b[i] = 1
        else:
            diff_token_type_ids_b[i] = 2

    diff_token_type_ids = [0] + diff_token_type_ids_a + [0] + diff_token_type_ids_b + [0]
    output_token_type_ids = []
    for i in range(len(diff_token_type_ids)):
        output_token_type_ids.append(diff_token_type_ids[i] * 10 + token_type_ids[i])
    for _ in range(total_len - len(output_token_type_ids)):
        output_token_type_ids.append(0)
    return output_token_type_ids

def get_mask_probility_matrix(inputs,
                              sep_token_id=3,
                              mlm_probability=0.15):
    diff_mlm_probability = mlm_probability * 1.5
    same_mlm_probability = mlm_probability * 0.7
    inputs = inputs.detach().tolist()
    max_seq_len = len(inputs[0])
    outputs = []
    for i in range(len(inputs)):
        cur_inputs = inputs[i]
        input_a, input_b = set(), set()
        segment_type = 0
        for j in range(max_seq_len):
            if cur_inputs[j] == -100:
                break
            if cur_inputs[j] == sep_token_id:
                segment_type = 1
            if segment_type == 0:
                input_a.add(cur_inputs[j])
            else:
                input_b.add(cur_inputs[j])
        insection_tokens = input_a & input_b
        cur_outputs = []
        for j in range(max_seq_len):
            if cur_inputs[j] in insection_tokens:
                cur_outputs.append(same_mlm_probability)
            else:
                cur_outputs.append(diff_mlm_probability)
        outputs.append(cur_outputs)
    return torch.tensor(outputs)

import torch
from transformers import PreTrainedTokenizer
from typing import Tuple
import random
import pickle as pkl

_ngram_dict = None

def get_ngram_dict():
    global _ngram_dict
    if _ngram_dict is None:
        with open('../user_data/data/train_data/ngram_dict.pkl', 'rb') as fin:
            _ngram_dict = pkl.load(fin)
            return _ngram_dict
    else:
        return _ngram_dict


def get_all_spans(token_list, no_mask_tokens,
                  max_ngram=3):
    output_spans = []
    cur_seq_len = 0
    ngram_dict = get_ngram_dict()
    for i in range(len(token_list)):
        if token_list[i] == -100:
            break
        if token_list[i] not in no_mask_tokens:
            output_spans.append((i, i+1))
        for ngram_num in range(2, max_ngram + 1):
            if i + ngram_num > len(token_list):
                break
            if all([token_list[i+j] not in no_mask_tokens for j in range(ngram_num)]) and \
                tuple(token_list[i:i+ngram_num]) in ngram_dict['ngrams_set']:
                output_spans.append((i, i+ngram_num))
        cur_seq_len += 1
    return output_spans, cur_seq_len - 4

def get_mask_span(input_tokens):
    input_tokens = tuple(input_tokens)
    ngram_dict = get_ngram_dict()
    random_rate = random.random()
    if random_rate < 0.7 and input_tokens in ngram_dict['sim_ngrams']:
        # replace similar ngram.
        return random.choice(ngram_dict['sim_ngrams'][input_tokens])
    if random_rate < 0.9:
        # replace random ngram.
        return random.choice(ngram_dict['ngrams'][len(input_tokens)])
    return input_tokens

def process_single_record(inputs_item,
                          labels_item,
                          masked_indices_item,
                          no_mask_tokens,
                          args):
    all_spans, cur_seq_len = get_all_spans(inputs_item, no_mask_tokens)
    random.shuffle(all_spans)
    total_mask_len = 0
    for span in all_spans:
        if total_mask_len > args.mlm_probability * cur_seq_len:
            break
        if any([masked_indices_item[idx] for idx in range(span[0], span[1])]):
            continue
        if total_mask_len + span[1] - span[0] > 2 * args.mlm_probability * cur_seq_len:
            continue
        inputs_item[span[0]:span[1]] = get_mask_span(labels_item[span[0]:span[1]])
        masked_indices_item[span[0]:span[1]] = [True] * (span[1] - span[0])
        total_mask_len += (span[1] - span[0])
    return inputs_item, masked_indices_item

def process_batch(inputs, labels, masked_indices, no_mask_tokens, args):
    # with Pool() as p:
    #     outputs = p.map(process_single_record,
    #           inputs,
    #           labels,
    #           masked_indices,
    #           [no_mask_tokens for _ in range(len(inputs))],
    #           [args for _ in range(len(inputs))])
    outputs = []
    for i in range(len(inputs)):
        outputs.append(process_single_record(inputs[i],
                                             labels[i],
                                             masked_indices[i],
                                             no_mask_tokens, args))
    masked_indices, inputs = [], []
    for item in outputs:
        inputs.append(item[0])
        masked_indices.append(item[1])
    return inputs, masked_indices


def mask_tokens_enhanced(inputs:torch.Tensor, tokenizer: PreTrainedTokenizer, args) -> Tuple[torch.Tensor, torch.Tensor]:
    labels_tensor = inputs.clone()
    labels = inputs.detach().tolist()
    no_mask_tokens = [tokenizer.convert_tokens_to_ids(token) for token in args.no_mask_tokens + args.label_tokens] + [
        tokenizer.sep_token_id, tokenizer.cls_token_id, -100]
    batch_size, seq_len = labels_tensor.shape[0], labels_tensor.shape[1]
    inputs = inputs.detach().tolist()
    masked_indices = [[False] * seq_len for _ in range(batch_size)]
    inputs, masked_indices = process_batch(inputs, labels, masked_indices, no_mask_tokens, args)
    masked_indices = torch.tensor(masked_indices)
    labels_tensor[~masked_indices] = -100
    inputs = torch.tensor(inputs)
    inputs[inputs == -100] = 0
    return inputs, labels_tensor

import torch
import torch.nn as nn
import threading


def distribute_module(module, device):
    return module.cuda(device)


def parallel_apply(modules, inputs, kwargs_tup=None, devices=None):
    assert len(modules) == len(inputs)
    if kwargs_tup is not None:
        assert len(kwargs_tup) == len(modules)
    else:
        kwargs_tup = ({},) * len(modules)
    if devices is not None:
        assert len(modules) == len(devices)
    else:
        raise VauleError('devices is None')

    lock = threading.Lock()
    results = {}
    #grad_enabled = torch.is_grad_enabled()

    def _worker(i, module, input, kwargs, device=None):
        # torch.set_grad_enabled(grad_enabled)
        try:
            with torch.cuda.device(device):
                output = module(input)
            with lock:
                results[i] = output
        except Exception as e:
            with lock:
                results[i] = e

    if len(modules) > 1:
        threads = [threading.Thread(target=_worker,
                                    args=(i, module, input, kwargs, device))
                   for i, (module, input, kwargs, device) in
                   enumerate(zip(modules, inputs, kwargs_tup, devices))]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
    else:
        _worker(0, modules[0], inputs[0], kwargs_tup[0], devices[0])

    outputs = []
    for i in range(len(inputs)):
        output = results[i]
        if isinstance(output, Exception):
            raise output
        outputs.append(output)
    return outputs
