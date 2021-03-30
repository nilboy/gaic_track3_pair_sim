from __future__ import absolute_import, division, print_function

import copy
import csv
import json
import linecache
import os
import sys
from collections import Counter
from io import open
from multiprocessing import Pool, cpu_count

import torch
import torch.nn as nn
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import f1_score, matthews_corrcoef
from torch.utils.data import Dataset
from tqdm.auto import tqdm

try:
    import torchvision
    import torchvision.transforms as transforms

    torchvision_available = True
    from PIL import Image
except ImportError:
    torchvision_available = False


csv.field_size_limit(2147483647)

from simpletransformers.classification.classification_utils import convert_example_to_feature as convert_example_to_feature_inner
from simpletransformers.classification.classification_utils import InputExample, InputFeatures

def discrete_auc(truth, predictions):
    truth = [t for t in truth]
    predictions = [p for p in predictions]
    pairs = sorted(zip(predictions, truth), key=lambda x: x[0])
    sum_idx, num_true = 0, 0
    for idx, (p, t) in enumerate(pairs):
        if t == 1.0:
            sum_idx += (idx + 1)
            num_true += 1
    num_false = len(truth) - num_true
    return (sum_idx - num_true * (num_true + 1)/2)/(num_false * num_true)

def convert_examples_to_features(
    examples,
    max_seq_length,
    tokenizer,
    output_mode,
    cls_token_at_end=False,
    sep_token_extra=False,
    pad_on_left=False,
    cls_token="[CLS]",
    sep_token="[SEP]",
    pad_token=0,
    sequence_a_segment_id=0,
    sequence_b_segment_id=1,
    cls_token_segment_id=1,
    pad_token_segment_id=0,
    mask_padding_with_zero=True,
    process_count=cpu_count() - 2,
    multi_label=False,
    silent=False,
    use_multiprocessing=True,
    sliding_window=False,
    flatten=False,
    stride=None,
    add_prefix_space=False,
    pad_to_max_length=True,
    args=None,
):
    """ Loads a data file into a list of `InputBatch`s
        `cls_token_at_end` define the location of the CLS token:
            - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
            - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
        `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
    """

    examples = [
        (
            example,
            max_seq_length,
            tokenizer,
            output_mode,
            cls_token_at_end,
            cls_token,
            sep_token,
            cls_token_segment_id,
            pad_on_left,
            pad_token_segment_id,
            sep_token_extra,
            multi_label,
            stride,
            pad_token,
            add_prefix_space,
            pad_to_max_length,
        )
        for example in examples
    ]

    if use_multiprocessing:
        with Pool(process_count) as p:
            features = list(
                tqdm(
                    p.imap(convert_example_to_feature, examples, chunksize=args.multiprocessing_chunksize),
                    total=len(examples),
                    disable=silent,
                )
            )
    else:
        features = [convert_example_to_feature(example) for example in tqdm(examples, disable=silent)]

    return features

def convert_example_to_feature(
    example_row,
    pad_token=0,
    sequence_a_segment_id=0,
    sequence_b_segment_id=1,
    cls_token_segment_id=1,
    pad_token_segment_id=0,
    mask_padding_with_zero=True,
    sep_token_extra=False,
):
    example_a = [copy.deepcopy(example_row[0])] + list(example_row[1:])
    example_b = [copy.deepcopy(example_row[0])] + list(example_row[1:])
    text_a = example_row[0].text_a
    segments_a = text_a.split('[SEP]')
    if len(segments_a) == 2:
        example_a[0].text_a = segments_a[0]
        example_a[0].text_b = segments_a[1]
    else:
        example_a[0].text_a = segments_a[0]
        example_a[0].text_b = None
    feature_a = convert_example_to_feature_inner(
        example_a,
        pad_token,
        sequence_a_segment_id,
        sequence_b_segment_id,
        cls_token_segment_id,
        pad_token_segment_id,
        mask_padding_with_zero,
        sep_token_extra)
    text_b = example_row[0].text_b
    if not text_b:
        feature_b = feature_a
    else:
        segments_b = text_b.split('[SEP]')
        if len(segments_b) == 2:
            example_b[0].text_a = segments_b[0]
            example_b[0].text_b = segments_b[1]
        else:
            example_b[0].text_a = segments_b[0]
            example_b[0].text_b = None
        feature_b = convert_example_to_feature_inner(
            example_b,
            pad_token,
            sequence_a_segment_id,
            sequence_b_segment_id,
            cls_token_segment_id,
            pad_token_segment_id,
            mask_padding_with_zero,
            sep_token_extra)
    return {
        'feature_a': feature_a,
        'feature_b': feature_b
    }


