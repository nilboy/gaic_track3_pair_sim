import copy
import fire
import json
import random
import networkx as nx
import os
import re
import pandas as pd
import pickle as pkl
import uuid
import logging

from tqdm.auto import tqdm

logging.basicConfig()
logger = logging.getLogger('convert_data')
logger.setLevel(logging.INFO)

def convert_tsv_to_jsonl(input_file, output_file):
    input_df = pd.read_csv(input_file, header=None, delimiter='\t')
    records = []
    for _, record in tqdm(input_df.iterrows()):
        if len(record) > 2:
            labels = record[2]
        else:
            labels = 2
        records.append({
            'rid'   : uuid.uuid4().hex,
            'labels': labels,
            'text_a': str(record[0]),
            'text_b': str(record[1])
        })
    with open(output_file, 'w') as fout:
        for record in tqdm(records):
            fout.write(json.dumps(record, ensure_ascii=False) + '\n')

def construct_meta_info(input_data_file, output_file):
    meta_info = {
        'sid2sentence': {},
        'sentence2sid': {},
        'sentence_graph': nx.Graph()
    }
    with open(input_data_file) as f:
        for line in tqdm(f):
            record = json.loads(line)
            sentences = [record['text_a'], record['text_b']]
            for sentence in sentences:
                if sentence not in meta_info['sentence2sid']:
                    sid = uuid.uuid4().hex
                    meta_info['sid2sentence'][sid] = sentence
                    meta_info['sentence2sid'][sentence] = sid
                    meta_info['sentence_graph'].add_node(sid)
            if record['labels'] != 2:
                meta_info['sentence_graph'].add_edge(meta_info['sentence2sid'][sentences[0]],
                                                     meta_info['sentence2sid'][sentences[1]])
    with open(output_file, 'wb') as fout:
        pkl.dump(meta_info, fout)

def get_kfold_data(input_data_file,
                   meta_info_file,
                   output_dir,
                   n_splits=5,
                   random_seed=9527):
    random.seed(random_seed)
    # 获取所有连通图
    with open(meta_info_file, 'rb') as f:
        meta_info = pkl.load(f)
    logger.info('compute connected components')
    components = list(nx.connected_components(meta_info['sentence_graph']))
    logger.info(f'components number: {len(components)}')
    sid2cid = {}
    for cid, component in enumerate(components):
        for sid in component:
            sid2cid[sid] = cid
    records_components = [[] for _ in range(len(components))]
    logger.info('compute records components')
    with open(input_data_file) as f:
        for line in tqdm(f):
            record = json.loads(line)
            sid_a, sid_b = meta_info['sentence2sid'][record['text_a']], meta_info['sentence2sid'][record['text_b']]
            assert sid2cid[sid_a] == sid2cid[sid_b], 'cid_a != cid_b'
            records_components[sid2cid[sid_a]].append(record)
    random.shuffle(records_components)
    # 生成 kfold data.
    logger.info('generate kfold data')
    per_split_num = int(len(records_components)/n_splits)
    for split_id in tqdm(range(n_splits)):
        os.makedirs(os.path.join(output_dir, str(split_id)), exist_ok=True)
        dev_ids = list(range(split_id * per_split_num, (split_id+1)*per_split_num))
        train_records, dev_records = [], []
        for idx, records_component in enumerate(records_components):
            if idx in dev_ids:
                dev_records.extend(records_component)
            else:
                train_records.extend(records_component)
        random.shuffle(train_records)
        with open(os.path.join(output_dir, str(split_id), 'train.jsonl'), 'w') as fout:
            for record in tqdm(train_records):
                fout.write(json.dumps(record, ensure_ascii=False) + '\n')
        with open(os.path.join(output_dir, str(split_id), 'dev.jsonl'), 'w') as fout:
            for record in tqdm(dev_records):
                fout.write(json.dumps(record, ensure_ascii=False) + '\n')
        # generate enhanced train data
        enhanced_train_records = []
        for record in tqdm(train_records):
            enhanced_train_records.append(record)
            record_dup = copy.deepcopy(record)
            record_dup['text_a'], record_dup['text_b'] = record['text_b'], record['text_a']
            enhanced_train_records.append(record_dup)
        random.shuffle(enhanced_train_records)
        with open(os.path.join(output_dir, str(split_id), 'train_enhanced.jsonl'), 'w') as fout:
            for record in tqdm(enhanced_train_records):
                fout.write(json.dumps(record, ensure_ascii=False) + '\n')


def construct_mlm_data(train_data_file,
                       test_data_file,
                       output_data_file,
                       random_seed=9527):
    random.seed(random_seed)
    records = []
    text_pair_set = set()
    for f in [train_data_file, test_data_file]:
        with open(f) as f:
            for line in tqdm(f):
                record = json.loads(line)
                if (record['text_a'], record['text_b']) in text_pair_set:
                    continue
                text_pair_set.add((record['text_a'], record['text_b']))
                records.append(record)
                record_dup = copy.deepcopy(record)
                record_dup['text_a'], record_dup['text_b'] = record['text_b'], record['text_a']
                records.append(record_dup)

    random.shuffle(records)
    with open(output_data_file, 'w') as fout:
        for record in tqdm(records):
            fout.write(json.dumps(record, ensure_ascii=False) + '\n')

def construct_enhanced_data(input_data_file,
                            output_data_file,
                            random_seed=9527):
    random.seed(random_seed)
    records = []
    with open(input_data_file) as f:
        for line in tqdm(f):
            record = json.loads(line)
            records.append(record)
            record_dup = copy.deepcopy(record)
            record_dup['text_a'], record_dup['text_b'] = record['text_b'], record['text_a']
            records.append(record_dup)
    random.shuffle(records)
    with open(output_data_file, 'w') as fout:
        for record in tqdm(records):
            fout.write(json.dumps(record, ensure_ascii=False) + '\n')

def construct_vocab(train_file,
                    test_file,
                    output_vocab_file):
    # construct vocab
    vocab = {
        'num2token': {},
        'token2num': {}
    }
    num_id = 0
    for token in list(range(0x4e00, 0x9fa6)) + list(range(0x0800, 0x4e00)):
        if chr(token) not in vocab['token2num']:
            vocab['num2token'][num_id] = chr(token)
            vocab['token2num'][chr(token)] = num_id
            num_id += 1
    with open(output_vocab_file, 'w') as fout:
        json.dump(vocab, fout, ensure_ascii=False, indent=2)
    return vocab

def convert_record_style(input_file, vocab, output_file):
    input_df = pd.read_csv(input_file, header=None, delimiter='\t')
    def convert_str_style(input_str):
        token_ids = [int(item) for item in input_str.split() if item]
        tokens = [vocab['num2token'][idx] for idx in token_ids]
        return ''.join(tokens)
    input_df[0] = input_df[0].apply(convert_str_style)
    input_df[1] = input_df[1].apply(convert_str_style)
    input_df.to_csv(output_file, header=None, sep='\t', index=False)

def construct_tokenizer_data(input_file, output_file):
    with open(output_file, 'w') as fout:
        with open(input_file) as fin:
            for line in tqdm(fin):
                record = json.loads(line)
                fout.write(record['text_a'] + '\n')
                fout.write(record['text_b'] + '\n')


def convert_data(train_file='../user_data/data/train.tsv',
                 test_file='../user_data/data/test.tsv',
                 output_dir='../user_data/data/train_data',
                 n_splits=5,
                 random_seed=9527):
    os.makedirs(output_dir, exist_ok=True)
    logger.info('construct vocabulary...')
    vocab = construct_vocab(train_file,
                    test_file,
                    output_vocab_file=os.path.join(output_dir, 'vocab.json'))
    logger.info('convert ids record to string')
    convert_record_style(train_file, vocab,
                         train_file + '.str')
    convert_record_style(test_file, vocab, test_file + '.str')
    train_file += '.str'
    test_file += '.str'
    logger.info('convert training data')
    convert_tsv_to_jsonl(train_file, os.path.join(output_dir, 'train.jsonl'))
    logger.info('convert test data')
    convert_tsv_to_jsonl(test_file, os.path.join(output_dir, 'test.jsonl'))
    logger.info('construct meta_info')
    construct_meta_info(os.path.join(output_dir, 'train.jsonl'),
                        os.path.join(output_dir, 'meta_info.pkl'))
    logger.info('construct kfold data')
    get_kfold_data(os.path.join(output_dir, 'train.jsonl'),
                   os.path.join(output_dir, 'meta_info.pkl'),
                   os.path.join(output_dir, 'kfold'),
                   n_splits=n_splits,
                   random_seed=random_seed)
    logger.info('construct enhanced data')
    construct_enhanced_data(os.path.join(output_dir, 'train.jsonl'),
                            os.path.join(output_dir, 'train_enhanced.jsonl'),)
    logger.info('construct mlm train data')
    construct_mlm_data(os.path.join(output_dir, 'train.jsonl'),
                       os.path.join(output_dir, 'test.jsonl'),
                       os.path.join(output_dir, 'train_mlm.jsonl'),
                       random_seed=random_seed)
    logger.info('constuct tokenizer training data')
    construct_tokenizer_data(os.path.join(output_dir, 'train_mlm.jsonl'),
                             os.path.join(output_dir, 'tokenizer_data.txt'))


if __name__ == '__main__':
    fire.Fire(convert_data)
