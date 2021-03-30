import pickle as pkl
from transformers import BertTokenizer
from gensim.models import KeyedVectors
from tqdm.auto import tqdm

def convert_to_ids(tokens, tokenizer):
    return tuple(tokenizer.convert_tokens_to_ids(item) for item in tokens)

def get_similar_words(token, w2v):
    results = w2v.similar_by_word(token, topn=100)
    sim_words = []
    for idx, item in enumerate(results):
        if len(sim_words) >= 15:
            break
        if len(item[0]) == len(token) and item[1] > 0.68:
            sim_words.append(item[0])
    return sim_words

def construct_ngram_meta_info(w2v_file="../user_data/data/train_data/w2v.model",
                              ngram_words_file="../user_data/data/train_data/ngram_words.pkl",
                              tokenizer_path="../user_data/tokenizer",
                              output_meta_file="../user_data/data/train_data/ngram_dict.pkl"):
    ngram_dict = {
        "ngrams": {i: [] for i in range(1, 5)},
        "ngrams_set": {},
        "sim_ngrams": {

        }
    }
    #
    tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
    w2v = KeyedVectors.load(w2v_file)
    ngram_words = pkl.load(open(ngram_words_file, 'rb'))
    all_words = list(set(list(tokenizer.get_vocab().keys()) + list(ngram_words.keys())))
    ngram_dict['ngrams_set'] = set()
    num = 0
    for word in tqdm(all_words):
        ngram_dict['ngrams_set'].add(convert_to_ids(word, tokenizer))
        if word not in w2v:
            continue
        ngram_dict['ngrams'][len(word)].append(convert_to_ids(word, tokenizer))
        sim_words = get_similar_words(word, w2v)
        if sim_words:
            num += 1
            ngram_dict['sim_ngrams'][convert_to_ids(word, tokenizer)] = []
            for sim_word in sim_words:
                ngram_dict['sim_ngrams'][convert_to_ids(word, tokenizer)].append(convert_to_ids(sim_word, tokenizer))
    print(num)
    with open(output_meta_file, 'wb') as fout:
        pkl.dump(ngram_dict, fout)

if __name__ == '__main__':
    construct_ngram_meta_info()
