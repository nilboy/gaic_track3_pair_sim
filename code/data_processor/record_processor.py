import json

class RecordProcessor(object):
    def __init__(self,
                 normal_vocab_path,
                 idmap_path,
                 vocab_path,
                 min_frequence=5,
                 min_oov_word_idx=35000):
        with open(normal_vocab_path) as fin:
            self.normal_vocab = json.load(fin)
        with open(idmap_path) as fin:
            self.idmap = json.load(fin)
        self.min_frequence = min_frequence
        self.min_oov_word_idx = min_oov_word_idx
        with open(vocab_path) as fin:
            self.vocab = json.load(fin)

    def process_oov(self, text_a, text_b):
        tokens_a = text_a.split()
        tokens_b = text_b.split()
        oov_word_map = {}
        cur_oov_idx = self.min_oov_word_idx
        for tokens in [tokens_a, tokens_b]:
            for i in range(len(tokens)):
                if self.normal_vocab.get(tokens[i], 0) < self.min_frequence:
                    if tokens[i] not in oov_word_map:
                        oov_word_map[tokens[i]] = str(cur_oov_idx)
                        cur_oov_idx += 1
                    tokens[i] = oov_word_map[tokens[i]]
                else:
                    tokens[i] = self.idmap[tokens[i]]
        return " ".join(tokens_a), " ".join(tokens_b)

    def process_record(self, text_a, text_b):
        text_a, text_b = self.process_oov(text_a, text_b)
        def convert_str_style(input_str):
            token_ids = [item for item in input_str.split() if item]
            tokens = [self.vocab['num2token'][idx] for idx in token_ids]
            return ''.join(tokens)
        text_a = convert_str_style(text_a)
        text_b = convert_str_style(text_b)
        return text_a, text_b

