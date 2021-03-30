# gai_track3_pair_sim

## 预训练模型准备
* 下载预训练模型
    - nezha-base:
      
      https://drive.google.com/file/d/1HmwMG2ldojJRgMVN0ZhxqOukhuOBOKUb/view?usp=sharing
    - nezha-large:
      
      https://drive.google.com/file/d/1EtahNvdjEpugm8juFuPIN_Fs2skFmeMU/view?usp=sharing
    - uer/bert-base:
      
      https://share.weiyun.com/5QOzPqq
    - uer/bert-large:
    
      https://share.weiyun.com/5G90sMJ
    - macbert, chinese-bert-wwm-ext, chinese-roberta-wwm-ext-large
    
      https://huggingface.co/models
* 预训练模型开源仓库
    - https://github.com/dbiir/UER-py
    - https://github.com/huawei-noah/Pretrained-Language-Model
* 下载并解压, 解压到文件夹 data, 文件夹结构如下:
    ```
    data/
    └── official_model
        └── download
            ├── chinese-bert-wwm-ext
            │   ├── added_tokens.json
            │   ├── config.json
            │   ├── pytorch_model.bin
            │   ├── special_tokens_map.json
            │   ├── tokenizer_config.json
            │   └── vocab.txt
            ├── chinese-roberta-wwm-ext-large
            │   ├── config.json
            │   ├── pytorch_model.bin
            │   ├── special_tokens_map.json
            │   ├── tokenizer.json
            │   ├── tokenizer_config.json
            │   └── vocab.txt
            ├── macbert-base
            │   ├── added_tokens.json
            │   ├── config.json
            │   ├── pytorch_model.bin
            │   ├── special_tokens_map.json
            │   ├── tokenizer.json
            │   ├── tokenizer_config.json
            │   └── vocab.txt
            ├── macbert-large
            │   ├── added_tokens.json
            │   ├── config.json
            │   ├── pytorch_model.bin
            │   ├── special_tokens_map.json
            │   ├── tokenizer.json
            │   ├── tokenizer_config.json
            │   └── vocab.txt
            ├── mixed_corpus_bert_base_model.bin
            ├── mixed_corpus_bert_large_model.bin
            └── nezha-cn-base
                ├── bert_config.json
                ├── pytorch_model.bin
                └── vocab.txt
    ```
* 预训练模型[md5](user_data/md5.txt)

## 环境准备
* torch==1.7.0
* transformers=4.3.0.rc1
* simpletransformers==0.51.15

## 端到端训练脚本
```
bash ./run.sh
```
