# gaic_track3_pair_sim

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
* TensorRT-7.2.1.6

## 端到端训练脚本
```
cd code
bash ./run.sh
```
## 不同版本方案

* 方案一: 预训练(多个模型) + finetune-分类(多个模型) + 生成软标签 + 训练regression模型(软标签，单模型)
    ```
    cd code
    bash ./train.sh
    ```
    初赛使用的该方案，初赛成绩为0.9220；

* 方案二: 预训练(多个模型) + 加载预训练参数，初始化一个大模型 + 训练分类模型(单模型)
    ```
    pipeline/pipeline_b.py
    ```
    训练一个144层模型(6 * 12 + 24 * 3);
  
    该模型单模型在复赛A榜成绩0.9561；推理平均时间15ms；

* 方案三: 预训练(多个模型) + finetune-分类(多个模型) + 平均融合
    ```
    pipeline/pipeline_d.py
    ```
    融合6个bert-base + 3个bert-large模型；
    
    该模型在复赛A榜没测试，B榜成绩0.9593；推理平均时间15ms；
