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
* 预训练模型开源仓库
    - https://github.com/dbiir/UER-py
    - https://github.com/huawei-noah/Pretrained-Language-Model
* 下载并解压, 解压到文件夹user_data/official_model/download, 文件夹结构如下:
    ```
    .
    ├── mixed_corpus_bert_base_model.bin
    ├── mixed_corpus_bert_large_model.bin
    ├── nezha-cn-base
    │   ├── bert_config.json
    │   ├── pytorch_model.bin
    │   └── vocab.txt
    └── nezha-large
        ├── bert_config.json
        ├── pytorch_model.bin
        └── vocab.txt
    ```
* 预训练模型[md5](user_data/md5.txt)

## 环境准备
* torch==1.6.0
* transformers=4.3.0.rc1
* simpletransformers==0.51.15

## 训练
```
cd code
bash ./train.sh
```
## 预测
```
cd code
bash ./test.sh
```
