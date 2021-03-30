mkdir -p ../user_data/ensemble/tensorrt
python deploy/build_tensorrt_engine_2.py \
    -s 32 \
    -w 8192 \
    -m ../user_data/ensemble/outputs/pytorch_model.bin \
    -o ../user_data/ensemble/tensorrt/model-128.engine \
    -c ../user_data/classification \
    -f
cp ../user_data/data/normal_vocab.json ../user_data/ensemble/tensorrt/normal_vocab.json
cp ../user_data/data/idmap.json ../user_data/ensemble/tensorrt/idmap.json
cp ../user_data/data/train_data/vocab.json ../user_data/ensemble/tensorrt/vocab.json
cp ../user_data/classification/bert-base/outputs/vocab.txt ../user_data/ensemble/tensorrt/vocab.txt
cp ../user_data/classification/bert-base/outputs/tokenizer_config.json ../user_data/ensemble/tensorrt/tokenizer_config.json
cp ../user_data/classification/bert-base/outputs/special_tokens_map.json ../user_data/ensemble/tensorrt/special_tokens_map.json
