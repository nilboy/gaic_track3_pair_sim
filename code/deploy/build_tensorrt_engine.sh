mkdir -p ../user_data/ensemble/tensorrt
python deploy/build_tensorrt_engine.py \
    -m ../user_data/ensemble/base/best/pytorch_model.bin \
    -o ../user_data/ensemble/tensorrt/model-128.engine \
    -c ../user_data/ensemble/base/best \
    -f
cp ../user_data/data/normal_vocab.json ../user_data/ensemble/tensorrt/normal_vocab.json
cp ../user_data/data/train_data/vocab.json ../user_data/ensemble/tensorrt/vocab.json
cp ../user_data/ensemble/base/best/vocab.txt ../user_data/ensemble/tensorrt/vocab.txt
cp ../user_data/ensemble/base/best/tokenizer_config.json ../user_data/ensemble/tensorrt/tokenizer_config.json
cp ../user_data/ensemble/base/best/special_tokens_map.json ../user_data/ensemble/tensorrt/special_tokens_map.json
