mkdir -p ../user_data/ensemble/tensorrt
python deploy/build_tensorrt_engine_2.py \
    -s 32 \
    -w 8192 \
    -m ../user_data/ensemble/outputs/pytorch_model.bin \
    -o ../user_data/ensemble/tensorrt/model-128.engine \
    -c ../user_data/classification \
    -f
