export PATH=/root/libs/cuda-11.0/bin:$PATH
export LD_LIBRARY_PATH=/root/libs/cuda-11.0/lib64:/root/libs/cudnn-8.0.5/lib64:/root/libs/TensorRT-7.2.1.6/lib:$LD_LIBRARY_PATH
export LANG=C.UTF-8
export PYTHONPATH=.:$PYTHONPATH
bash run_inner_2.sh
