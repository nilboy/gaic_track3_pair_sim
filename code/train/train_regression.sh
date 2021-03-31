#!/bin/bash
model_name=$1
for data_id in {0..4}
  do
    python train/train_best_pipeline_regression.py --data_id=$data_id \
                                        --model_name=$model_name \
                                        --model_num=1
  done
