#!/bin/bash
# stage 1: 训练多个分类模型
bash ./data_processor/process_data.sh
bash ./model_processor/convert_official_model.sh
bash ./model_processor/convert_model.sh
bash ./train/train.sh
# stage 2: 训练kfold回归模型
bash ./data_processor/construct_regression_data.sh
python model_processor/build_ensemble_model.py
bash ./train/train_regression.sh ensemble
# stage 3: 训练全量数据回归模型
bash ./data_processor/construct_regression_all_data.sh
python train/train_regression_all.py
