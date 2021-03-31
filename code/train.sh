#!/bin/bash
# stage 1: 训练多个分类模型
bash ./data_processor/process_data.sh
bash ./model_processor/convert_official_model.sh
bash ./model_processor/convert_model.sh
bash ./train/train.sh

