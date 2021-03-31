#!/bin/bash
bash ./data_processor/process_data.sh
bash ./model_processor/convert_official_model.sh
bash ./model_processor/convert_model.sh
bash ./train/train.sh

