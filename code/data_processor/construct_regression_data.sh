#!/bin/bash
python data_processor/construct_soft_label.py --model_num=4
python data_processor/construct_regression_data.py
