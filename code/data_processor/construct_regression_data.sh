#!/bin/bash
python data_processor/construct_soft_label.py
python data_processor/merge_predicts.py
python data_processor/construct_regression_data.py
