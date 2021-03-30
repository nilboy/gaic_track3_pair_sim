#!/bin/bash
python data_processor/construct_soft_label.py --task=regression
python data_processor/construct_regression_data.py --task=regression
