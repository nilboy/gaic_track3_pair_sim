python model_processor/convert_transformers_model.py --input_model_name=../user_data/official_model/transformers/bert-base \
                        --output_model_name=../user_data/pretrained/bert-base

python model_processor/convert_transformers_model.py --input_model_name=../user_data/official_model/transformers/bert-large \
                        --output_model_name=../user_data/pretrained/bert-large

python model_processor/convert_transformers_model.py --input_model_name=../user_data/official_model/transformers/nezha-base \
                        --output_model_name=../user_data/pretrained/nezha-base \
                        --model_type=nezha

python model_processor/convert_transformers_model.py --input_model_name=../user_data/official_model/transformers/nezha-large \
                        --output_model_name=../user_data/pretrained/nezha-large \
                        --model_type=nezha

python model_processor/convert_transformers_model.py --input_model_name=hfl/chinese-macbert-base \
                        --output_model_name=../user_data/pretrained/macbert-base \
                        --model_type=bert

python model_processor/convert_transformers_model.py --input_model_name=hfl/chinese-macbert-large \
                        --output_model_name=../user_data/pretrained/macbert-large \
                        --model_type=bert
