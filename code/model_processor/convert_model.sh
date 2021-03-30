python model_processor/convert_transformers_model.py --input_model_name=../user_data/official_model/transformers/bert-base \
                        --output_model_name=../user_data/pretrained/bert-base

python model_processor/convert_transformers_model.py --input_model_name=../user_data/official_model/transformers/bert-large \
                        --output_model_name=../user_data/pretrained/bert-large

python model_processor/convert_transformers_model.py --input_model_name=../data/official_model/download/macbert-base \
                        --output_model_name=../user_data/pretrained/macbert-base \
                        --model_type=bert

python model_processor/convert_transformers_model.py --input_model_name=../data/official_model/download/macbert-large \
                        --output_model_name=../user_data/pretrained/macbert-large \
                        --model_type=bert

python model_processor/convert_transformers_model.py --input_model_name=../data/official_model/download/chinese-bert-wwm-ext \
                        --output_model_name=../user_data/pretrained/roberta-base \
                        --model_type=bert

python model_processor/convert_transformers_model.py --input_model_name=../data/official_model/download/chinese-roberta-wwm-ext-large \
                        --output_model_name=../user_data/pretrained/roberta-large \
                        --model_type=bert
