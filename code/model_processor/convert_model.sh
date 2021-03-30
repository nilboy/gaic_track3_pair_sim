python model_processor/convert_transformers_model.py --input_model_name=/nfs/users/jiangxinghua/models/transformers/uer/bert_base \
                        --output_model_name=../user_data/pretrained/bert-base

python model_processor/convert_transformers_model.py --input_model_name=/nfs/users/jiangxinghua/models/transformers/uer/bert_large \
                        --output_model_name=../user_data/pretrained/bert-large

python model_processor/convert_transformers_model.py --input_model_name=/nfs/users/jiangxinghua/models/transformers/nezha/nezha-cn-base \
                        --output_model_name=../user_data/pretrained/nezha-base \
                        --model_type=nezha

python model_processor/convert_transformers_model.py --input_model_name=/nfs/users/jiangxinghua/models/transformers/nezha/nezha-large \
                        --output_model_name=../user_data/pretrained/nezha-large \
                        --model_type=nezha

