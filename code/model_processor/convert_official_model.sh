echo "convert uer base model..."
python model_processor/convert_uer_to_transformers.py \
       --input_model_path=../user_data/official_model/download/mixed_corpus_bert_base_model.bin \
       --output_model_path=../user_data/official_model/transformers/bert_base.bin
       --layers_num=12 \
       --target=mlm
echo "convert uer large model..."
python model_processor/convert_uer_to_transformers.py \
       --input_model_path=../user_data/official_model/download/mixed_corpus_bert_large_model.bin \
       --output_model_path=../user_data/official_model/transformers/bert_large.bin \
       --layers_num=24 \
       --target=mlm
echo "convert official models..."
python model_processor/convert_official_model.py
