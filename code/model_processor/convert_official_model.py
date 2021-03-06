from transformers import BertForMaskedLM, BertConfig

print('convert uer base model...')
config = BertConfig(vocab_size=21128,
                   hidden_size=768,
                   num_hidden_layers=12,
                   num_attention_heads=12,
                   intermediate_size=3072)
model = BertForMaskedLM.from_pretrained('../user_data/official_model/transformers/bert_base.bin',
                                       config=config)
model.save_pretrained('../user_data/official_model/transformers/bert-base')
print('convert uer large model...')
config = BertConfig(vocab_size=21128,
                   hidden_size=1024,
                   num_hidden_layers=24,
                   num_attention_heads=16,
                   intermediate_size=4096)
model = BertForMaskedLM.from_pretrained('../user_data/official_model/transformers/bert_large.bin',
                                       config=config)
model.save_pretrained('../user_data/official_model/transformers/bert-large')
