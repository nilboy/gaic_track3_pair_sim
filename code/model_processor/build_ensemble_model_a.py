from simpletransformers_addons.models.sim_text.transformer_models.ensemble_model import EnsembleModel

ensemble_model = EnsembleModel.init_from_models_path([
    "../user_data/mlm/bert-base_fgm/outputs",
    "../user_data/mlm/bert-base_fgm/outputs",
    "../user_data/mlm/macbert-base_fgm/outputs",
    "../user_data/mlm/bert-large/outputs",
    "../user_data/mlm/macbert-large/outputs",
    "../user_data/mlm/bert-large/outputs"
], ["bert"]*6)
ensemble_model.save_pretrained("../user_data/mlm/ensemble")
from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained("../user_data/mlm/bert-large/outputs")
tokenizer.save_pretrained("../user_data/mlm/ensemble")
