from simpletransformers_addons.models.sim_text.transformer_models.ensemble_model import EnsembleModel

ensemble_model = EnsembleModel.init_from_models_path([
    "../user_data/mlm/nezha-large/outputs",
    "../user_data/mlm/bert-large/outputs",
], ["nezha", 'bert'])
ensemble_model.save_pretrained("../user_data/mlm/ensemble")
from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained("../user_data/mlm/bert-large/outputs")
tokenizer.save_pretrained("../user_data/mlm/ensemble")
