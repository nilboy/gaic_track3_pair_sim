import fire
import pandas as pd
from simpletransformers_addons.models.sim_text.sim_text_model import SimTextArgs, SimTextModel
from simpletransformers_addons.model_wrappers import FGMWrapper, PGDWrapper

train_df = pd.read_json('../user_data/data/train_data/train_enhanced.jsonl', lines=True)
test_df = pd.read_json('../user_data/data/train_data/kfold/0/dev.jsonl', lines=True)
test_df = test_df.iloc[0:8000]

# Optional model configuration
model_args = SimTextArgs()
model_args.use_bimodel = False
model_args.num_train_epochs=3
model_args.train_batch_size = 128
model_args.eval_batch_size = 32
model_args.evaluate_during_training = False
model_args.evaluate_during_training_steps = 1500
model_args.save_eval_checkpoints = False
model_args.no_cache = False
model_args.max_seq_length = 32
model_args.learning_rate = 4e-5
model_args.use_early_stopping = True
model_args.early_stopping_metric = "auroc"
model_args.early_stopping_metric_minimize = False
model_args.early_stopping_consider_epochs = True
model_args.early_stopping_patience = 2
model_args.gradient_accumulation_steps = 1
model_args.save_steps = 0
model_args.save_model_every_epoch = False
import os
test_dir = '../user_data/ensemble'
model_args.tensorboard_dir = os.path.join(test_dir, 'runs')
model_args.cache_dir = os.path.join(test_dir, 'cached')
model_args.output_dir = os.path.join(test_dir, 'outputs')
model_args.best_model_dir = os.path.join(test_dir, 'best')
model_args.labels_list = [0, 1]

model_args.manual_seed = 124525601

## swa
model_args.scheduler = "constant_schedule_with_warmup"
model_args.use_swa = True
# 300
model_args.swa_steps = 300
model_args.swa_lr =2.0e-5
model_args.config = {
    "hidden_dropout_prob": 0.1
}
# 5000
model_args.swa_start_step = 6000
##
model_args.submodel_type = 'ensemble'
# Create a ClassificationModel
model = SimTextModel(
    "bert", "../user_data/mlm/ensemble", args=model_args
)

model = FGMWrapper(model, epsilon=0.3)

def train_model(max_run_time=None):
    # Train the model
    model.train_model(train_df, eval_df=test_df,
                      max_run_time=max_run_time)

if __name__ == '__main__':
    fire.Fire(train_model)
