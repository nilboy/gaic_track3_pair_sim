import sys

import pandas as pd
from simpletransformers_addons.models.sim_text.sim_text_model import SimTextArgs, SimTextModel
from simpletransformers_addons.model_wrappers import FGMWrapper, PGDWrapper

from sklearn.metrics import roc_curve, auc, accuracy_score

def manual_auc(labels, preds):
    labels = [1 if item >= 0.5 else 0 for item in labels]
    fpr, tpr, thresholds = roc_curve(labels, preds)
    auroc = auc(fpr, tpr)
    return auroc

train_df = pd.read_json('../user_data/data/train_data/kfold/0/train_regression.jsonl', lines=True)
test_df = pd.read_json('../user_data/data/train_data/kfold/0/dev.jsonl', lines=True)
# Optional model configuration
model_args = SimTextArgs()
model_args.num_train_epochs=1
model_args.train_batch_size = 64
model_args.eval_batch_size = 64
model_args.evaluate_during_training = True
model_args.no_cache = False
model_args.max_seq_length = 32
model_args.learning_rate = 2e-5
model_args.use_early_stopping = True
model_args.early_stopping_metric = "auroc"
model_args.early_stopping_metric_minimize = False
model_args.early_stopping_consider_epochs = True
model_args.early_stopping_patience = 6
model_args.gradient_accumulation_steps = 1
# save
model_args.save_steps = 0
model_args.evaluate_during_training_steps = 1000
model_args.save_eval_checkpoints = True
model_args.save_model_every_epoch = True

#
model_args.scheduler = "constant_schedule_with_warmup"
model_args.use_swa = True
model_args.swa_steps = 300
model_args.swa_lr = 2.0e-5
model_args.swa_start_step = 5000
#

import os
test_dir = '../user_data/ensemble/base'
model_args.tensorboard_dir = os.path.join(test_dir, 'runs')
model_args.cache_dir = os.path.join(test_dir, 'cached')
model_args.output_dir = os.path.join(test_dir, 'outputs')
model_args.best_model_dir = os.path.join(test_dir, 'best')
model_args.temperature = 1.0
model_args.regression = True
model_args.use_bimodel = False
model_args.manual_seed = 124525601

model_args.submodel_type = 'ensemble'

# Create a ClassificationModel
model = SimTextModel(
    "bert", "../user_data/mlm/ensemble",
    num_labels=1, args=model_args
)

# Train the model
model.train_model(train_df, eval_df=test_df,
                  auroc=manual_auc)
