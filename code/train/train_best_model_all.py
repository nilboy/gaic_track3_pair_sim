import pandas as pd
from simpletransformers_addons.models.sim_text.sim_text_model import SimTextArgs, SimTextModel
from simpletransformers_addons.model_wrappers import FGMWrapper, PGDWrapper

train_df = pd.read_json('../user_data/data/train_data/kfold/0/train_enhanced.jsonl', lines=True)

# Optional model configuration
model_args = SimTextArgs()
model_args.use_bimodel = True
model_args.num_train_epochs=3
model_args.train_batch_size = 128
model_args.eval_batch_size = 64
model_args.evaluate_during_training = True
model_args.evaluate_during_training_steps = 300
model_args.no_cache = False
model_args.max_seq_length = 32
model_args.learning_rate = 8e-5
model_args.use_early_stopping = True
model_args.early_stopping_metric = "auroc"
model_args.early_stopping_metric_minimize = False
model_args.early_stopping_consider_epochs = True
model_args.early_stopping_patience = 3
model_args.gradient_accumulation_steps = 1
model_args.save_steps = 0
model_args.save_model_every_epoch = False
import os
test_dir = '../user_data/dev/ensemble-fgm'
model_args.tensorboard_dir = os.path.join(test_dir, 'runs')
model_args.cache_dir = os.path.join(test_dir, 'cached')
model_args.output_dir = os.path.join(test_dir, 'outputs')
model_args.best_model_dir = os.path.join(test_dir, 'best')
model_args.labels_list = [0, 1]

# model_args.fp16 = False
model_args.manual_seed = 124525601

# model_args.config = {
#     "use_mean_pooling": True
# }

## swa
model_args.scheduler = "constant_schedule_with_warmup"
model_args.use_swa = True
# 300
model_args.swa_steps = 100
model_args.swa_lr =4.0e-5
# 5000
model_args.swa_start_step = 1000
##
model_args.submodel_type = 'ensemble'
#model_args.submodel_type = 'ensemble'
# model_args.use_diff = True
# model_args.optimizer = 'lookahead'
# Create a ClassificationModel
model = SimTextModel(
    "bert", "../user_data/mlm/ensemble-fgm", args=model_args
)

model = FGMWrapper(model, epsilon=0.3)

# model = PGDWrapper(model,
#                    epsilon=0.6,
#                    alpha=0.2)

# Train the model
model.train_model(train_df, eval_df=test_df)

