import fire
import pandas as pd
from simpletransformers_addons.models.sim_text.sim_text_model import SimTextArgs, SimTextModel
from simpletransformers_addons.model_wrappers import FGMWrapper, PGDWrapper

train_df = pd.read_json('../user_data/data/train_data/train_enhanced.jsonl', lines=True)
test_df = pd.read_json('../user_data/data/train_data/kfold/0/dev.jsonl', lines=True)
test_df = test_df.iloc[0:8000]

def train_model(model_name, max_run_time=None):
    if 'large' in model_name:
        batch_size = 128
        swa_start_step = 6000
        swa_steps = 300
        learning_rate = 4.0e-5
    else:
        batch_size = 256
        swa_start_step = 3000
        swa_steps = 150
        learning_rate = 8.0e-5
    # Optional model configuration
    model_args = SimTextArgs()
    model_args.use_bimodel = False
    model_args.num_train_epochs = 3
    model_args.train_batch_size = batch_size
    model_args.eval_batch_size = 32
    model_args.evaluate_during_training = False
    model_args.evaluate_during_training_steps = 1500
    model_args.save_eval_checkpoints = False
    model_args.no_cache = False
    model_args.max_seq_length = 32
    model_args.learning_rate = learning_rate
    model_args.use_early_stopping = True
    model_args.early_stopping_metric = "auroc"
    model_args.early_stopping_metric_minimize = False
    model_args.early_stopping_consider_epochs = True
    model_args.early_stopping_patience = 2
    model_args.gradient_accumulation_steps = 1
    model_args.save_steps = 0
    model_args.save_model_every_epoch = False
    import os
    test_dir = f'../user_data/classification/{model_name}'
    model_args.tensorboard_dir = os.path.join(test_dir, 'runs')
    model_args.cache_dir = os.path.join(test_dir, 'cached')
    model_args.output_dir = os.path.join(test_dir, 'outputs')
    model_args.best_model_dir = os.path.join(test_dir, 'best')
    model_args.labels_list = [0, 1]

    #model_args.manual_seed = 124525601

    ## swa
    model_args.scheduler = "constant_schedule_with_warmup"
    model_args.use_swa = True
    # 300
    model_args.swa_steps = swa_steps
    model_args.swa_lr = learning_rate / 2.0
    model_args.config = {
        "hidden_dropout_prob": 0.1
    }
    # 5000
    model_args.swa_start_step = swa_start_step
    ##
    model_args.submodel_type = 'bert'
    # Create a ClassificationModel
    model = SimTextModel(
        "bert", f"../user_data/mlm/{model_name}/outputs", args=model_args
    )
    model = FGMWrapper(model, epsilon=0.3)

    # Train the model
    model.train_model(train_df, eval_df=test_df,
                      max_run_time=max_run_time)

if __name__ == '__main__':
    fire.Fire(train_model)
