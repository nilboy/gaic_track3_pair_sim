#!/usr/bin/env python
# coding: utf-8


from __future__ import absolute_import, division, print_function

import json
import logging
import math
import os
import random
import warnings
from dataclasses import asdict, dataclass
from multiprocessing import cpu_count
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm.auto import tqdm, trange
from tqdm.contrib import tenumerate
from transformers.optimization import (
    get_constant_schedule,
    get_constant_schedule_with_warmup,
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_cosine_with_hard_restarts_schedule_with_warmup,
    get_polynomial_decay_schedule_with_warmup,
)
from transformers.optimization import AdamW, Adafactor
from transformers import (
    AlbertConfig,
    AlbertTokenizer,
    BertConfig,
    BertTokenizer,
    BertweetTokenizer,
    CamembertConfig,
    CamembertTokenizer,
    DistilBertConfig,
    DistilBertTokenizer,
    ElectraConfig,
    ElectraTokenizer,
    FlaubertConfig,
    FlaubertTokenizer,
    LayoutLMConfig,
    LayoutLMTokenizer,
    LongformerConfig,
    LongformerForSequenceClassification,
    LongformerTokenizer,
    MobileBertConfig,
    MobileBertForSequenceClassification,
    MobileBertTokenizer,
    RobertaConfig,
    RobertaTokenizer,
    WEIGHTS_NAME,
    XLMConfig,
    XLMRobertaConfig,
    XLMRobertaTokenizer,
    XLMTokenizer,
    XLNetConfig,
    XLNetTokenizer,
)
from transformers.convert_graph_to_onnx import convert, quantize

from simpletransformers.classification.classification_utils import (
    InputExample,
    LazyClassificationDataset
)
from simpletransformers.classification.transformer_models.albert_model import AlbertForSequenceClassification
from simpletransformers_addons.models.text_pair.transformer_models.bert_model import BertForTextPair
from simpletransformers.classification.transformer_models.camembert_model import CamembertForSequenceClassification
from simpletransformers.classification.transformer_models.distilbert_model import DistilBertForSequenceClassification
from simpletransformers.classification.transformer_models.flaubert_model import FlaubertForSequenceClassification
from simpletransformers.classification.transformer_models.layoutlm_model import LayoutLMForSequenceClassification
from simpletransformers.classification.transformer_models.roberta_model import RobertaForSequenceClassification
from simpletransformers.classification.transformer_models.xlm_model import XLMForSequenceClassification
from simpletransformers.classification.transformer_models.xlm_roberta_model import XLMRobertaForSequenceClassification
from simpletransformers.classification.transformer_models.xlnet_model import XLNetForSequenceClassification
from simpletransformers.config.global_args import global_args
from simpletransformers.config.model_args import ClassificationArgs
from simpletransformers.config.utils import sweep_config_to_sweep_values
from simpletransformers.custom_models.models import ElectraForSequenceClassification

from simpletransformers_addons.models.text_pair.text_pair_utils import discrete_auc, convert_examples_to_features

try:
    import wandb

    wandb_available = True
except ImportError:
    wandb_available = False

logger = logging.getLogger(__name__)

@dataclass
class TextPairArgs(ClassificationArgs):
    margin: float = 0.1
    # 混合loss参数.
    loss_alpha: float = 1.0
    # 第一阶段，单样本训练epoch
    stage1_epoch_num: int = 0
    # 难负样本挖掘
    hard_sample_mining_margin: float = 0.1
    # 单个正样本，最大负样本数
    max_neg_nums: int = 0

class TextPairModel:
    def __init__(
        self,
        model_type,
        model_name,
        args=None,
        use_cuda=True,
        cuda_device=-1,
        onnx_execution_provider=None,
        is_continue_training=True,
        **kwargs
    ):

        """
        Initializes a TextPairModel model.

        Args:
            model_type: The type of model (bert, xlnet, xlm, roberta, distilbert)
            model_name: The exact architecture and trained weights to use. This may be a Hugging Face Transformers compatible pre-trained model, a community model, or the path to a directory containing model files.
            args (optional): Default args will be used if this parameter is not provided. If provided, it should be a dict containing the args that should be changed in the default args.
            use_cuda (optional): Use GPU if available. Setting to False will force model to use CPU only.
            cuda_device (optional): Specific GPU that should be used. Will use the first available GPU by default.
            onnx_execution_provider (optional): ExecutionProvider to use with ONNX Runtime. Will use CUDA (if use_cuda) or CPU (if use_cuda is False) by default.
            **kwargs (optional): For providing proxies, force_download, resume_download, cache_dir and other options specific to the 'from_pretrained' implementation where this will be supplied.
        """  # noqa: ignore flake8"

        MODEL_CLASSES = {
            "albert": (AlbertConfig, AlbertForSequenceClassification, AlbertTokenizer),
            "bert": (BertConfig, BertForTextPair, BertTokenizer),
            "bertweet": (RobertaConfig, RobertaForSequenceClassification, BertweetTokenizer),
            "camembert": (CamembertConfig, CamembertForSequenceClassification, CamembertTokenizer),
            "distilbert": (DistilBertConfig, DistilBertForSequenceClassification, DistilBertTokenizer),
            "electra": (ElectraConfig, ElectraForSequenceClassification, ElectraTokenizer),
            "flaubert": (FlaubertConfig, FlaubertForSequenceClassification, FlaubertTokenizer),
            "layoutlm": (LayoutLMConfig, LayoutLMForSequenceClassification, LayoutLMTokenizer),
            "longformer": (LongformerConfig, LongformerForSequenceClassification, LongformerTokenizer),
            "mobilebert": (MobileBertConfig, MobileBertForSequenceClassification, MobileBertTokenizer),
            "roberta": (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer),
            "xlm": (XLMConfig, XLMForSequenceClassification, XLMTokenizer),
            "xlmroberta": (XLMRobertaConfig, XLMRobertaForSequenceClassification, XLMRobertaTokenizer),
            "xlnet": (XLNetConfig, XLNetForSequenceClassification, XLNetTokenizer),
        }
        self.is_continue_training = is_continue_training
        if is_continue_training:
            self.args = self._load_model_args(model_name)
        else:
            self.args = self._load_model_args(None)

        if isinstance(args, dict):
            self.args.update_from_dict(args)
        elif isinstance(args, TextPairArgs):
            self.args = args

        if self.args.thread_count:
            torch.set_num_threads(self.args.thread_count)

        if "sweep_config" in kwargs:
            self.is_sweeping = True
            sweep_config = kwargs.pop("sweep_config")
            sweep_values = sweep_config_to_sweep_values(sweep_config)
            self.args.update_from_dict(sweep_values)
        else:
            self.is_sweeping = False

        if self.args.manual_seed:
            random.seed(self.args.manual_seed)
            np.random.seed(self.args.manual_seed)
            torch.manual_seed(self.args.manual_seed)
            if self.args.n_gpu > 0:
                torch.cuda.manual_seed_all(self.args.manual_seed)

        config_class, model_class, tokenizer_class = MODEL_CLASSES[model_type]
        num_labels = 2
        self.config = config_class.from_pretrained(model_name, num_labels=num_labels, **self.args.config)
        self.num_labels = num_labels

        if use_cuda:
            if torch.cuda.is_available():
                if cuda_device == -1:
                    self.device = torch.device("cuda")
                else:
                    self.device = torch.device(f"cuda:{cuda_device}")
            else:
                raise ValueError(
                    "'use_cuda' set to True when cuda is unavailable."
                    " Make sure CUDA is available or set use_cuda=False."
                )
        else:
            self.device = "cpu"
        self.weight = None
        if self.args.onnx:
            from onnxruntime import InferenceSession, SessionOptions

            if not onnx_execution_provider:
                onnx_execution_provider = "CUDAExecutionProvider" if use_cuda else "CPUExecutionProvider"

            options = SessionOptions()
            options.intra_op_num_threads = 1

            if self.args.dynamic_quantize:
                model_path = quantize(Path(os.path.join(model_name, "onnx_model.onnx")))
                self.model = InferenceSession(model_path.as_posix(), options, providers=[onnx_execution_provider])
            else:
                model_path = os.path.join(model_name, "onnx_model.onnx")
                self.model = InferenceSession(model_path, options, providers=[onnx_execution_provider])
        else:
            if not self.args.quantized_model:
                if self.weight:
                    self.model = model_class.from_pretrained(
                        model_name, config=self.config, weight=torch.Tensor(self.weight).to(self.device), **kwargs,
                    )
                else:
                    self.model = model_class.from_pretrained(model_name, config=self.config, **kwargs)
            else:
                quantized_weights = torch.load(os.path.join(model_name, "pytorch_model.bin"))
                if self.weight:
                    self.model = model_class.from_pretrained(
                        None,
                        config=self.config,
                        state_dict=quantized_weights,
                        weight=torch.Tensor(self.weight).to(self.device),
                    )
                else:
                    self.model = model_class.from_pretrained(None, config=self.config, state_dict=quantized_weights)

            if self.args.dynamic_quantize:
                self.model = torch.quantization.quantize_dynamic(self.model, {torch.nn.Linear}, dtype=torch.qint8)
            if self.args.quantized_model:
                self.model.load_state_dict(quantized_weights)
            if self.args.dynamic_quantize:
                self.args.quantized_model = True

        self.results = {}

        if not use_cuda:
            self.args.fp16 = False

        if self.args.fp16:
            try:
                from torch.cuda import amp
            except AttributeError:
                raise AttributeError("fp16 requires Pytorch >= 1.6. Please update Pytorch or turn off fp16.")

        if model_name in [
            "vinai/bertweet-base",
            "vinai/bertweet-covid19-base-cased",
            "vinai/bertweet-covid19-base-uncased",
        ]:
            self.tokenizer = tokenizer_class.from_pretrained(
                model_name, do_lower_case=self.args.do_lower_case, normalization=True, **kwargs
            )
        else:
            self.tokenizer = tokenizer_class.from_pretrained(
                model_name, do_lower_case=self.args.do_lower_case, **kwargs
            )

        if self.args.special_tokens_list:
            self.tokenizer.add_tokens(self.args.special_tokens_list, special_tokens=True)
            self.model.resize_token_embeddings(len(self.tokenizer))

        self.args.model_name = model_name
        self.args.model_type = model_type

        if model_type in ["camembert", "xlmroberta"]:
            warnings.warn(
                f"use_multiprocessing automatically disabled as {model_type}"
                " fails when using multiprocessing for feature conversion."
            )
            self.args.use_multiprocessing = False

        if self.args.wandb_project and not wandb_available:
            warnings.warn("wandb_project specified but wandb is not available. Wandb disabled.")
            self.args.wandb_project = None

    def train_model(
        self,
        train_df,
        multi_label=False,
        output_dir=None,
        show_running_loss=True,
        args=None,
        eval_df=None,
        verbose=True,
        **kwargs,
    ):
        """
        Trains the model using 'train_df'

        Args:
            train_df: Pandas Dataframe containing at least two columns. If the Dataframe has a header, it should contain a 'text' and a 'labels' column. If no header is present,
            the Dataframe should contain at least two columns, with the first column containing the text, and the second column containing the label. The model will be trained on this Dataframe.
            output_dir: The directory where model files will be saved. If not given, self.args.output_dir will be used.
            show_running_loss (optional): Set to False to prevent running loss from being printed to console. Defaults to True.
            args (optional): Optional changes to the args dict of the model. Any changes made will persist for the model.
            eval_df (optional): A DataFrame against which evaluation will be performed when evaluate_during_training is enabled. Is required if evaluate_during_training is enabled.
            **kwargs: Additional metrics that should be used. Pass in the metrics as keyword arguments (name of metric: function to use). E.g. f1=sklearn.metrics.f1_score.
                        A metric function should take in two parameters. The first parameter will be the true labels, and the second parameter will be the predictions.

        Returns:
            global_step: Number of global steps trained
            training_details: Average training loss if evaluate_during_training is False or full training progress scores if evaluate_during_training is True
        """  # noqa: ignore flake8"

        if args:
            self.args.update_from_dict(args)

        if self.args.silent:
            show_running_loss = False

        if self.args.evaluate_during_training and eval_df is None:
            raise ValueError(
                "evaluate_during_training is enabled but eval_df is not specified."
                " Pass eval_df to model.train_model() if using evaluate_during_training."
            )

        if not output_dir:
            output_dir = self.args.output_dir

        if os.path.exists(output_dir) and os.listdir(output_dir) and not self.args.overwrite_output_dir:
            raise ValueError(
                "Output directory ({}) already exists and is not empty."
                " Set overwrite_output_dir: True to automatically overwrite.".format(output_dir)
            )
        self._move_model_to_device()

        os.makedirs(output_dir, exist_ok=True)

        global_step, training_details = self.train(
            train_df,
            output_dir,
            multi_label=multi_label,
            show_running_loss=show_running_loss,
            eval_df=eval_df,
            verbose=verbose,
            **kwargs,
        )

        self.save_model(model=self.model)

        if verbose:
            logger.info(" Training of {} model complete. Saved to {}.".format(self.args.model_type, output_dir))

        return global_step, training_details

    def train(
        self,
        train_df,
        output_dir,
        multi_label=False,
        show_running_loss=True,
        eval_df=None,
        verbose=True,
        **kwargs,
    ):
        """
        Trains the model on train_dataset.

        Utility function to be used by the train_model() method. Not intended to be used directly.
        """

        model = self.model
        args = self.args

        tb_writer = SummaryWriter(logdir=args.tensorboard_dir)

        t_total = len(train_df) // args.gradient_accumulation_steps * args.num_train_epochs

        no_decay = ["bias", "LayerNorm.weight"]

        optimizer_grouped_parameters = []
        custom_parameter_names = set()
        for group in self.args.custom_parameter_groups:
            params = group.pop("params")
            custom_parameter_names.update(params)
            param_group = {**group}
            param_group["params"] = [p for n, p in model.named_parameters() if n in params]
            optimizer_grouped_parameters.append(param_group)

        for group in self.args.custom_layer_parameters:
            layer_number = group.pop("layer")
            layer = f"layer.{layer_number}."
            group_d = {**group}
            group_nd = {**group}
            group_nd["weight_decay"] = 0.0
            params_d = []
            params_nd = []
            for n, p in model.named_parameters():
                if n not in custom_parameter_names and layer in n:
                    if any(nd in n for nd in no_decay):
                        params_nd.append(p)
                    else:
                        params_d.append(p)
                    custom_parameter_names.add(n)
            group_d["params"] = params_d
            group_nd["params"] = params_nd

            optimizer_grouped_parameters.append(group_d)
            optimizer_grouped_parameters.append(group_nd)

        if not self.args.train_custom_parameters_only:
            optimizer_grouped_parameters.extend(
                [
                    {
                        "params": [
                            p
                            for n, p in model.named_parameters()
                            if n not in custom_parameter_names and not any(nd in n for nd in no_decay)
                        ],
                        "weight_decay": args.weight_decay,
                    },
                    {
                        "params": [
                            p
                            for n, p in model.named_parameters()
                            if n not in custom_parameter_names and any(nd in n for nd in no_decay)
                        ],
                        "weight_decay": 0.0,
                    },
                ]
            )

        warmup_steps = math.ceil(t_total * args.warmup_ratio)
        args.warmup_steps = warmup_steps if args.warmup_steps == 0 else args.warmup_steps

        if args.optimizer == "AdamW":
            optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        elif args.optimizer == "Adafactor":
            optimizer = Adafactor(
                optimizer_grouped_parameters,
                lr=args.learning_rate,
                eps=args.adafactor_eps,
                clip_threshold=args.adafactor_clip_threshold,
                decay_rate=args.adafactor_decay_rate,
                beta1=args.adafactor_beta1,
                weight_decay=args.weight_decay,
                scale_parameter=args.adafactor_scale_parameter,
                relative_step=args.adafactor_relative_step,
                warmup_init=args.adafactor_warmup_init,
            )
            print("Using Adafactor for T5")
        else:
            raise ValueError(
                "{} is not a valid optimizer class. Please use one of ('AdamW', 'Adafactor') instead.".format(
                    args.optimizer
                )
            )

        if args.scheduler == "constant_schedule":
            scheduler = get_constant_schedule(optimizer)

        elif args.scheduler == "constant_schedule_with_warmup":
            scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps)

        elif args.scheduler == "linear_schedule_with_warmup":
            scheduler = get_linear_schedule_with_warmup(
                optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
            )

        elif args.scheduler == "cosine_schedule_with_warmup":
            scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=args.warmup_steps,
                num_training_steps=t_total,
                num_cycles=args.cosine_schedule_num_cycles,
            )

        elif args.scheduler == "cosine_with_hard_restarts_schedule_with_warmup":
            scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(
                optimizer,
                num_warmup_steps=args.warmup_steps,
                num_training_steps=t_total,
                num_cycles=args.cosine_schedule_num_cycles,
            )

        elif args.scheduler == "polynomial_decay_schedule_with_warmup":
            scheduler = get_polynomial_decay_schedule_with_warmup(
                optimizer,
                num_warmup_steps=args.warmup_steps,
                num_training_steps=t_total,
                lr_end=args.polynomial_decay_schedule_lr_end,
                power=args.polynomial_decay_schedule_lr_end,
            )

        else:
            raise ValueError("{} is not a valid scheduler.".format(args.scheduler))

        if args.n_gpu > 1:
            model = torch.nn.DataParallel(model)

        global_step = 0
        training_progress_scores = None
        tr_loss, logging_loss = 0.0, 0.0
        model.zero_grad()
        train_iterator = trange(int(args.num_train_epochs), desc="Epoch", disable=args.silent, mininterval=0)
        epoch_number = 0
        best_eval_metric = None
        early_stopping_counter = 0
        steps_trained_in_current_epoch = 0
        epochs_trained = 0
        current_loss = "Initializing"

        if args.model_name and os.path.exists(args.model_name) and self.is_continue_training:
            try:
                # set global_step to gobal_step of last saved checkpoint from model path
                checkpoint_suffix = args.model_name.split("/")[-1].split("-")
                if len(checkpoint_suffix) > 2:
                    checkpoint_suffix = checkpoint_suffix[1]
                else:
                    checkpoint_suffix = checkpoint_suffix[-1]
                global_step = int(checkpoint_suffix)
                epochs_trained = global_step // (len(train_df) // args.gradient_accumulation_steps)
                steps_trained_in_current_epoch = global_step % (
                    len(train_df) // args.gradient_accumulation_steps
                )

                logger.info("   Continuing training from checkpoint, will skip to saved global_step")
                logger.info("   Continuing training from epoch %d", epochs_trained)
                logger.info("   Continuing training from global step %d", global_step)
                logger.info("   Will skip the first %d steps in the current epoch", steps_trained_in_current_epoch)
            except ValueError:
                logger.info("   Starting fine-tuning.")

        if args.evaluate_during_training:
            training_progress_scores = self._create_training_progress_scores(multi_label, **kwargs)
        if args.wandb_project:
            if not wandb.setup().settings.sweep_id:
                logger.info(" Initializing WandB run for training.")
                wandb.init(project=args.wandb_project, config={**asdict(args)}, **args.wandb_kwargs)
            wandb.watch(self.model)

        if self.args.fp16:
            from torch.cuda import amp

            scaler = amp.GradScaler()

        for _ in train_iterator:
            model.train()
            if epochs_trained > 0:
                epochs_trained -= 1
                continue
            train_iterator.set_description(f"Epoch {epoch_number + 1} of {args.num_train_epochs}")
            train_dataloader = self._get_train_dataloader(train_df, epoch_number)
            batch_iterator = tqdm(
                train_dataloader,
                desc=f"Running Epoch {epoch_number} of {args.num_train_epochs}",
                disable=args.silent,
                mininterval=0,
            )
            for step, batch in enumerate(batch_iterator):
                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    continue

                inputs = self._get_inputs_dict(batch)
                if self.args.fp16:
                    with amp.autocast():
                        outputs = model(**inputs, margin=self.args.margin,
                                        loss_alpha=self.args.loss_alpha)
                        # model outputs are always tuple in pytorch-transformers (see doc)
                        loss = outputs[0]
                else:
                    outputs = model(**inputs, margin=self.args.margin,
                                    loss_alpha=self.args.loss_alpha)
                    # model outputs are always tuple in pytorch-transformers (see doc)
                    loss = outputs[0]

                if args.n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu parallel training

                current_loss = loss.item()

                if show_running_loss:
                    batch_iterator.set_description(
                        f"Epochs {epoch_number}/{args.num_train_epochs}. Running Loss: {current_loss:9.4f}"
                    )

                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                if self.args.fp16:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                tr_loss += loss.item()
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    if self.args.fp16:
                        scaler.unscale_(optimizer)
                    if args.optimizer == "AdamW":
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                    if self.args.fp16:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    scheduler.step()  # Update learning rate schedule
                    model.zero_grad()
                    global_step += 1

                    if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                        # Log metrics
                        tb_writer.add_scalar("lr", scheduler.get_last_lr()[0], global_step)
                        tb_writer.add_scalar("loss", (tr_loss - logging_loss) / args.logging_steps, global_step)
                        logging_loss = tr_loss
                        if args.wandb_project or self.is_sweeping:
                            wandb.log(
                                {
                                    "Training loss": current_loss,
                                    "lr": scheduler.get_last_lr()[0],
                                    "global_step": global_step,
                                }
                            )

                    if args.save_steps > 0 and global_step % args.save_steps == 0:
                        # Save model checkpoint
                        output_dir_current = os.path.join(output_dir, "checkpoint-{}".format(global_step))

                        self.save_model(output_dir_current, optimizer, scheduler, model=model)

                    if args.evaluate_during_training and (
                        args.evaluate_during_training_steps > 0
                        and global_step % args.evaluate_during_training_steps == 0
                    ):
                        # Only evaluate when single GPU otherwise metrics may not average well
                        results = self.eval_model(
                            eval_df,
                            verbose=verbose and args.evaluate_during_training_verbose,
                            silent=args.evaluate_during_training_silent,
                            wandb_log=False,
                            **kwargs,
                        )
                        for key, value in results.items():
                            tb_writer.add_scalar("eval_{}".format(key), value, global_step)

                        output_dir_current = os.path.join(output_dir, "checkpoint-{}".format(global_step))

                        if args.save_eval_checkpoints:
                            self.save_model(output_dir_current, optimizer, scheduler, model=model, results=results)

                        training_progress_scores["global_step"].append(global_step)
                        training_progress_scores["train_loss"].append(current_loss)
                        for key in results:
                            training_progress_scores[key].append(results[key])
                        report = pd.DataFrame(training_progress_scores)
                        report.to_csv(
                            os.path.join(args.output_dir, "training_progress_scores.csv"), index=False,
                        )

                        if args.wandb_project or self.is_sweeping:
                            wandb.log(self._get_last_metrics(training_progress_scores))

                        if not best_eval_metric:
                            best_eval_metric = results[args.early_stopping_metric]
                            self.save_model(args.best_model_dir, optimizer, scheduler, model=model, results=results)
                        if best_eval_metric and args.early_stopping_metric_minimize:
                            if best_eval_metric - results[args.early_stopping_metric] > args.early_stopping_delta:
                                best_eval_metric = results[args.early_stopping_metric]
                                self.save_model(
                                    args.best_model_dir, optimizer, scheduler, model=model, results=results
                                )
                                early_stopping_counter = 0
                            else:
                                if args.use_early_stopping:
                                    if early_stopping_counter < args.early_stopping_patience:
                                        early_stopping_counter += 1
                                        if verbose:
                                            logger.info(f" No improvement in {args.early_stopping_metric}")
                                            logger.info(f" Current step: {early_stopping_counter}")
                                            logger.info(f" Early stopping patience: {args.early_stopping_patience}")
                                    else:
                                        if verbose:
                                            logger.info(f" Patience of {args.early_stopping_patience} steps reached")
                                            logger.info(" Training terminated.")
                                            train_iterator.close()
                                        return (
                                            global_step,
                                            tr_loss / global_step
                                            if not self.args.evaluate_during_training
                                            else training_progress_scores,
                                        )
                        else:
                            if results[args.early_stopping_metric] - best_eval_metric > args.early_stopping_delta:
                                best_eval_metric = results[args.early_stopping_metric]
                                self.save_model(
                                    args.best_model_dir, optimizer, scheduler, model=model, results=results
                                )
                                early_stopping_counter = 0
                            else:
                                if args.use_early_stopping:
                                    if early_stopping_counter < args.early_stopping_patience:
                                        early_stopping_counter += 1
                                        if verbose:
                                            logger.info(f" No improvement in {args.early_stopping_metric}")
                                            logger.info(f" Current step: {early_stopping_counter}")
                                            logger.info(f" Early stopping patience: {args.early_stopping_patience}")
                                    else:
                                        if verbose:
                                            logger.info(f" Patience of {args.early_stopping_patience} steps reached")
                                            logger.info(" Training terminated.")
                                            train_iterator.close()
                                        return (
                                            global_step,
                                            tr_loss / global_step
                                            if not self.args.evaluate_during_training
                                            else training_progress_scores,
                                        )
                        model.train()

            epoch_number += 1
            output_dir_current = os.path.join(output_dir, "checkpoint-{}-epoch-{}".format(global_step, epoch_number))

            if args.save_model_every_epoch or args.evaluate_during_training:
                os.makedirs(output_dir_current, exist_ok=True)

            if args.save_model_every_epoch:
                self.save_model(output_dir_current, optimizer, scheduler, model=model)

            if args.evaluate_during_training and args.evaluate_each_epoch:
                results = self.eval_model(
                    eval_df,
                    verbose=verbose and args.evaluate_during_training_verbose,
                    silent=args.evaluate_during_training_silent,
                    wandb_log=False,
                    **kwargs,
                )

                self.save_model(output_dir_current, optimizer, scheduler, results=results)

                training_progress_scores["global_step"].append(global_step)
                training_progress_scores["train_loss"].append(current_loss)
                for key in results:
                    training_progress_scores[key].append(results[key])
                report = pd.DataFrame(training_progress_scores)
                report.to_csv(os.path.join(args.output_dir, "training_progress_scores.csv"), index=False)

                if args.wandb_project or self.is_sweeping:
                    wandb.log(self._get_last_metrics(training_progress_scores))

                if not best_eval_metric:
                    best_eval_metric = results[args.early_stopping_metric]
                    self.save_model(args.best_model_dir, optimizer, scheduler, model=model, results=results)
                if best_eval_metric and args.early_stopping_metric_minimize:
                    if best_eval_metric - results[args.early_stopping_metric] > args.early_stopping_delta:
                        best_eval_metric = results[args.early_stopping_metric]
                        self.save_model(args.best_model_dir, optimizer, scheduler, model=model, results=results)
                        early_stopping_counter = 0
                    else:
                        if args.use_early_stopping and args.early_stopping_consider_epochs:
                            if early_stopping_counter < args.early_stopping_patience:
                                early_stopping_counter += 1
                                if verbose:
                                    logger.info(f" No improvement in {args.early_stopping_metric}")
                                    logger.info(f" Current step: {early_stopping_counter}")
                                    logger.info(f" Early stopping patience: {args.early_stopping_patience}")
                            else:
                                if verbose:
                                    logger.info(f" Patience of {args.early_stopping_patience} steps reached")
                                    logger.info(" Training terminated.")
                                    train_iterator.close()
                                return (
                                    global_step,
                                    tr_loss / global_step
                                    if not self.args.evaluate_during_training
                                    else training_progress_scores,
                                )
                else:
                    if results[args.early_stopping_metric] - best_eval_metric > args.early_stopping_delta:
                        best_eval_metric = results[args.early_stopping_metric]
                        self.save_model(args.best_model_dir, optimizer, scheduler, model=model, results=results)
                        early_stopping_counter = 0
                    else:
                        if args.use_early_stopping and args.early_stopping_consider_epochs:
                            if early_stopping_counter < args.early_stopping_patience:
                                early_stopping_counter += 1
                                if verbose:
                                    logger.info(f" No improvement in {args.early_stopping_metric}")
                                    logger.info(f" Current step: {early_stopping_counter}")
                                    logger.info(f" Early stopping patience: {args.early_stopping_patience}")
                            else:
                                if verbose:
                                    logger.info(f" Patience of {args.early_stopping_patience} steps reached")
                                    logger.info(" Training terminated.")
                                    train_iterator.close()
                                return (
                                    global_step,
                                    tr_loss / global_step
                                    if not self.args.evaluate_during_training
                                    else training_progress_scores,
                                )

        return (
            global_step,
            tr_loss / global_step if not self.args.evaluate_during_training else training_progress_scores,
        )

    def eval_model(
        self, eval_df, multi_label=False, output_dir=None, verbose=True, silent=False, wandb_log=True, **kwargs
    ):
        """
        Evaluates the model on eval_df. Saves results to output_dir.

        Args:
            eval_df: Pandas Dataframe containing at least two columns. If the Dataframe has a header, it should contain a 'text' and a 'labels' column. If no header is present,
            the Dataframe should contain at least two columns, with the first column containing the text, and the second column containing the label. The model will be evaluated on this Dataframe.
            output_dir: The directory where model files will be saved. If not given, self.args.output_dir will be used.
            verbose: If verbose, results will be printed to the console on completion of evaluation.
            silent: If silent, tqdm progress bars will be hidden.
            wandb_log: If True, evaluation results will be logged to wandb.
            **kwargs: Additional metrics that should be used. Pass in the metrics as keyword arguments (name of metric: function to use). E.g. f1=sklearn.metrics.f1_score.
                        A metric function should take in two parameters. The first parameter will be the true labels, and the second parameter will be the predictions.

        Returns:
            result: Dictionary containing evaluation results.
            model_outputs: List of model outputs for each row in eval_df
            wrong_preds: List of InputExample objects corresponding to each incorrect prediction by the model
        """  # noqa: ignore flake8"

        if not output_dir:
            output_dir = self.args.output_dir

        self._move_model_to_device()
        logger.info('evaluate...')
        result = self.evaluate(
            eval_df, output_dir, multi_label=multi_label, verbose=verbose, silent=silent, wandb_log=wandb_log, **kwargs
        )
        self.results.update(result)

        if verbose:
            logger.info(self.results)

        return result

    def evaluate(
        self, eval_df, output_dir, multi_label=False, prefix="", verbose=True, silent=False, wandb_log=True, **kwargs
    ):
        """
        Evaluates the model on eval_df.

        Utility function to be used by the eval_model() method. Not intended to be used directly.
        """

        model = self.model
        args = self.args
        eval_output_dir = output_dir

        results = {}
        if isinstance(eval_df, str) and self.args.lazy_loading:
            if self.args.model_type == "layoutlm":
                raise NotImplementedError("Lazy loading is not implemented for LayoutLM models")
            eval_dataset = LazyClassificationDataset(eval_df, self.tokenizer, self.args)
            eval_examples = None
        else:
            if self.args.lazy_loading:
                raise ValueError("Input must be given as a path to a file when using lazy loading")

            eval_examples = [
                InputExample(i, text, None, label)
                for i, (text, label) in enumerate(
                    zip(eval_df["text"].astype(str), eval_df["labels"])
                )
            ]

            eval_dataset = self.load_and_cache_examples(
                eval_examples, evaluate=True, verbose=verbose, silent=silent
            )
        os.makedirs(eval_output_dir, exist_ok=True)

        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

        if args.n_gpu > 1:
            model = torch.nn.DataParallel(model)

        n_batches = len(eval_dataloader)
        preds = np.empty((len(eval_dataset)))
        out_labels = np.empty((len(eval_dataset)))
        model.eval()

        if self.args.fp16:
            from torch.cuda import amp

        for i, batch in enumerate(tqdm(eval_dataloader, disable=args.silent or silent, desc="Running Evaluation")):
            with torch.no_grad():
                inputs = self._get_inputs_dict(batch)

                if self.args.fp16:
                    with amp.autocast():
                        outputs = model(**inputs, single_text=True)
                        logits = outputs[0]
                else:
                    outputs = model(**inputs, single_text=True)
                    logits = outputs[0]
                logits = torch.softmax(logits.view(-1, 2), 1)

            start_index = self.args.eval_batch_size * i
            end_index = start_index + self.args.eval_batch_size if i != (n_batches - 1) else len(eval_dataset)
            preds[start_index:end_index] = logits.detach().cpu().numpy()[:, 1]
            out_labels[start_index:end_index] = inputs["labels"].detach().cpu().numpy()

        model_outputs = preds

        result = self.compute_metrics(preds, out_labels)
        result['eval_loss'] = 1.0 - result['discrete_auc']
        results.update(result)

        output_eval_file = os.path.join(eval_output_dir, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            for key in sorted(result.keys()):
                writer.write("{} = {}\n".format(key, str(result[key])))

        return results

    def load_and_cache_examples(
        self, examples, evaluate=False, no_cache=False, multi_label=False, verbose=True, silent=False
    ):
        """
        Converts a list of InputExample objects to a TensorDataset containing InputFeatures. Caches the InputFeatures.

        Utility function for train() and eval() methods. Not intended to be used directly.
        """

        process_count = self.args.process_count

        tokenizer = self.tokenizer
        args = self.args

        os.makedirs(self.args.cache_dir, exist_ok=True)

        features = convert_examples_to_features(
            examples,
            args.max_seq_length,
            tokenizer,
            "classification",
            # XLNet has a CLS token at the end
            cls_token_at_end=bool(args.model_type in ["xlnet"]),
            cls_token=tokenizer.cls_token,
            cls_token_segment_id=2 if args.model_type in ["xlnet"] else 0,
            sep_token=tokenizer.sep_token,
            # RoBERTa uses an extra separator b/w pairs of sentences,
            # cf. github.com/pytorch/fairseq/commit/1684e166e3da03f5b600dbb7855cb98ddfcd0805
            sep_token_extra=bool(args.model_type in ["roberta", "camembert", "xlmroberta", "longformer"]),
            # PAD on the left for XLNet
            pad_on_left=bool(args.model_type in ["xlnet"]),
            pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
            pad_token_segment_id=4 if args.model_type in ["xlnet"] else 0,
            process_count=process_count,
            multi_label=multi_label,
            silent=args.silent or silent,
            use_multiprocessing=args.use_multiprocessing,
            sliding_window=args.sliding_window,
            flatten=not evaluate,
            stride=args.stride,
            add_prefix_space=bool(args.model_type in ["roberta", "camembert", "xlmroberta", "longformer"]),
            # avoid padding in case of single example/online inferencing to decrease execution time
            pad_to_max_length=bool(len(examples) > 1),
            args=args,
        )

        all_input_ids_a = torch.tensor([f['feature_a'].input_ids for f in features], dtype=torch.long)
        all_input_mask_a = torch.tensor([f['feature_a'].input_mask for f in features], dtype=torch.long)
        all_segment_ids_a = torch.tensor([f['feature_a'].segment_ids for f in features], dtype=torch.long)
        all_input_ids_b = torch.tensor([f['feature_b'].input_ids for f in features], dtype=torch.long)
        all_input_mask_b = torch.tensor([f['feature_b'].input_mask for f in features], dtype=torch.long)
        all_segment_ids_b = torch.tensor([f['feature_b'].segment_ids for f in features], dtype=torch.long)

        all_label_ids = torch.tensor([f['feature_a'].label_id for f in features], dtype=torch.long)

        dataset = TensorDataset(all_input_ids_a, all_input_mask_a, all_segment_ids_a, all_label_ids,
                                all_input_ids_b, all_input_mask_b, all_segment_ids_b)

        return dataset

    def compute_metrics(self, preds, labels):
        """
        Computes the evaluation metrics for the model predictions.

        Args:
            preds: Model predictions
            model_outputs: Model outputs
            labels: Ground truth labels
            eval_examples: List of examples on which evaluation was performed
            **kwargs: Additional metrics that should be used. Pass in the metrics as keyword arguments (name of metric: function to use). E.g. f1=sklearn.metrics.f1_score.
                        A metric function should take in two parameters. The first parameter will be the true labels, and the second parameter will be the predictions.

        Returns:
            result: Dictionary containing evaluation results.
            For non-binary classification, the dictionary format is: (Matthews correlation coefficient, tp, tn, fp, fn).
            For binary classification, the dictionary format is: (Matthews correlation coefficient, tp, tn, fp, fn, AUROC, AUPRC).
            wrong: List of InputExample objects corresponding to each incorrect prediction by the model
        """  # noqa: ignore flake8"

        assert len(preds) == len(labels)
        return {
            'discrete_auc': discrete_auc(labels, preds)
        }

    def predict(self, to_predict, multi_label=False):
        """
        Performs predictions on a list of text.

        Args:
            to_predict: A python list of text (str) to be sent to the model for prediction.

        Returns:
            preds: A python list of the predictions (0 or 1) for each text.
            model_outputs: A python list of the raw model outputs for each text.
        """

        model = self.model
        args = self.args

        preds = np.empty((len(to_predict)))

        self._move_model_to_device()

        if args.n_gpu > 1:
            model = torch.nn.DataParallel(model)
        dummy_label = 0
        eval_examples = [InputExample(i, text, None, dummy_label) for i, text in enumerate(to_predict)]

        eval_dataset = self.load_and_cache_examples(
            eval_examples, evaluate=True, multi_label=multi_label, no_cache=True
        )

        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

        if self.args.fp16:
            from torch.cuda import amp

        n_batches = len(eval_dataloader)
        for i, batch in enumerate(tqdm(eval_dataloader, disable=args.silent)):
            model.eval()
            # batch = tuple(t.to(device) for t in batch)

            with torch.no_grad():
                inputs = self._get_inputs_dict(batch)

                if self.args.fp16:
                    with amp.autocast():
                        outputs = model(**inputs, single_text=True)
                        logits = outputs[0]
                else:
                    outputs = model(**inputs, single_text=True)
                    logits = outputs[0]

                logits = torch.softmax(logits.view(-1, 2), 1)

            start_index = self.args.eval_batch_size * i
            end_index = start_index + self.args.eval_batch_size if i != (n_batches - 1) else len(eval_dataset)
            preds[start_index:end_index] = logits.detach().cpu().numpy()[:, 1]

        return preds

    def convert_to_onnx(self, output_dir=None, set_onnx_arg=True):
        """Convert the model to ONNX format and save to output_dir

        Args:
            output_dir (str, optional): If specified, ONNX model will be saved to output_dir (else args.output_dir will be used). Defaults to None.
            set_onnx_arg (bool, optional): Updates the model args to set onnx=True. Defaults to True.
        """  # noqa
        if not output_dir:
            output_dir = os.path.join(self.args.output_dir, "onnx")
        os.makedirs(output_dir, exist_ok=True)

        if os.listdir(output_dir):
            raise ValueError(
                "Output directory ({}) already exists and is not empty."
                " Output directory for onnx conversion must be empty.".format(output_dir)
            )

        onnx_model_name = os.path.join(output_dir, "onnx_model.onnx")

        with tempfile.TemporaryDirectory() as temp_dir:
            self.save_model(output_dir=temp_dir, model=self.model)

            convert(
                framework="pt",
                model=temp_dir,
                tokenizer=self.tokenizer,
                output=Path(onnx_model_name),
                pipeline_name="sentiment-analysis",
                opset=11,
            )

        self.args.onnx = True
        self.tokenizer.save_pretrained(output_dir)
        self.config.save_pretrained(output_dir)
        self.save_model_args(output_dir)

    def _threshold(self, x, threshold):
        if x >= threshold:
            return 1
        return 0

    def _move_model_to_device(self):
        self.model.to(self.device)

    def _get_inputs_dict(self, batch):
        if isinstance(batch[0], dict):
            inputs = {key: value.squeeze().to(self.device) for key, value in batch[0].items()}
            inputs["labels"] = batch[1].to(self.device)
        else:
            batch = tuple(t.to(self.device) for t in batch)

            inputs = {"input_ids_a": batch[0],
                      "attention_mask_a": batch[1],
                      "labels": batch[3],
                      "input_ids_b": batch[4],
                      "attention_mask_b": batch[5],
                      }

            # XLM, DistilBERT and RoBERTa don't use segment_ids
            if self.args.model_type != "distilbert":
                inputs["token_type_ids_a"] = (
                    batch[2] if self.args.model_type in ["bert", "xlnet", "albert", "layoutlm"] else None
                )
                inputs["token_type_ids_b"] = (
                    batch[6] if self.args.model_type in ["bert", "xlnet", "albert", "layoutlm"] else None
                )
        return inputs

    def _get_last_metrics(self, metric_values):
        return {metric: values[-1] for metric, values in metric_values.items()}

    def _create_training_progress_scores(self, multi_label, **kwargs):
        extra_metrics = {key: [] for key in kwargs}

        training_progress_scores = {
            "global_step": [],
            "train_loss": [],
            "eval_loss": [],
            "discrete_auc": [],
            **extra_metrics,
        }
        return training_progress_scores

    def save_model(self, output_dir=None, optimizer=None, scheduler=None, model=None, results=None):
        if not output_dir:
            output_dir = self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)

        if model and not self.args.no_save:
            # Take care of distributed/parallel training
            model_to_save = model.module if hasattr(model, "module") else model
            model_to_save.save_pretrained(output_dir)
            self.tokenizer.save_pretrained(output_dir)
            torch.save(self.args, os.path.join(output_dir, "training_args.bin"))
            if optimizer and scheduler and self.args.save_optimizer_and_scheduler:
                torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
            self.save_model_args(output_dir)

        if results:
            output_eval_file = os.path.join(output_dir, "eval_results.txt")
            with open(output_eval_file, "w") as writer:
                for key in sorted(results.keys()):
                    writer.write("{} = {}\n".format(key, str(results[key])))

    def save_model_args(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        self.args.save(output_dir)

    def _load_model_args(self, input_dir):
        args = TextPairArgs()
        args.load(input_dir)
        return args

    def get_named_parameters(self):
        return [n for n, p in self.model.named_parameters()]

    def _get_train_dataloader(self, train_df, epoch_id):
        pair_train_df = self.create_pair_df(train_df, epoch_id)
        train_examples = [
            InputExample(i, text_a, text_b, label)
            for i, (text_a, text_b, label) in enumerate(
                zip(pair_train_df["text_a"].astype(str), pair_train_df["text_b"].astype(str), pair_train_df["labels"])
            )
        ]
        train_dataset = self.load_and_cache_examples(train_examples, verbose=True)
        train_sampler = RandomSampler(train_dataset)
        train_dataloader = DataLoader(
            train_dataset,
            sampler=train_sampler,
            batch_size=self.args.train_batch_size,
            num_workers=self.args.dataloader_num_workers,
        )
        return train_dataloader

    def create_pair_df(self, train_df, epoch_id):
        os.makedirs(self.args.cache_dir, exist_ok=True)
        logger.info(f'Epoch {epoch_id + 1} construct training data...')
        pos_df = train_df[train_df['labels'] == 1]
        neg_df = train_df[train_df['labels'] == 0]
        if epoch_id < self.args.stage1_epoch_num:
            if len(pos_df) > len(neg_df):
                neg_df = pd.concat([neg_df, neg_df.iloc[0:len(pos_df)-len(neg_df), :]])
            else:
                pos_df = pd.concat([pos_df, pos_df.iloc[0:len(neg_df)-len(pos_df), :]])
            pair_df = pd.DataFrame({'text_a': pos_df['text'], 'text_b': neg_df['text'], 'labels': [1] * len(pos_df)})
        else:
            train_text_list = train_df['text'].to_list()
            train_label_list = train_df['labels'].to_list()
            train_pred_list = self.predict(train_text_list)
            self.model.train()
            train_records = []
            for i in range(len(train_text_list)):
                if train_label_list[i] == 0:
                    predict_score = train_pred_list[i] + self.args.hard_sample_mining_margin
                else:
                    predict_score = train_pred_list[i]
                train_records.append({
                    'text': train_text_list[i],
                    'labels': train_label_list[i],
                    'predict': predict_score
                })
            train_records.sort(key=lambda record: -record['predict'])
            hard_pair_records = []
            neg_train_record_ids = []
            for idx in tqdm(range(len(train_records))):
                if train_records[idx]['labels'] == 0:
                    neg_train_record_ids.append(idx)
                elif train_records[idx]['labels'] == 1:
                    if self.args.max_neg_nums > 0:
                        cur_neg_record_ids = random.sample(neg_train_record_ids, k=min(self.args.max_neg_nums, len(neg_train_record_ids)))
                    else:
                        cur_neg_record_ids = neg_train_record_ids
                    for neg_record_id in cur_neg_record_ids:
                        neg_train_record_ids.remove(neg_record_id)
                        hard_pair_records.append({
                            'text_a': train_records[idx]['text'],
                            'text_b': train_records[neg_record_id]['text'],
                            'labels': 1
                        })
                else:
                    raise ValueError('labels value error.')
            logging.info(f'hard mining training data number: {len(hard_pair_records)}')
            hard_pair_df = pd.DataFrame(hard_pair_records)
            hard_pair_df.to_json(os.path.join(self.args.cache_dir, f'hard_pairs_{epoch_id + 1}.jsonl'),
                            lines=True, orient='records', force_ascii=False)
            # 随机正负样本对
            rand_pair_records = []
            pos_number, neg_number = len(pos_df), len(neg_df)
            for neg_i in range(neg_number):
                pos_i = random.randint(0, pos_number - 1)
                rand_pair_records.append({
                    'text_a': pos_df.iloc[pos_i]['text'],
                    'text_b': neg_df.iloc[neg_i]['text'],
                    'labels': 1
                })
            pair_df = pd.DataFrame(hard_pair_records + rand_pair_records)
            # pair_df = pd.DataFrame(rand_pair_records)
        return pair_df
