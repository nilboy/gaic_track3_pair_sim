import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss, MSELoss
from simpletransformers_addons.transformers.bert.modeling_bert import BertModel, BertPreTrainedModel
from simpletransformers_addons.layers import LabelSmoothCrossEntropyLoss

from simpletransformers_addons.transformers.bert.configuration_bert import BertConfig
from simpletransformers_addons.transformers.bert.modeling_nezha import NeZhaConfig, NeZhaModel
from transformers import DebertaModel
from simpletransformers_addons.models.sim_text.transformer_models.ensemble_model import EnsembleModel

submodel_map = {
    "bert": BertModel,
    "nezha": NeZhaModel,
    "deberta": DebertaModel,
    "ensemble": EnsembleModel
}

def mean_pooling(sequence_outputs, attention_mask):
    token_embeddings = sequence_outputs #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask

def masked_max_pooling(data_tensor, mask, dim):
    """
    Performs masked max-pooling across the specified dimension of a Tensor.
    :param data_tensor: ND Tensor.
    :param mask: Tensor containing a binary mask that can be broad-casted to the shape of data_tensor.
    :param dim: Int that corresponds to the dimension.
    :return: (N-1)D Tensor containing the result of the max-pooling operation.
    """

    if dim < 0:
        dim = len(data_tensor.shape) + dim

    mask = mask.view(list(mask.shape) + [1] * (len(data_tensor.shape) - len(mask.shape)))
    data_tensor = data_tensor.masked_fill(mask == 0, -1e9)

    max_vals, max_ids = torch.max(data_tensor, dim=dim)

    return max_vals


def masked_min_pooling(data_tensor, mask, dim):
    """
    Performs masked min-pooling across the specified dimension of a Tensor.
    :param data_tensor: ND Tensor.
    :param mask: Tensor containing a binary mask that can be broad-casted to the shape of data_tensor.
    :param dim: Int that corresponds to the dimension.
    :return: (N-1)D Tensor containing the result of the min-pooling operation.
    """

    if dim < 0:
        dim = len(data_tensor.shape) + dim

    mask = mask.view(list(mask.shape) + [1] * (len(data_tensor.shape) - len(mask.shape)))
    data_tensor = data_tensor.masked_fill(mask == 0, 1e9)

    min_vals, min_ids = torch.min(data_tensor, dim=dim)

    return min_vals


def masked_mean_pooling(data_tensor, mask, dim):
    """
    Performs masked mean-pooling across the specified dimension of a Tensor.
    :param data_tensor: ND Tensor.
    :param mask: Tensor containing a binary mask that can be broad-casted to the shape of data_tensor.
    :param dim: Int that corresponds to the dimension.
    :return: (N-1)D Tensor containing the result of the mean-pooling operation.
    """

    if dim < 0:
        dim = len(data_tensor.shape) + dim

    mask = mask.view(list(mask.shape) + [1] * (len(data_tensor.shape) - len(mask.shape)))
    data_tensor = data_tensor.masked_fill(mask == 0, 0)

    nominator = torch.sum(data_tensor, dim=dim)
    denominator = torch.sum(mask.type(nominator.type()), dim=dim)

    return nominator / denominator

def pooling_fn(outputs,
               attention_mask):
    cls_outputs = outputs[1]
    sequence_outputs = outputs[0]
    # mean_outputs = masked_mean_pooling(sequence_outputs, attention_mask, 1)
    #return torch.cat([cls_outputs, mean_outputs], dim=-1)
    return cls_outputs
    #return cls_outputs

class BertForSimText(BertPreTrainedModel):
    r"""
        **labels**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size,)``:
            Labels for computing the sequence classification/regression loss.
            Indices should be in ``[0, ..., config.num_labels - 1]``.
            If ``config.num_labels == 1`` a regression loss is computed (Mean-Square loss),
            If ``config.num_labels > 1`` a classification loss is computed (Cross-Entropy).
    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **loss**: (`optional`, returned when ``labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Classification (or regression if config.num_labels==1) loss.
        **logits**: ``torch.FloatTensor`` of shape ``(batch_size, config.num_labels)``
            Classification (or regression if config.num_labels==1) scores (before SoftMax).
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.
    Examples::
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute")).unsqueeze(0)  # Batch size 1
        labels = torch.tensor([1]).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids, labels=labels)
        loss, logits = outputs[:2]
    """  # noqa: ignore flake8"

    def __init__(self, config, weight=None, label_smooth=0.0, temperature=0.0,
                 wsl_alpha=0.0, use_bimodel=True, submodel_type="bert"):
        super(BertForSimText, self).__init__(config)
        self.num_labels = config.num_labels
        self.bert = submodel_map[submodel_type](config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.temperature = temperature
        self.normal_mode = self.temperature <= 0.0
        if self.normal_mode:
            self.classifier = nn.Linear(config.hidden_size * 1, self.config.num_labels)
        else:
            self.classifier = nn.Linear(config.hidden_size * 1, 2)
        self.weight = weight
        self.label_smooth = label_smooth
        self.wsl_alpha = wsl_alpha
        self.use_wsl = self.wsl_alpha > 0.0
        self.use_bimodel = use_bimodel
        self.init_weights()

    def forward(
        self,
        input_ids_a=None,
        attention_mask_a=None,
        token_type_ids_a=None,
        position_ids_a=None,
        input_ids_b=None,
        attention_mask_b=None,
        token_type_ids_b=None,
        position_ids_b=None,
        head_mask_a=None,
        inputs_embeds_a=None,
        head_mask_b=None,
        inputs_embeds_b=None,
        labels=None,
        single_text=False,
        margin=0.1,
        loss_alpha=0.0
    ):
        outputs_a = self.bert(
            input_ids_a,
            attention_mask=attention_mask_a,
            token_type_ids=token_type_ids_a,
            position_ids=position_ids_a,
            head_mask=head_mask_a,
        )
        # Complains if input_embeds is kept
        pooled_output_a = pooling_fn(outputs_a, attention_mask_a)
        if self.use_bimodel:
            outputs_b = self.bert(
                input_ids_b,
                attention_mask=attention_mask_b,
                token_type_ids=token_type_ids_b,
                position_ids=position_ids_b,
                head_mask=head_mask_b,
            )
            # Complains if input_embeds is kept
            pooled_output_b = pooling_fn(outputs_b, attention_mask_b)
            pooled_output = (pooled_output_b + pooled_output_a) / 2.0
        else:
            pooled_output = pooled_output_a
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        if self.normal_mode:
            outputs = (logits,) + outputs_a[2:]  # add hidden states and attention if they are here
        else:
            logits /= self.temperature
            outputs = (torch.softmax(logits, dim=1)[..., 1:],) + outputs_a[2:]

        if labels is not None:
            if self.num_labels == 1:
                if self.normal_mode:
                    # We are doing regression
                    loss_fct = MSELoss()
                    loss = loss_fct(logits.view(-1), labels.view(-1))
                else:
                    from torch.nn import functional as F
                    if not self.use_wsl:
                        targets = torch.stack([1.0 - labels.view(-1), labels.view(-1)], dim=1)
                        lsm = F.log_softmax(logits, -1)
                        loss = -(targets * lsm).sum(-1).mean() * (self.temperature ** 2)
                    else:
                        lsm = F.log_softmax(logits, -1)
                        hard_labels = torch.nn.functional.one_hot((labels > 0).type(torch.long)).type(logits.dtype)
                        soft_labels = torch.stack([1.0 - torch.abs(labels).view(-1), torch.abs(labels).view(-1)], dim=1)
                        student_ce = -(hard_labels * lsm).sum(-1)
                        teacher_ce = -(hard_labels * torch.log(soft_labels)).sum(-1)
                        loss_kd = -(soft_labels * lsm).sum(-1) * (self.temperature ** 2)
                        loss_wsl = (1.0 - torch.exp(-student_ce.detach()/teacher_ce.detach())) * loss_kd
                        loss = (student_ce + self.wsl_alpha * loss_wsl).mean()
            else:
                if self.weight is not None:
                    weight = self.weight.to(labels.device)
                else:
                    weight = None
                loss_fct = LabelSmoothCrossEntropyLoss(weight=weight, smoothing=self.label_smooth)
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)
