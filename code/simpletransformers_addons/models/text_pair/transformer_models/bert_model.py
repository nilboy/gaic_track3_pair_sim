import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss, MSELoss
from transformers.models.bert.modeling_bert import BertModel, BertPreTrainedModel


class BertForTextPair(BertPreTrainedModel):
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

    def __init__(self, config, weight=None):
        super(BertForTextPair, self).__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, 2)
        self.weight = weight

        self.init_weights()

    def forward(
        self,
        input_ids_a=None,
        attention_mask_a=None,
        token_type_ids_a=None,
        position_ids_a=None,
        head_mask_a=None,
        inputs_embeds_a=None,
        input_ids_b=None,
        attention_mask_b=None,
        token_type_ids_b=None,
        position_ids_b=None,
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
        pooled_output_a = outputs_a[1]

        pooled_output_a = self.dropout(pooled_output_a)
        logits_a = self.classifier(pooled_output_a)
        if single_text:
            outputs = (logits_a,) + outputs_a[2:]
            return outputs

        outputs_b = self.bert(
            input_ids_b,
            attention_mask=attention_mask_b,
            token_type_ids=token_type_ids_b,
            position_ids=position_ids_b,
            head_mask=head_mask_b,
        )
        # Complains if input_embeds is kept
        pooled_output_b = outputs_b[1]

        pooled_output_b = self.dropout(pooled_output_b)
        logits_b = self.classifier(pooled_output_b)


        outputs = (logits_b,) + outputs_b[2:]  # add hidden states and attention if they are here

        if labels is not None:
            if self.weight is not None:
                weight = self.weight.to(labels.device)
            else:
                weight = None
            loss_fct = CrossEntropyLoss(weight=weight)
            loss_a = loss_fct(logits_a.view(-1, 2), torch.ones_like(labels))
            loss_b = loss_fct(logits_b.view(-1, 2), torch.zeros_like(labels))
            compare_value = torch.softmax(logits_a.view(-1, 2), 1)[:, 1] - torch.softmax(logits_b.view(-1, 2), 1)[:, 1]
            loss_compare =  torch.mean(torch.max(margin - compare_value, torch.zeros_like(compare_value)))
            loss = (loss_a + loss_b) / 2.0 * loss_alpha + (1.0 - loss_alpha) * loss_compare
            outputs = (loss,) + outputs_a[2:]

        return outputs  # (loss), logits, (hidden_states), (attentions)
