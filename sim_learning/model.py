import warnings

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss, MSELoss, BCELoss
from dataclasses import dataclass

from typing import Optional, Tuple

from transformers.activations import ACT2FN, gelu
from transformers.file_utils import ModelOutput
from transformers.models.roberta.modeling_roberta import RobertaModel, RobertaPreTrainedModel


# Copied from transformers.modeling_roberta.RobertaLMHead
class RobertaLMHead(nn.Module):
    """Roberta Head for masked language modeling."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))

        # Need a link between the two variables so that the bias is correctly resized with `resize_token_embeddings`
        self.decoder.bias = self.bias

    def forward(self, features, **kwargs):
        x = self.dense(features)
        x = gelu(x)
        x = self.layer_norm(x)

        # project back to size of vocabulary with bias
        x = self.decoder(x)

        return x
    
    
# Copied from transformers.modeling_roberta.RobertaClassificationHead
class RobertaClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

class SimbertForPreTraining(RobertaPreTrainedModel):
    authorized_missing_keys = [r"position_ids"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
            
        self.roberta = RobertaModel(config)
        if self.config.mlm_layer6:
            self.mlm_head6 = RobertaLMHead(config)
        self.classifier = RobertaClassificationHead(config)
        
        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        return_dict=None,
        antonym_ids=None,
        antonym_label=None,
        synonym_ids=None,
        synonym_label=None,
        **kwargs,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape ``(batch_size, sequence_length)``, `optional`):
            Labels for computing the masked language modeling loss.
            Indices should be in ``[-100, 0, ..., config.vocab_size]`` (see ``input_ids`` docstring)
            Tokens with indices set to ``-100`` are ignored (masked), the loss is only computed for the tokens with labels
            in ``[0, ..., config.vocab_size]``
        replace_label (``torch.LongTensor`` of shape ``(batch_size,sequence_length)``, `optional`):
            Labels for computing the token replace type prediction (classification) loss.
            Indices should be in ``[0, 1, 2, 3, 4, 5, 6]``:
            - 0 indicates the token is the original token,
            - 1 indicates the token is replaced with the lemminflect token,
            - 2 indicates the token is replaced with the synonym,
            - 3 indicates the token is replaced with the hypernyms,
            - 4 indicates the token is replaced with the adjacency,
            - 5 indicates the token is replaced with the antonym,
            - 6 indicates the token is replaced with the random word.
        kwargs (:obj:`Dict[str, any]`, optional, defaults to `{}`):
            Used to hide legacy arguments that have been deprecated.
        Returns:
        """
        if "masked_lm_labels" in kwargs:
            warnings.warn(
                "The `masked_lm_labels` argument is deprecated and will be removed in a future version, use `labels` instead.",
                FutureWarning,
            )
            labels = kwargs.pop("masked_lm_labels")
        assert kwargs == {}, f"Unexpected keyword arguments: {list(kwargs.keys())}."
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        seq_len = input_ids.size()[1] // 2
        class_labels, mlm_labels = labels[:, 0], labels[:, 1:]
        a, b = input_ids[:, :seq_len], input_ids[:, seq_len:]
        input_ids = torch.cat((a, b), dim=0)
        a, b = attention_mask[:, :seq_len], attention_mask[:, seq_len:]
        attention_mask = torch.cat((a, b), dim=0)

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            return_dict=return_dict,
            output_hidden_states=True
        )
        batch_size = input_ids.size(0) // 2
        sequence_output, pooled_output = outputs[:2]
        class_scores = self.classifier(sequence_output[:batch_size])
        hidden_states = outputs[2]

        if self.config.mlm_layer6:
            mlm_scores6 = self.mlm_head6(hidden_states[6][batch_size:])

        total_loss = None
        if labels is not None:
            loss_tok = CrossEntropyLoss()
            loss_class = CrossEntropyLoss()
            class_loss = loss_class(class_scores, class_labels)

            if self.config.mlm_layer6:
                mlm_loss = loss_tok(mlm_scores6.view(-1, self.config.vocab_size), mlm_labels.reshape(-1))
                total_loss = mlm_loss + class_loss
            else:
                total_loss = class_loss

            #print(mlm_loss.item(), tec_loss.item(), sec_loss.item())
        if not return_dict:
            output = (mlm_scores,) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output
    
        return SimbertOutput(
            loss=total_loss,
            prediction_logits=class_scores,
        )


@dataclass
class SimbertOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    prediction_logits: torch.FloatTensor = None

        
from transformers import RobertaConfig

class SimConfig(RobertaConfig):
    mlm_layer6 = True
