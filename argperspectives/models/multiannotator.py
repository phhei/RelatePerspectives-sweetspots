import operator
from functools import reduce

import torch
import torch.nn as nn

from typing import (
    Optional, 
    Tuple,    
    Union
)

import random
import shutil
from pathlib import Path
from torch.nn import CrossEntropyLoss
from functorch import combine_state_for_ensemble

from tqdm import tqdm

from transformers.modeling_outputs import SequenceClassifierOutput

from transformers.utils import logging

from transformers import AutoModel
class PerAnnotatorModelForSequenceClassification(torch.nn.Module):
    def __init__(
            self, 
            checkpoint, 
            annotators_mapping,
            label_weights,
            num_labels: Union[int, Tuple[int]] = 2,
            use_return_dict=True
    ):
        super().__init__()
        self.num_labels = num_labels
        self.use_return_dict = use_return_dict

        self.pretrained_model = AutoModel.from_pretrained(checkpoint)
        self.dim = self.pretrained_model.config.hidden_size

        self.annotators_mapping = annotators_mapping

        self.heads = nn.ModuleList([
            nn.Sequential(
                # dropout
                nn.Dropout(0.1),
                # dense
                self._init_linear(nn.Linear(
                    in_features=self.dim,
                    out_features=self.dim
                )),
                # non-linear activation
                torch.nn.Tanh(),
                # dropout
                nn.Dropout(0.1),
                # classfication projection
                self._init_linear(nn.Linear(
                    in_features=self.dim,
                    out_features=self.num_labels if isinstance(self.num_labels, int) else reduce(operator.mul, self.num_labels, 1)
                ))
            ) for _ in tqdm(annotators_mapping.values())
        ])

        self.label_weights = label_weights
        if torch.cuda.is_available() and label_weights:
            self.label_weights = \
                self.label_weights.cuda() if isinstance(self.label_weights, torch.Tensor) else \
                    {k: v.cuda() for k, v in self.label_weights.items()}
            
    # adapted from transformers.models.roberta.RobertaPreTrainedModel._init_weights
    def _init_linear(self, layer):
            layer.weight.data.normal_(mean=0.0, std=1.0)
            if layer.bias is not None:
                layer.bias.data.zero_()
            return layer

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        annotator_indecies: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[SequenceClassifierOutput, Tuple[torch.Tensor, ...]]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.use_return_dict

        batch_losses = []
        batch_logits = []
        all_annotator_indecies = set(self.annotators_mapping.values())
        # FIXME How to replace looping with tensor operations?
        for batch_i in range(labels.shape[0]):
            pretrained_output = self.pretrained_model(
                input_ids=input_ids[batch_i:batch_i+1,:],
                attention_mask=attention_mask[batch_i:batch_i+1,:],
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            hidden_state = pretrained_output[0]  # (bs, seq_len, dim)
            pooled_output = hidden_state[:, 0]  # (bs, dim)

            example_losses = []
            example_logits = torch.full(
                size=(1, len(all_annotator_indecies), *([self.num_labels] if isinstance(self.num_labels, int) else self.num_labels)),
                fill_value=float('nan'),
                device=next(self.parameters()).device
            )

            if not self.training:
                # usually will be ordered correctly based on insertion order from index calculation
                # but can not be sure if unknown calling code, hence explicit ordering
                all_annotator_indecies = list(set(self.annotators_mapping.values()))
                all_annotator_indecies = sorted(all_annotator_indecies)

                # NOTE: eval functions pick annotators given their index to compare with their labels
                # - but should we here have the option to respect annotator indecies from examples?
                for annotator_index in all_annotator_indecies:
                    logits = self.heads[annotator_index](pooled_output)  # (bs, num_labels)
                    example_logits[0, annotator_index] = logits if isinstance(self.num_labels, int) \
                                         else logits.reshape(self.num_labels)
            else:
                not_missing_mask = annotator_indecies[batch_i] > -1
                annotator_indecies_on_example = annotator_indecies[batch_i][not_missing_mask].int().tolist()
                for i, annotator_index in enumerate(annotator_indecies_on_example):
                    logits = self.heads[annotator_index](pooled_output)  # (bs, num_labels)
                    example_logits[0, annotator_index] = logits if isinstance(self.num_labels, int) \
                                                                else logits.reshape(self.num_labels)

                    loss = None
                    if labels is not None:
                        problem_type = "single_label_classification"
                        if problem_type == "single_label_classification":

                            label = labels[batch_i, i]
                            label = label.to(next(self.parameters()).device, dtype=torch.long)
                            if label.dim() == 0:
                                #self.num_labels if isinstance(self.num_labels, int) else self.num_labels[-1]
                                loss = torch.nn.functional.cross_entropy(
                                    input=logits.view(-1, self.num_labels),
                                    target=label.view(-1),
                                    weight=self.label_weights[annotator_index], 
                                    reduction='none',
                                    ignore_index=-100
                                )
                            else:
                                loss = torch.mean(
                                        torch.stack(
                                            tensors=[
                                                torch.nn.functional.cross_entropy(
                                                input=logits.view(-1, self.num_labels[-1])[label_pos],
                                                target=label.view(-1)[label_pos],
                                                weight=self.label_weights[annotator_index][label_pos], 
                                                reduction='none',
                                                ignore_index=-100
                                            ) for label_pos in range(len(label))]
                                        )
                                    )

                        else:
                            raise NotImplementedError('Only supports single label classification')
                        example_losses.append(loss)

            if example_losses:
                batch_losses.append(
                    torch.mean(torch.stack(example_losses)).squeeze()
                )
            batch_logits.append(example_logits)
        
        loss_avg = torch.mean(torch.stack(batch_losses)).squeeze() if batch_losses else torch.tensor(0.0)

        if not return_dict:
            output = (torch.cat(batch_logits),) + pretrained_output[1:]
            return ((loss_avg,) + output) if loss_avg else output
        

        return SequenceClassifierOutput(
            loss=loss_avg,
            logits=torch.cat(batch_logits),
            hidden_states=pretrained_output.hidden_states,
            attentions=pretrained_output.attentions
        )