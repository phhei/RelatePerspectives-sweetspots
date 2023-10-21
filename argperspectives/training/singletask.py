import torch
import torch.nn as nn
import transformers


class WeightedTrainer(transformers.Trainer):
    def __init__(self, label_weights = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.label_weights = label_weights
        if torch.cuda.is_available():
             self.label_weights = self.label_weights.cuda()

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        # forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")
        # compute custom loss (suppose one has 2 labels with different weights)
        # loss_fct = nn.CrossEntropyLoss(weight=self.label_weights)
        # if torch.cuda.is_available():
        #     loss_fct = loss_fct.cuda()
        if labels.shape[-1] == 1:
            loss = torch.nn.functional.cross_entropy(
                input=logits.view(-1, self.model.num_labels_tuple[-1]),
                target=labels.view(-1),
                weight=self.label_weights, 
                reduction='mean',
                ignore_index=-100
            )
        elif labels.shape[-1] == 2:
            loss = torch.mean(
                    torch.stack(
                        tensors=[
                            torch.nn.functional.cross_entropy(
                            input=logits.view(-1, self.model.num_labels_tuple[-1])[label_pos],
                            target=labels.view(-1)[label_pos],
                            weight=self.label_weights[label_pos], 
                            reduction='none',
                            ignore_index=-100
                        ) for label_pos in range(labels.shape[-1])]
                    )
                )
        else:
            raise ValueError('Expect labels dimensionality 1 or 2')
        return (loss, outputs) if return_outputs else loss