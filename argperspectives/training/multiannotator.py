from typing import Dict

import torch
from sklearn.utils.class_weight import compute_class_weight

def compute_label_weights(
        labels, # shape (num_examples, num_annotators) if one label (which can have multiple classes) or (num_examples, num_annotators, num_labels) if multiple labels (which each can have multiple classes)
        annotators_on_example,
        classes=[0,1],
        missing_annotator_val = -1
    ) -> Dict[str, torch.Tensor]:
    annotator_indecies = annotators_on_example[annotators_on_example > missing_annotator_val].unique().int().tolist()
    #TODO does it makes sense to just have it as a tuple based on indecies instead of dict?
    label_freq_dict = {}
    MISSING_LABEL_VAL = -100
    if labels.dim() == 3:
        labels_tuple = tuple(labels[:,:,i] for i in range(labels.shape[-1]))
    elif labels.dim() == 2:
        labels_tuple = [labels]
    else:
        raise ValueError(f'Labels tensor is expected to be either 2d or 3d, got labels of shape {labels.shape}')
    
    for labels in labels_tuple:       
        for annotator_index in annotator_indecies:
            if  annotators_on_example.shape[-1] == labels.shape[-1]:
                annotator_labels = labels[annotators_on_example == annotator_index]
            else:
                # assume annotator x label matrix
                annotator_labels = labels[:,annotator_index]
            annotator_labels = annotator_labels[annotator_labels > MISSING_LABEL_VAL]
            annotator_classes = annotator_labels.unique().int().numpy()
            label_weights = torch.tensor(compute_class_weight(
                    'balanced',
                    classes=annotator_classes,
                    y=annotator_labels.numpy()
                ), dtype=torch.float32)
            if len(label_weights) < len(classes):
                # set weights for missing classes to zero
                weights = torch.zeros(len(classes))
                for cl, weight in zip(annotator_classes, label_weights):
                    weights[cl] = weight
                label_weights = weights
            if annotator_index in label_freq_dict:
                label_freq_dict[annotator_index] = torch.stack((label_freq_dict[annotator_index], label_weights))
            else:
                label_freq_dict[annotator_index] = label_weights
    return label_freq_dict