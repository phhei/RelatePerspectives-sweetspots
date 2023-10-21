from typing import Any, Dict, List
from dataclasses import dataclass
from transformers.data.data_collator import (
    DataCollatorMixin,
    default_data_collator
)

@dataclass
class AnnotatorSetDataCollator(DataCollatorMixin):

    def __init__(self, 
                 annotators_key: str = 'annotator_indecies',
                 labels_key: str = 'labels'):
        self.annotators_key = annotators_key
        self.labels_key = labels_key

    """
    Args:
        return_tensors (`str`):
            The type of Tensor to return. Allowable values are "np", "pt" and "tf".
    """

    return_tensors: str = "pt"

    def __call__(self, features: List[Dict[str, Any]], return_tensors=None) -> Dict[str, Any]:
        if return_tensors is None:
            return_tensors = self.return_tensors
        collated = default_data_collator(features, return_tensors)
        collated[self.annotators_key] = collated[self.annotators_key].flatten().unique()
        collated[self.labels_key] = collated[self.labels_key][:, collated[self.annotators_key]]
        return collated