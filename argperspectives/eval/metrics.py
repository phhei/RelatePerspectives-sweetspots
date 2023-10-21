import numpy as np
import scipy
import math

from transformers import EvalPrediction
from sklearn.metrics import classification_report

from typing import (
    Dict,
    List,
    Optional
)

class MajorityLabelMetrics:
        
        def __init__(self, classes: List[int]):
            self.classes = classes

        def compute(self, p: EvalPrediction) -> Dict:
            logits = p.predictions
            labels = p.label_ids

            logits_argmax = np.argmax(logits, axis=-1)
            report = classification_report(
                        y_true=labels, 
                        y_pred=logits_argmax,
                        labels = self.classes,
                        output_dict=True
                    )
            
            result = {}
            for name, metric in report.items():
                if isinstance(metric, dict):
                    metric_dict = {f'majority_{name}_{k}':v for k, v in metric.items()}
                else:
                    metric_dict = {f'majority_{name}': metric}
                result.update(metric_dict)
            return result

def majority(labels_per_example, missing_label_val = -1):
    if labels_per_example.ndim == 1:
        return labels_per_example
    if labels_per_example.ndim > 2:
        raise ValueError(f'ndim of labels per example has to be 2 or 1, got {labels_per_example.ndim}')
    if labels_per_example.shape[-1] == 1:
        return labels_per_example.flatten()
    # docs on multiple modes:
    # "If there is more than one such value, only the smallest is returned"
    return np.apply_along_axis(lambda a: scipy.stats.mode(a[a > missing_label_val], nan_policy='omit').mode, 1, labels_per_example).flatten()

from collections import defaultdict
class MultiAnnotatorMetrics:
        
        def __init__(self, 
                     annotator_ids_to_indecies: Dict[str, int], 
                     classes: List[int],
                     is_annotator_set_batches=False,
                     label_names = None):
            self.all_annotator_indecies = tuple(sorted(annotator_ids_to_indecies.values()))
            self.classes = classes
            self.is_annotator_set_batches = is_annotator_set_batches
            self.label_names = label_names

        def compute_single_predictions(self, p: EvalPrediction) -> Dict:
            logits = p.predictions
            labels, annotator_indecies = p.label_ids

            if labels.ndim == 3:
                result = {}
                num_labels = labels.shape[2]
                logits = logits.reshape((
                    logits.shape[0],    # num_examples
                    num_labels,         # num_labels (e.g. if validity and novelty then 2) 
                    len(self.classes))  # num_classes (e.g. if negative, undecided and positive then 3) 
                )
                for label_pos in range(num_labels):
                    logits_argmax = np.argmax(logits[:,:,label_pos], axis=-1) # num_examples
                    argmax_reshaped = logits_argmax.reshape((logits_argmax.shape[0], 1)) # num_examples x 1 
                    repeated_majority_predictions = np.repeat(argmax_reshaped, annotator_indecies.shape[-1], axis = 1) # num_examples x num_annotations_per_example
                    # repeated prediction for comparision with individual labeling decisions
                    predictions_for_known_annotators = repeated_majority_predictions.flatten().astype(int)
                    partial_results = self._compute_reports(
                        logits_argmax, 
                        predictions_for_known_annotators, 
                        labels[:,:,label_pos], 
                        annotator_indecies
                    )
                    label_name = self.label_names[label_pos] if self.label_names else str(label_pos) 
                    partial_results = {f'{label_name}_{k}': v for k,v in partial_results.items()}
                    result.update(partial_results)
            else:
                logits_argmax = np.argmax(logits, axis=-1) # num_examples
                argmax_reshaped = logits_argmax.reshape((logits_argmax.shape[0], 1)) # num_examples x 1 
                repeated_majority_predictions = np.repeat(argmax_reshaped, annotator_indecies.shape[-1], axis = 1) # num_examples x num_annotations_per_example
                # repeated prediction for comparision with individual labeling decisions
                predictions_for_known_annotators = repeated_majority_predictions.flatten().astype(int)
                result = self._compute_reports(
                    logits_argmax, 
                    predictions_for_known_annotators, 
                    labels, 
                    annotator_indecies
                )

            return result

        def compute(self, p: EvalPrediction) -> Dict:
            if type(p.predictions) == tuple:
                logits = p.predictions[0]
            else:
                logits = p.predictions
            labels, annotator_indecies = p.label_ids

            if labels.ndim == 3:
                result = {}
                num_labels = labels.shape[2]
                for label_pos in range(num_labels):
                    partial_results = self._prepare_eval_results(
                        logits[:,:,label_pos], 
                        labels[:,:,label_pos], 
                        annotator_indecies)
                    label_name = self.label_names[label_pos] if self.label_names else str(label_pos) 
                    partial_results = {f'{label_name}_{k}': v for k,v in partial_results.items()}
                    result.update(partial_results)
            else:
                result = self._prepare_eval_results(logits, labels, annotator_indecies)

            return result
        
        def _reconstruct_matrix(self, labels, indecies_sets, batch_size=8, missing_label_value=-100):
                """Takes labels/logits for batch-specific annotator sets and their annotator indecies 
                to reconstruct the full matrix format.
                
                While in the batch, annotator 3 might be at index 2 depending on the other annotators in the batch, 
                in the output they will be at index 3."""
                num_annotators = len(self.all_annotator_indecies)
                matrix_labels_list = []
                for i, annotators in enumerate(indecies_sets):
                    batch_labels = labels[i*batch_size:(i+1)*batch_size]
                    matrix_labels_batch = []
                    for a in range(num_annotators):
                        if a in annotators:
                            matrix_labels_col = batch_labels[:,annotators.index(a)]  
                        else:
                            matrix_labels_col = np.tile(missing_label_value, batch_labels[:,a].shape)
                        matrix_labels_batch.append(matrix_labels_col)
                    matrix_labels_list.append(np.stack(matrix_labels_batch, axis=1))
                labels = np.concatenate(matrix_labels_list)
                return labels
               
        def _prepare_eval_results(self, logits, labels, annotator_indecies):
            # NOTE if annotator set batches, then label indecies are only meaningful in relation to the batch
            if self.is_annotator_set_batches:
                # reconstruct per-batch annotator sets
                indecies_sets = []
                current_set = []
                for i, (x,y) in enumerate(zip(annotator_indecies, annotator_indecies[1:])):
                    current_set.append(x)
                    if x > y:
                        indecies_sets.append(current_set)
                        current_set = []
                    if i == len(annotator_indecies) - 2:
                        current_set.append(y)
                        indecies_sets.append(current_set)
                # reconstruct label matrix from per-batch set-based labels
                labels = self._reconstruct_matrix(labels, indecies_sets)
                logits = self._reconstruct_matrix(logits, indecies_sets) #FIXME is this enough to fix wrong label count per annotator?
                # reconstruct annotator indecies 
                full_indecies = np.tile(np.arange(0, labels.shape[-1]), (labels.shape[0], 1))
                annotator_indecies = np.where(labels > -100, full_indecies, -1)

            logits_argmax = np.argmax(logits, axis=-1)

            majorities = majority(logits_argmax)

            if logits_argmax.shape[-1] == 1:
                # each row is for only a single annotator (from recommender)
                predictions_for_known_annotators = logits_argmax.flatten()
            else:
                # each row is for multiple single annotators (from multi-annotator)
                predictions_for_known_annotators = [label for logits, annotators in zip(logits_argmax, annotator_indecies) for label in np.take(logits, annotators[annotators > -1].astype(int))]

            result = self._compute_reports(
                majorities,
                predictions_for_known_annotators, 
                labels, 
                annotator_indecies
            )
            result['labels_raw'] = labels.tolist()
            result['logits'] = logits.tolist()
            result['annotator_indecies'] = annotator_indecies.tolist()
            return result


        def _compute_reports(self, majorities, predictions_for_known_annotators, labels, annotator_indecies, missing_label_value=-100) -> Dict:

            # if labels are of higher dimensionality, they are interpreted to be label indecies
            # from each annotator for each example with "-1" indicating no label from the respective annotator
            references = majority(labels)
            result = {}
            
            known_annotators = annotator_indecies[annotator_indecies > -1]
            if self.is_annotator_set_batches:
                # If annotator sets, then annotator indecies already have been filtered
                # by available labels, i.e. if an annotator did not assign a class for a specific label
                # we have already excluded that annotator
                # Thus, we need to filter missing values from labels to get the same length/shape
                labels = labels[labels > missing_label_value]

            unique_annotators = np.unique(known_annotators)
            for annotator in self.all_annotator_indecies:
                if annotator in unique_annotators:
                    preds_labels_for_annotator = [(pred, label) for a, pred, label in zip(known_annotators, predictions_for_known_annotators, labels.flatten()) if (a == annotator) and (label > missing_label_value)]
                    labels_for_annotator = [label for _, label in preds_labels_for_annotator]
                    predictions_for_annotator = [pred for pred, _ in preds_labels_for_annotator]
                else:
                    labels_for_annotator = [0]
                    predictions_for_annotator = [1]

                report_for_annotator = classification_report(
                        y_true=labels_for_annotator, 
                        y_pred=predictions_for_annotator,
                        #TODO make configurable
                        labels = self.classes,
                        output_dict=True,
                        zero_division=0
                    )
                for name, metric in report_for_annotator.items():
                    if isinstance(metric, dict):
                        metric_dict = {f'annotator_{annotator}_{name}_{k}':v for k, v in metric.items()}
                    else:
                        metric_dict = {f'annotator_{annotator}_{name}': metric}
                    result.update(metric_dict)


            majority_report = classification_report(
                        y_true=references, 
                        y_pred=majorities,
                        #TODO make configurable
                        labels = self.classes,
                        output_dict=True,
                        zero_division=0
                    )
            
            individual_references = labels[labels > missing_label_value]
            # if annotator batches, the labels have been filtered for missing label value
            # else we need to include missing labels so that numbers of values for annotator indecies and labels match
            labels_including_missing = labels if self.is_annotator_set_batches else labels[annotator_indecies > -1]
            predictions_for_individual_references = [pred for pred, label in zip(predictions_for_known_annotators, labels_including_missing) if label > missing_label_value]
            
            individual_report = classification_report(
                        y_true=individual_references, 
                        y_pred=predictions_for_individual_references,
                        #TODO make configurable
                        labels = self.classes,
                        output_dict=True,
                        zero_division=0
                    )

            for name, metric in majority_report.items():
                if isinstance(metric, dict):
                    metric_dict = {f'majority_{name}_{k}':v for k, v in metric.items()}
                else:
                    metric_dict = {f'majority_{name}': metric}
                result.update(metric_dict)

            for name, metric in individual_report.items():
                if isinstance(metric, dict):
                    metric_dict = {f'individual_{name}_{k}':v for k, v in metric.items()}
                else:
                    metric_dict = {f'individual_{name}': metric}
                result.update(metric_dict)

            # delete accuracy / micro averages if present for uniform result format
            # see documenatation on return type https://scikit-learn.org/1.0/modules/generated/sklearn.metrics.classification_report.html
            result = {k: v for k, v in result.items() if 'micro avg' not in k and 'accuracy' not in k}

            result['predictions_per_annotator'] = predictions_for_individual_references
            result['labels_per_annotator'] = individual_references.tolist()
            result['predictions_majority'] = majorities.tolist()
            result['labels_majority'] = references.tolist()

            return result