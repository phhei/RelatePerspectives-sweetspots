from typing import Union, List

import numpy
from sklearn.metrics import roc_curve


def compute_optimal_threshold(predicted: Union[List[float], numpy.ndarray],
                              reference: Union[List[float], numpy.ndarray]) -> float:
    """
    Compute the optimal threshold when predicting the positive label having a binary classification task

    :param predicted: a 1d-array of the predicted probabilities for belonging to the positive class
    :param reference: a 1d-array (of equal length) of the reference labels (0-NEGATIVE CLASS/ 1-POSITIVE CLASS)
    :return: the optimal threshold (>= probability) when to go with the positive class
    """

    # logger.trace("OK, having {} entries...", len(predicted))
    try:
        fpr, tpr, thresholds = roc_curve(y_score=predicted, y_true=reference, pos_label=1)
        true_false_rate = tpr - fpr
        ix = numpy.argmax(true_false_rate)
        # logger.trace("Found following true-positive-rates: {}, false-positive-rates: {} "
        #              "under following thresholds: {}", tpr, fpr, thresholds)
        threshold = thresholds[ix]
        # logger.info("Found the optimal threshold: {}", round(threshold, 5))
    except ValueError:
        # logger.opt(exception=True).critical("Something went wrong in calculating the optimal threshold "
        #                                     "(fall back to .5). "
        #                                     "The values for predicted_arg_kp_matches: {}. "
        #                                     "The values for ground_truth_arg_kp_matches: {}",
        #                                     predicted.tolist(),
        #                                     reference.tolist())
        threshold = .5
    return threshold
