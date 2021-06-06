from typing import Tuple

import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn

import config


def get_class_weight(path=f"{config.data_root}/train.csv"):
    labels = pd.read_csv(path, usecols=["label"])
    classes_count = labels["label"].value_counts().sort_index().values
    class_weight = [(len(labels) / (len(classes_count) * y)) for y in classes_count]
    return torch.tensor(class_weight, dtype=torch.float)


def get_predict_index(pred: torch.Tensor, threshold=config.threshold):
    # detach graph
    pred = pred.detach()

    # get probabilities
    pred = F.softmax(pred, dim=1)

    # if the prob of "isnull" is less than threshold, set the prob of "isnull" to threshold
    indices = torch.logical_not(torch.gt(pred[:, -1], threshold))
    pred[:, -1][indices] = threshold

    # argmax
    pred = pred.argmax(dim=-1)
    return pred


def compute_acc(pred, target):
    return (get_predict_index(pred) == target).float().mean()


class F1Score:
    """
    Class for f1 calculation in Pytorch.
    """

    def __init__(self, num_classes=801):
        """
        Init.

        Args:
            average: averaging method
        """
        self.num_classes = 801

    @staticmethod
    def calc_f1_count_for_label(predictions: torch.Tensor,
                                labels: torch.Tensor, label_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate f1 and true count for the label

        Args:
            predictions: tensor with predictions
            labels: tensor with original labels
            label_id: id of current label

        Returns:
            f1 score and true count for label
        """
        # label count
        true_count = torch.eq(labels, label_id).sum()

        # true positives: labels equal to prediction and to label_id
        true_positive = torch.logical_and(torch.eq(labels, predictions),
                                          torch.eq(labels, label_id)).sum().float()
        # precision for label
        precision = torch.div(true_positive, torch.eq(predictions, label_id).sum().float())
        # replace nan values with 0
        precision = torch.where(torch.isnan(precision),
                                torch.zeros_like(precision).type_as(true_positive),
                                precision)

        # recall for label
        recall = torch.div(true_positive, true_count)
        # f1
        f1 = 2 * precision * recall / (precision + recall)
        # replace nan values with 0
        f1 = torch.where(torch.isnan(f1), torch.zeros_like(f1).type_as(true_positive), f1)
        return f1

    def __call__(self, predictions: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Calculate f1 score based on averaging method defined in init.

        Args:
            predictions: tensor with predictions
            labels: tensor with original labels

        Returns:
            f1 score
        """
        if predictions.ndim == 2:
            predictions = predictions.argmax(dim=-1)

        f1_score = 0
        for label_id in labels.unique():
            f1 = self.calc_f1_count_for_label(predictions, labels, label_id)
            f1_score += f1

        f1_score = torch.div(f1_score, len(labels.unique()))

        return f1_score
