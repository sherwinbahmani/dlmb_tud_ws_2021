# Adapted from score written by wkentaro
# https://github.com/wkentaro/pytorch-fcn/blob/master/torchfcn/utils.py

import itertools
import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict


class runningScore():
    """
    Args:
        n_classes: number of classes defined in dataset
        label_true: array with shape [1, 1, 512, 512]
        label_pred: array with shape [1, 1, 512, 512]
    """

    def __init__(self,
                 n_classes: int):

        self.n_classes = n_classes
        self.confusion_matrix = np.zeros((n_classes, n_classes))

    def _fast_hist(self,
                   label_true: np.ndarray,
                   label_pred: np.ndarray,
                   n_class: int):

        mask = (label_true >= 0) & (label_true < n_class)
        hist = np.bincount(
            n_class * label_true[mask].astype(int) + label_pred[mask], minlength=n_class ** 2
        ).reshape(n_class, n_class)
        return hist

    def update(self,
               label_trues: np.ndarray,
               label_preds: np.ndarray):

        for lt, lp in zip(label_trues, label_preds):
            self.confusion_matrix += self._fast_hist(lt.flatten(), lp.flatten(), self.n_classes)

    def get_scores(self):
        """
        Returns accuracy score evaluation result.
            - overall accuracy
            - mean accuracy
            - mean IU
            - fwavacc
        """
        hist = self.confusion_matrix
        acc = np.diag(hist).sum() / hist.sum()
        acc_cls = np.diag(hist) / hist.sum(axis=1)
        acc_cls = np.nanmean(acc_cls)
        iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
        mean_iu = np.nanmean(iu)
        freq = hist.sum(axis=1) / hist.sum()
        fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
        cls_iu = dict(zip(range(self.n_classes), iu))

        return (
            {
                "Overall Acc:": acc,
                "Mean Acc :": acc_cls,
                "FreqW Acc :t": fwavacc,
                "Mean IoU :": mean_iu,
            },
            cls_iu,
            hist,
            iu,
        )

    def reset(self):
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))


class averageMeter():
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self,
               val: int,
               n: int = 1):

        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def plot_confusion_matrix(cm: np.ndarray,
                          class_names: List):
    """
    Returns a matplotlib figure containing the plotted confusion matrix.

    Args:
       cm (array, shape = [n, n]): a confusion matrix of integer classes
       class_names (list, shape = [n]): String names of the integer classes
    """

    # figure and settings
    figure = plt.figure(figsize=(8, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # Normalize the confusion matrix.
    cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)

    # Use white text if squares are dark; otherwise black.
    threshold = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        color = "white" if cm[i, j] > threshold else "black"
        plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    return figure


def plot_class_table(data_per_class: Dict,
                     class_names: List,
                     column_head: str):
    """
    Returns a matplotlib figure containing the plotted table containing the class ious.

    Args:
       data_per_class:
       class_iou (dict, shape = [n, n]): Dict including class ious
       class_names (list, shape = [n]): String names of the integer classes
    """

    if isinstance(data_per_class, torch.Tensor):
        data_per_class = dict(np.ndenumerate(data_per_class.detach().cpu().numpy()))


    # define column and row names and table cell content
    row_headers = class_names
    cell_text1 = []
    for cls_num in data_per_class: cell_text1.append([str(data_per_class[cls_num])])

    # plot table figure
    plt.figure()
    the_table = plt.table(cellText=cell_text1,
                          rowLabels=row_headers,
                          rowLoc='left',
                          colWidths=[0.33, 0.33],
                          cellLoc='right',
                          loc='center')
    # Make the rows taller (i.e., make cell y scale larger).
    the_table.scale(1, 1.5)
    # supress gca and axis
    ax = plt.gca()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.box(on=None)
    ciu_figure = plt.gcf()

    return ciu_figure