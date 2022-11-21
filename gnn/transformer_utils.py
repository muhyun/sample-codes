#-*- coding:utf-8 -*-

# Author:james Zhang
# Datetime:20-2-23 上午9:37
# Project: RED
# Helper file, including fuctions that can do metric eveluation or visualization


from sklearn.metrics import roc_auc_score, confusion_matrix, f1_score, roc_curve, auc, precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt
import datetime
import time
import numpy as np


def plot_roc(y_true, y_pred_prob, fname=None):
    fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
    roc_auc = auc(fpr, tpr)

    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.4f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    if fname is None:
        plt.show()
    else:
        plt.savefig(fname)


def get_f1_score(y_true, y_pred):
    """
    Attention!
    tn, fp, fn, tp = cf_m[0,0],cf_m[0,1],cf_m[1,0],cf_m[1,1]

    :param y_true:
    :param y_pred:
    :return:
    """
    # print(y_true, y_pred)

    cf_m = confusion_matrix(y_true, y_pred)
    print(cf_m)

    precision = cf_m[1,1] / (cf_m[1,1] + cf_m[0,1] + 10e-5)
    recall = cf_m[1,1] / (cf_m[1,1] + cf_m[1,0])
    f1 = 2 * (precision * recall) / (precision + recall + 10e-5)

    # My caculation is same as f1_score.
    # val_f1 = f1_score(y_true, y_pred)
    # print(val_f1)

    return precision, recall, f1


def get_recall(y_true, y_pred):
    """
    Attention!
    tn, fp, fn, tp = cf_m[0,0],cf_m[0,1],cf_m[1,0],cf_m[1,1]

    :param y_true:
    :param y_pred:
    :return:
    """
    cf_m = confusion_matrix(y_true, y_pred)

    # print(cf_m)
    return cf_m[1,1] / (cf_m[1,1] + cf_m[1,0])


def get_auc_score(y_true, y_pred_prob):
    return roc_auc_score(y_true, y_pred_prob)


def plot_p_r_curve(y_true, y_pred_prob, fname=None):

    precision, recall, thresholds = precision_recall_curve(y_true, y_pred_prob)

    avp = average_precision_score(y_true, y_pred_prob)

    plt.plot(recall, precision, color='blue', lw=2, label='P-R Curve with Avergage Precision {:.4f}'.format(avp))
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve for Binary Classification')
    plt.legend(loc="lower left")
    if fname is None:
        plt.show()
    else:
        plt.savefig(fname)


def recall_at_perc_precision(y_true, y_pred_prob, threshold):
    """
    Get recall value at given threshold precision value. This will help to evaluate skewed ground truth.
    :param y_true:
    :param y_pred_prob:
    :return:
    """

    precision, recall, thresholds = precision_recall_curve(y_true, y_pred_prob)

    # print(precision)
    # print(recall)

    # use the last above threshold precision value as the index of recall
    recall_result = 0
    for i, p in enumerate(precision):
        if p >= threshold:
            recall_result = recall[i]
            break

    return recall_result


def time_diff(t_end, t_start):
    """
    t_end, t_start are datetime format, so use deltatime
    Parameters
    ----------
    t_end
    t_start

    Returns
    -------
    """
    diff_sec = (t_end - t_start).seconds
    diff_min, rest_sec = divmod(diff_sec, 60)
    diff_hrs, rest_min = divmod(diff_min, 60)
    return (diff_hrs, rest_min, rest_sec)


def build_one_hot(input, num_class=None, from_zero=True):
    """
    Args:
        input is a Numpy ndarry 1D (N,) assuming all integer and >0
        num_class is the output dimension. If is None, will use the largest value in input
    Return:
        Numpy Ndarry, (N, num_class)
    """
    if not from_zero:
        input = input - 1

    if num_class==None:
        num_class = input.max()

    return np.eye(num_class)[input]


if __name__ == '__main__':
    y_true = [1,1,1,0,0,0]
    # y_pred = [0,0,0,0,0,0]
    y_pred = [1, 1, 1, 1, 1, 1]
    # y_pred_prob = [0.95, 0.8, 0.45, 0.65, 0.2, 0.1]
    y_pred_prob = [0.15, 0.28, 0.65, 0.45, 0.82, 0.91]

    p, r, f1 = get_f1_score(y_true, y_pred)
    print(p, r, f1)

    print(recall_at_perc_precision(y_true, y_pred_prob, 0.6))

    # plot_p_r_curve(y_true, y_pred_prob)

    t_s = datetime.datetime.now()
    time.sleep(2)
    t_e = datetime.datetime.now()
    h, m, s = time_diff(t_e, t_s)
    print('{:02d}h {:02d}m {:02}s'.format(h, m, s))

    input = np.array([1,3,1,2,2,4,6,4,5])
    print(build_one_hot(input, 6, from_zero=False))