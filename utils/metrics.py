import torch
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, balanced_accuracy_score
from scipy.optimize import brentq
from scipy.interpolate import interp1d

def positive_negative_num(predictions, labels):
    conf_vector = predictions.reshape(labels.shape) / labels

    tp = torch.sum(conf_vector == 1)
    fp = torch.sum(conf_vector == float('inf'))
    tn = torch.sum(torch.isnan(conf_vector))
    fn = torch.sum(conf_vector == 0)

    return tp, fp, tn, fn

def triplet_accuracy(distp, distn, margin):
    pred = distn - distp - margin 
    return (pred > 0).sum()*1.0/distp.shape[0]

def classification_accuracy(predictions, labels):
    acc = torch.sum((predictions.reshape(-1).long() == labels.long())) / (labels.size(0))
    return acc

def running_balanced_accuracy(predictions, labels):
    sens = sensitivity(predictions, labels)
    spec = specificity(predictions, labels)
    
    return (sens + spec) / 2

def sensitivity(predictions, labels):
    tp, _, _, fn = positive_negative_num(predictions, labels) 
    return tp / (tp + fn)

def specificity(predictions, labels):
    _, fp, tn, _ = positive_negative_num(predictions, labels) 
    return tn / (fp + tn)

def balanced_accuracy(predictions, labels):
    acc = balanced_accuracy_score(labels, predictions) 
    return acc

def roc_auc(y_true, y_pred):
    return roc_auc_score(y_true, y_pred)

def equal_error_rate(y_true, y_pred):
    fpr, tpr, thresholds = roc_curve(y_true, y_pred, pos_label=1)
    eer = brentq(lambda x : 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    thresh = interp1d(fpr, thresholds)(eer)
    return eer, thresh

def get_confusion_matrix(y_true, y_pred):
    return confusion_matrix(y_true, y_pred)
