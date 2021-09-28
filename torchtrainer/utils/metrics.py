import torch
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, balanced_accuracy_score
from scipy.optimize import brentq
from scipy.interpolate import interp1d

def positive_negative_num(predictions, labels):
    conf_vector = predictions.reshape(labels.shape) / labels

    # model 1 - label 1
    tp = torch.sum(conf_vector == 1, dim=-1)
    # model 1 - label 0
    fp = torch.sum(torch.isinf(conf_vector), dim=-1) 
    # model 0 - label 0
    tn = torch.sum(torch.isnan(conf_vector), dim=-1)
    # model 0 - label 1
    fn = torch.sum(conf_vector == 0, dim=-1)

    return tp, fp, tn, fn

def triplet_accuracy(distp, distn, margin):
    pred = distn - distp - margin 
    return (pred > 0).sum()*1.0/distp.shape[0]

def classification_accuracy(predictions, labels):
    acc = torch.sum((predictions.reshape(-1).long() == labels.long())) / (labels.size(0))
    return acc

'''
Accuracy for the case of unbalanced dataset
Assumed that the batch size is the first dimension
Return the balanced accuracy averaged over the batch dimension
'''
def running_balanced_accuracy(predictions, labels):
    sens = sensitivity(predictions, labels)
    spec = specificity(predictions, labels)
    
    balanced_acc = (sens + spec) / 2
    # compute the mean over the batch
    mean_balaced_acc = balanced_acc.mean()
    return mean_balaced_acc

'''
True positive rate, tp / (tp + fn)
The percentage of model predictions of class 1 that are correct
'''
def sensitivity(predictions, labels):
    tp, _, _, fn = positive_negative_num(predictions, labels) 

    sens = tp / (tp + fn)
    # handle the case of no positive example in the labels
    sens[torch.isnan(sens)] = 0.0

    return sens

'''
True negative rate, tn / (fp + tn)
The percentage of model predictions of class 0 that are correct
'''
def specificity(predictions, labels):
    _, fp, tn, _ = positive_negative_num(predictions, labels) 

    spec = tn / (fp + tn)
    # handle the case of no negative example in the labels
    spec[torch.isnan(spec)] = 0.0

    return spec 

# sklearn metrics
def balanced_accuracy(predictions, labels):
    acc = balanced_accuracy_score(labels.cpu(), predictions.cpu()) 
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
