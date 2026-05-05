import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def evaluation4class(pred, label):
    pred = pred.cpu().numpy()
    label = label.cpu().numpy()
    classes = [0, 1, 2, 3]
    acc_all = accuracy_score(label, pred)
    results = []
    for c in classes:
        binary_pred = (pred == c).astype(int)
        binary_label = (label == c).astype(int)
        acc = accuracy_score(binary_label, binary_pred)
        prec = precision_score(binary_label, binary_pred, zero_division=0)
        rec = recall_score(binary_label, binary_pred, zero_division=0)
        f1 = f1_score(binary_label, binary_pred, zero_division=0)
        results.extend([acc, prec, rec, f1])
    return (acc_all,) + tuple(results)
