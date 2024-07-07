import numpy as np
from sklearn.metrics import precision_recall_fscore_support

def calculate_metrics_for_classes(predictions, references, class_list):
    metrics_dict = {}
    for class_id in class_list:
        y_pred = (predictions == class_id).flatten()
        y_true = (references == class_id).flatten()
        p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', pos_label=1)
        iou = get_iou(y_pred, y_true)
        metrics_dict[class_id] = {'precision': p, 'recall': r, 'f1-score': f, 'iou': iou}
    return metrics_dict

def adjust_labels_for_metrics(predictions, labels):
    adjusted_labels = np.where(labels == 255, 1, labels)
    return predictions, adjusted_labels

def get_iou(gt_mask, pred_mask):
    intersection = np.logical_and(pred_mask, gt_mask)
    union = np.logical_or(pred_mask, gt_mask)
    # If there is no union, we return 0
    if np.any(union):
        return np.sum(intersection) / np.sum(union)
    else:
        return 0