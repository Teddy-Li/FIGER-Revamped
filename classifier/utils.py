import json
from typing import Dict
import numpy as np
import torch
from transformers import EvalPrediction


# Below are borrowed from ./utils/utils.py from CFET project

def microf1(true_labels_list, pred_labels_list):
    assert len(true_labels_list) == len(pred_labels_list)
    l_true_cnt, l_pred_cnt, hit_cnt = 0, 0, 0
    for labels_true, labels_pred in zip(true_labels_list, pred_labels_list):
        # print(f"labels_true: {labels_true}")
        # print(f"labels_pred: {labels_pred}")
        curr_match_cnt = (labels_true * labels_pred).sum()
        # print(f"curr_match_cnt: {curr_match_cnt}")
        hit_cnt += curr_match_cnt

        l_true_cnt += labels_true.sum()
        l_pred_cnt += labels_pred.sum()
    p = hit_cnt / l_pred_cnt
    r = hit_cnt / l_true_cnt
    return 2 * p * r / (p + r + 1e-7)


def macrof1(true_labels_list, pred_labels_list, return_pnr=False):
    assert len(true_labels_list) == len(pred_labels_list)
    p_acc, r_acc = 0, 0
    for labels_true, labels_pred in zip(true_labels_list, pred_labels_list):
        num_preds, num_trues = labels_pred.sum(), labels_true.sum()
        if num_preds == 0 or num_trues == 0:
            continue

        match_cnt = (labels_true * labels_pred).sum()
        p_acc += match_cnt / num_preds
        r_acc += match_cnt / num_trues
    p, r = p_acc / len(pred_labels_list), r_acc / len(true_labels_list)
    f1 = 2 * p * r / (p + r + 1e-7)
    if return_pnr:
        return f1, p, r
    else:
        return f1


# Ratio of entries where the predictions and ture labels are exactly the same
def strict_acc(true_labels_list, pred_labels_list):
    assert len(true_labels_list) == len(pred_labels_list)
    hit_cnt = 0
    for labels_true, labels_pred in zip(true_labels_list, pred_labels_list):
        if np.array_equal(labels_true, labels_pred):
            hit_cnt += 1
    return hit_cnt / len(true_labels_list)


# Ratio of entries where the prediction and true labels have overlaps
def partial_acc(true_labels_list, pred_labels_list):
    assert len(true_labels_list) == len(pred_labels_list)
    hit_cnt = 0
    for labels_true, labels_pred in zip(true_labels_list, pred_labels_list):
        if (labels_pred*labels_true).sum() > 0:
            hit_cnt += 1
    return hit_cnt / len(true_labels_list)


# Above are borrowed from ./utils/utils.py from CFET project


def compute_metrics(p: EvalPrediction, eval_method: str = 'macrof1') -> Dict:
    """Compute metrics."""
    predictions, true_labels = p.predictions, p.label_ids

    eval_method = eval(eval_method)
    thresholds = [x*0.05 for x in range(1, 20)]

    best_macrof1 = 0
    best_macrof1_prec = None
    best_macrof1_rec = None
    best_microf1 = 0
    best_strict_acc = 0
    best_partial_acc = 0
    print(f"Beginning to compute metrics...")

    for thres in thresholds:
        print(f"Threshold: {thres}")
        # print(f"Predictions size: {predictions.shape}")
        pred_labels = np.where(predictions > thres, 1, 0)
        # print(f"pred labels shape: {pred_labels.shape}")
        # print(f"true labels shape: {true_labels.shape}")

        # print("!")
        s_acc = strict_acc(true_labels, pred_labels)
        # print("!!")
        p_acc = partial_acc(true_labels, pred_labels)
        # print("!!!")
        macro_f1, prec, rec = macrof1(true_labels, pred_labels, return_pnr=True)
        # print("!!!!")
        micro_f1 = microf1(true_labels, pred_labels)
        # print("!!!!!")

        if s_acc > best_strict_acc:
            best_strict_acc = s_acc
        if p_acc > best_partial_acc:
            best_partial_acc = p_acc
        if macro_f1 > best_macrof1:
            best_macrof1 = macro_f1
            best_macrof1_prec = prec
            best_macrof1_rec = rec
        if micro_f1 > best_microf1:
            best_microf1 = micro_f1
        # print("!!!!!!")

    print(f'Best Macro F1: {best_macrof1}; Precision: {best_macrof1_prec}; Recall: {best_macrof1_rec}')
    print(f'Best Micro F1: {best_microf1}')
    print(f'Best Strict Accuracy: {best_strict_acc}')
    print(f'Best Partial Accuracy: {best_partial_acc}')

    return {
        'eval_strict_acc': best_strict_acc,
        'eval_partial_acc': best_partial_acc,
        'eval_macro_f1': best_macrof1,
        'eval_micro_f1': best_microf1,
    }


def get_labelset(typeset_fn):
    with open(typeset_fn, 'r') as f:
        labelset = json.load(f)
    return labelset
