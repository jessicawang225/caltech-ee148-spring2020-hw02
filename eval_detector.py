import os
import json
import numpy as np
import matplotlib.pyplot as plt


def compute_iou(box_1, box_2):
    '''
    This function takes a pair of bounding boxes and returns intersection-over-
    union (IoU) of two bounding boxes.
    '''

    left_1, top_1, right_1, bottom_1 = box_1
    left_2, top_2, right_2, bottom_2 = box_2

    left_i = max(left_1, left_2)
    top_i = max(top_1, top_2)
    right_i = min(right_1, right_2)
    bottom_i = min(bottom_1, bottom_2)

    if (left_i > right_i or top_i > bottom_i):
        i = 0
    else:
        i = (right_i - left_i) * (bottom_i - top_i)

    u = (right_1 - left_1) * (bottom_1 - top_1) + (right_2 - left_2) * (bottom_2 - top_2) - i
    iou = i / u

    assert (iou >= 0) and (iou <= 1.0)

    return iou


def compute_counts(preds, gts, iou_thr=0.5, conf_thr=0.5):
    '''
    This function takes a pair of dictionaries (with our JSON format; see ex.)
    corresponding to predicted and ground truth bounding boxes for a collection
    of images and returns the number of true positives, false positives, and
    false negatives.
    <preds> is a dictionary containing predicted bounding boxes and confidence
    scores for a collection of images.
    <gts> is a dictionary containing ground truth bounding boxes for a
    collection of images.
    '''
    TP = 0
    FP = 0
    FN = 0

    for pred_file, pred in preds.items():
        gt = gts[pred_file]
        tp, fp, fn = compute_counts_helper(gt, pred, iou_thr, conf_thr)
        TP += tp
        FP += fp
        FN += fn

    return TP, FP, FN


def compute_counts_helper(gt, pred, iou_thr, conf_thr):
    TP = 0
    FP = 0
    FN = 0

    pred = list(filter(lambda x: x[4] >= conf_thr, pred))

    seen_gt = set()
    seen_pred = set()
    for i in range(len(gt)):
        max_iou = (-1, -1, -1)
        for j in range(len(pred)):
            iou = compute_iou(gt[i], pred[j][:4])
            if iou > iou_thr and iou > max_iou[0] and j not in seen_pred:
                max_iou = (iou, j, i)
        if max_iou != (-1, -1, -1):
            TP += 1
            seen_pred.add(max_iou[1])
            seen_gt.add(max_iou[2])

    FP = len(pred) - len(seen_pred)
    FN = len(gt) - len(seen_gt)

    return TP, FP, FN


preds_path = './predictions'
gts_path = './annotations'
plots_path = './plots'
os.makedirs(plots_path, exist_ok=True)

# load splits:
split_path = './splits'
file_names_train = np.load(os.path.join(split_path, 'file_names_train.npy'))
file_names_test = np.load(os.path.join(split_path, 'file_names_test.npy'))

# Set this parameter to True when you're done with algorithm development:
done_tweaking = True

'''
Load training data. 
'''
with open(os.path.join(preds_path, 'preds_train.json'), 'r') as f:
    preds_train = json.load(f)

with open(os.path.join(gts_path, 'annotations_train.json'), 'r') as f:
    gts_train = json.load(f)

if done_tweaking:
    '''
    Load test data.
    '''

    with open(os.path.join(preds_path, 'preds_test.json'), 'r') as f:
        preds_test = json.load(f)

    with open(os.path.join(gts_path, 'annotations_test.json'), 'r') as f:
        gts_test = json.load(f)

iou_thrs = [0.25, 0.50, 0.75]
confidence_thrs = []
for fname in preds_train:
    for bounding_box in preds_train[fname]:
        confidence_thrs.append(bounding_box[4])
confidence_thrs = sorted(confidence_thrs)
for iou_thr in iou_thrs:
    tp_train = np.zeros(len(confidence_thrs))
    fp_train = np.zeros(len(confidence_thrs))
    fn_train = np.zeros(len(confidence_thrs))
    for i, conf_thr in enumerate(confidence_thrs):
        tp_train[i], fp_train[i], fn_train[i] = compute_counts(preds_train, gts_train, iou_thr=iou_thr,
                                                               conf_thr=conf_thr)

    precision = tp_train / (tp_train + fp_train)
    recall = tp_train / (tp_train + fn_train)

    fig = plt.figure()
    plt.plot(recall, precision)
    plt.xlabel('Precision')
    plt.ylabel('Recall')
    fig.savefig('{}/PR_curve_train_{}.png'.format(plots_path, str(iou_thr)))

if done_tweaking:
    iou_thrs = [0.25, 0.50, 0.75]
    confidence_thrs = []
    for fname in preds_test:
        for bounding_box in preds_test[fname]:
            confidence_thrs.append(bounding_box[4])
    confidence_thrs = sorted(confidence_thrs)
    for iou_thr in iou_thrs:
        tp_test = np.zeros(len(confidence_thrs))
        fp_test = np.zeros(len(confidence_thrs))
        fn_test = np.zeros(len(confidence_thrs))
        for i, conf_thr in enumerate(confidence_thrs):
            tp_test[i], fp_test[i], fn_test[i] = compute_counts(preds_test, gts_test, iou_thr=iou_thr,
                                                                conf_thr=conf_thr)

        precision = tp_test / (tp_test + fp_test)
        recall = tp_test / (tp_test + fn_test)

        fig = plt.figure()
        plt.plot(recall, precision)
        plt.xlabel('Precision')
        plt.ylabel('Recall')
        fig.savefig('{}/PR_curve_test_{}.png'.format(plots_path, str(iou_thr)))
