import os
import json
import numpy as np


def compute_iou(box_1, box_2):
    '''
    This function takes a pair of bounding boxes and returns intersection-over-
    union (IoU) of two bounding boxes.
    '''

    left_1, top_1, right_1, bottom_1, score_1 = box_1
    left_2, top_2, right_2, bottom_2, score_2 = box_2

    left_i = max(left_1, left_2)
    top_i = max(top_1, top_2)
    right_i = min(right_1, right_2)
    bottom_i = min(bottom_1, bottom_2)

    i = (right_i - left_i) * (bottom_i - top_i)
    u = (right1 - left1) * (bottom1 - top1) + (right2 - left2) * (bottom2 - top2) - i

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

    seen_gt = []
    seen_pred = []

    for pred_file, pred in preds.iteritems():
        gt = gts[pred_file]
        for i in range(len(gt)):
            for j in range(len(pred)):
                iou = compute_iou(pred[j][:4], gt[i])
                if iou >= conf_thr and gt[i] not in seen_gt and pred[j][:4] not in seen_pred:
                    TP += 1
                    seen_gt.append(gt[i])
                    seen_pred.append(pred[j][:4])

    FP = len(pred) - len(seen_pred)
    FN = len(gt) - len(seen_gt)

    return TP, FP, FN


# set a path for predictions and annotations:
preds_path = './predictions'
gts_path = './annotations'

# load splits:
split_path = './splits'
file_names_train = np.load(os.path.join(split_path, 'file_names_train.npy'))
file_names_test = np.load(os.path.join(split_path, 'file_names_test.npy'))

# Set this parameter to True when you're done with algorithm development:
done_tweaking = False

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

# For a fixed IoU threshold, vary the confidence thresholds.
# The code below gives an example on the training set for one IoU threshold.

confidence_thrs = np.sort(np.array([preds_train[fname][4] for fname in preds_train],
                                   dtype=float))  # using (ascending) list of confidence scores as thresholds
tp_train = np.zeros(len(confidence_thrs))
fp_train = np.zeros(len(confidence_thrs))
fn_train = np.zeros(len(confidence_thrs))
for i, conf_thr in enumerate(confidence_thrs):
    tp_train[i], fp_train[i], fn_train[i] = compute_counts(preds_train, gts_train, iou_thr=0.5, conf_thr=conf_thr)

# Plot training set PR curves

if done_tweaking:
    print('Code for plotting test set PR curves.')
