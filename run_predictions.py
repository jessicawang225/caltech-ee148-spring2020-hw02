import os
import numpy as np
import json
from PIL import Image


def compute_convolution(I, T, stride=2):
    '''
    This function takes an image <I> and a template <T> (both numpy arrays)
    and returns a heatmap where each grid represents the output produced by
    convolution at each location. You can add optional parameters (e.g. stride,
    window_size, padding) to create additional functionality.
    '''

    (n_rows, n_cols, n_channels) = np.shape(I)
    (m_rows, m_cols, m_channels) = np.shape(T)

    template = T.flatten()
    template = normalize_image(template)

    heatmap = [[0 for j in range(n_cols)] for i in range(n_rows)]

    for i in range(0, n_rows - m_rows, stride):
        for j in range(0, n_cols - m_cols, stride):
            image = I[i:i + m_rows, j:j + m_cols, :].flatten()
            image = normalize_image(image)
            heatmap[i][j] = np.dot(image, template)

    return heatmap


def normalize_image(v):
    '''
    This function normalizes a vector
    '''
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm


def predict_boxes(heatmap):
    '''
    This function takes heatmap and returns the bounding boxes and associated
    confidence scores.
    '''

    heatmap = np.asarray(heatmap)
    output = []

    threshold = 0.80
    redlights = list(zip(*np.where(heatmap > threshold)))

    if len(redlights) <= 1:
        return [0, 0, 0, 0, 0]

    i = 1
    widths = set()
    heights = set()
    confidence_scores = []
    widths.add(redlights[0][0])
    heights.add(redlights[0][1])
    while (i < len(redlights)):
        x, y = redlights[i]
        if (x > max(widths) + 1 or x < min(widths) - 1) and (y > max(heights) + 1 or y < min(heights) - 1):
            left = min(widths)
            top = min(heights)
            right = max(widths)
            bottom = max(heights)
            if (abs(left - right) > 3 and abs(top - bottom) > 3):
                confidence_score = np.average(np.asarray(confidence_scores))
                output.append([int(left), int(top), int(right), int(bottom) - int((int(bottom) - int(top)) * 2 / 3),
                               confidence_score])
            widths = set()
            heights = set()
            confidence_score = 0
            widths.add(redlights[i][0])
            heights.add(redlights[i][1])
        else:
            widths.add(x)
            heights.add(y)
            confidence_scores.append(heatmap[x][y])
        i += 1

    return output


def detect_red_light_mf(I):
    '''
    This function takes a numpy array <I> and returns a list <output>.
    The length of <output> is the number of bounding boxes predicted for <I>.
    Each entry of <output> is a list <[row_TL,col_TL,row_BR,col_BR,score]>.
    The first four entries are four integers specifying a bounding box
    (the row and column index of the top left corner and the row and column
    index of the bottom right corner).
    <score> is a confidence score ranging from 0 to 1.

    Note that PIL loads images in RGB order, so:
    I[:,:,0] is the red channel
    I[:,:,1] is the green channel
    I[:,:,2] is the blue channel
    '''

    output = []

    templates_path = './templates'
    templates_names = os.listdir(templates_path)

    for template in templates_names:
        T = Image.open(os.path.join(templates_path, template))
        T = np.asarray(T)
        heatmap = compute_convolution(I, T)
        bounding_boxes = predict_boxes(heatmap)
        output.extend(bounding_boxes)

    for bounding_box in output:
        left, top, right, bottom, score = bounding_box
        if ((right - left) >= 1.5 * (bottom - top) or (bottom - top) >= 1.5 * (right - left)):
            if bounding_box in output:
                output.remove(bounding_box)

    for i in range(len(output)):
        assert len(output[i]) == 5
        assert (output[i][4] >= 0.0) and (output[i][4] <= 1.0)

    return output


# Note that you are not allowed to use test data for training.
# set the path to the downloaded data:
data_path = './data'

# load splits:
split_path = './splits'
file_names_train = np.load(os.path.join(split_path, 'file_names_train.npy'))
file_names_test = np.load(os.path.join(split_path, 'file_names_test.npy'))

# set a path for saving predictions:
preds_path = './predictions'
os.makedirs(preds_path, exist_ok=True)  # create directory if needed

# Set this parameter to True when you're done with algorithm development:
done_tweaking = False

'''
Make predictions on the training set.
'''
preds_train = {}
for i in range(len(file_names_train)):
    # for i in range(2):
    if (i % 5 == 0):
        print('Detection training completed for : {}/{} images'.format(i, len(file_names_train)))
    # read image using PIL:
    I = Image.open(os.path.join(data_path, file_names_train[i]))

    # convert to numpy array:
    I = np.asarray(I)

    preds_train[file_names_train[i]] = detect_red_light_mf(I)

# save preds (overwrites any previous predictions!)
with open(os.path.join(preds_path, 'preds_train.json'), 'w') as f:
    json.dump(preds_train, f)

if done_tweaking:
    '''
    Make predictions on the test set. 
    '''
    preds_test = {}
    for i in range(len(file_names_test)):
        if (i % 5 == 0):
            print('Detection testing completed for : {}/{} images'.format(i, len(file_names_test)))
        # read image using PIL:
        I = Image.open(os.path.join(data_path, file_names_test[i]))

        # convert to numpy array:
        I = np.asarray(I)

        preds_test[file_names_test[i]] = detect_red_light_mf(I)

    # save preds (overwrites any previous predictions!)
    with open(os.path.join(preds_path, 'preds_test.json'), 'w') as f:
        json.dump(preds_test, f)