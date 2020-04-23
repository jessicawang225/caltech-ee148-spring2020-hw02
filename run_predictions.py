import os
import numpy as np
import json
from PIL import Image


def compute_convolution(I, T, stride=1):
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

    heatmap = np.zeros((n_rows, n_cols))

    for i in range(0, n_rows - m_rows, stride):
        for j in range(0, n_cols - m_cols, stride):
            image = I[i:i + m_rows, j:j + m_cols, :].flatten()
            image = normalize_image(image)
            heatmap[i][j] = np.sum(image * template)

    return heatmap


def normalize_image(v):
    '''
    This function normalizes a vector
    '''
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm


def predict_boxes(heatmap, I, T):
    '''
    This function takes heatmap and returns the bounding boxes and associated
    confidence scores.
    '''

    (n_rows, n_cols) = np.shape(heatmap)
    (m_rows, m_cols, m_channels) = np.shape(T)

    output = []

    threshold = 0.90

    i = 0
    while i < n_rows - m_rows:
        j = 0
        while j < n_cols - m_cols:
            if (heatmap[i, j] > threshold and I[i + int(m_rows / 2), j + int(m_cols / 2), 0] >= 220):
                score = np.average(heatmap[i, j])
                output.append([i, j, i + m_rows, j + m_cols, score])
                i += m_rows
                j += n_cols
            else:
                j += 1
        i += 1
    return output


def overlap(bounding_boxes):
    bounding_boxes = sorted(bounding_boxes, key=lambda x: (x[0], x[1]))
    output = [bounding_boxes[0]]

    for i in range(1, len(bounding_boxes)):
        t_prev, l_prev, b_prev, r_prev, score_prev = bounding_boxes[i - 1]
        t_curr, l_curr, b_curr, r_curr, score_curr = bounding_boxes[i]
        if (t_curr not in range(t_prev, b_prev) and b_curr not in range(t_prev, b_prev) and l_curr not in range(l_prev,
                                                                                                                r_prev) and r_curr not in range(
                l_prev, b_prev)):
            output.append(bounding_boxes[i])

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
        bounding_boxes = predict_boxes(heatmap, I, T)
        for box in bounding_boxes:
            output.append(box)

    if len(output) > 5:
        output = sorted(output, key=lambda element: element[4], reverse=True)[:5]
    if output:
        output = overlap(output)

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
done_tweaking = True

'''
Make predictions on the training set.
'''
preds_train = {}
for i in range(len(file_names_train)):
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