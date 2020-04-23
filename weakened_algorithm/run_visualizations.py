import json
import numpy as np
from PIL import Image, ImageDraw
import os


def draw(I, boxes):
    for box in boxes:
        draw = ImageDraw.Draw(I)
        # Draw bounding box in neon yellow
        top, left, bottom, right = box[:4]
        draw.rectangle([left, top, right, bottom], outline=(204, 255, 0))
        del draw
    return I


# set the path to the downloaded data:
data_path = './data'

# set a path for saving predictions:
preds_path = './predictions'

# set a path for saving visualizations:
vis_path = './visualizations'

# load splits:
split_path = './splits'
file_names_train = np.load(os.path.join(split_path, 'file_names_train.npy'))
file_names_test = np.load(os.path.join(split_path, 'file_names_test.npy'))

# get bounding boxes
with open(os.path.join(preds_path, 'preds_train.json')) as f:
    bounding_boxes_train = json.load(f)

with open(os.path.join(preds_path, 'preds_test.json')) as f:
    bounding_boxes_test = json.load(f)

for i in range(len(file_names_train)):
    # read image using PIL:
    I = Image.open(os.path.join(data_path, file_names_train[i]))

    I = draw(I, bounding_boxes_train[file_names_train[i]])

    I.save(os.path.join(vis_path, file_names_train[i]))

for i in range(len(file_names_test)):
    # read image using PIL:
    I = Image.open(os.path.join(data_path, file_names_test[i]))

    I = draw(I, bounding_boxes_test[file_names_test[i]])

    I.save(os.path.join(vis_path, file_names_test[i]))