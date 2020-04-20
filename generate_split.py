import numpy as np
import os

np.random.seed(2020)  # to ensure you always get the same train/test split

data_path = './data'
gts_path = './annotations'
split_path = './splits'
os.makedirs(split_path, exist_ok=True)  # create directory if needed

split_test = True  # set to True and run when annotations are available

train_frac = 0.85

# get sorted list of files:
file_names = sorted(os.listdir(data_path))

# remove any non-JPEG files:
file_names = [f for f in file_names if '.jpg' in f]

np.random.shuffle(file_names)
total_files = len(file_names)

# split file names into train and test
file_names_train = file_names[:int(train_frac * total_files)]
file_names_test = file_names[int(train_frac * total_files):]

assert (len(file_names_train) + len(file_names_test)) == len(file_names)
assert len(np.intersect1d(file_names_train, file_names_test)) == 0

np.save(os.path.join(split_path, 'file_names_train.npy'), file_names_train)
np.save(os.path.join(split_path, 'file_names_test.npy'), file_names_test)

if split_test:
    with open(os.path.join(gts_path, 'annotations.json'), 'r') as f:
        gts = json.load(f)

    gts_train = {}
    gts_test = {}

    for fname in gts:
        if fname in file_names_train:
            gts_train[fname] = gts[fname]
        else:
            gts_test[fname] = gts[fname]

    with open(os.path.join(gts_path, 'annotations_train.json'), 'w') as f:
        json.dump(gts_train, f)

    with open(os.path.join(gts_path, 'annotations_test.json'), 'w') as f:
        json.dump(gts_test, f)