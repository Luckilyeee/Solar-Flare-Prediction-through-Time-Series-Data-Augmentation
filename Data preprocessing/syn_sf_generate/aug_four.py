import numpy as np
import pandas as pd
import os
import random
from sklearn.preprocessing import LabelEncoder
import utils.augmentation as aug

random.seed(42)
np.random.seed(42)

data_path = "aug_sf/data/four_class/"
results_path = "aug_sf/data/four_class/aug_train/"
for name in ['Solarflare']:
    try:
        print('Dataset:', name)


        X_train = np.load(data_path + "train/xtrain.npy")
        y_train = np.load(data_path + "train/ytrain.npy")

        X_train = X_train.transpose(0, 2, 1)
        print(X_train.shape, y_train.shape)

        print('permutation')
        X_aug = aug.permutation(X_train)
        if not os.path.exists(os.path.join(results_path, 'permutation', name)):
            os.makedirs(os.path.join(results_path, 'permutation', name), exist_ok=True)

        np.save(os.path.join(results_path, 'permutation', name, 'X_train_aug.npy'), \
                X_aug)

        print('SPAWNER')

        X_aug = aug.spawner(X_train, y_train, sigma=0.05, verbose=0)
        if not os.path.exists(os.path.join(results_path, 'SPAWNER', name)):
            os.makedirs(os.path.join(results_path, 'SPAWNER', name))
        np.save(os.path.join(results_path, 'SPAWNER', name, 'X_train_aug.npy'), \
                 X_aug)

    except Exception as ex:
        print(ex)










