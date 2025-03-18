import numpy as np
import os
import pandas as pd
import random
from sklearn.preprocessing import LabelEncoder
import utils.augmentation as aug


random.seed(42)
np.random.seed(42)

results_path = "aug_sf/data/aug_after_stratify/ratio_study"
for name in ['Solarflare']:
    try:
        print('Dataset:', name)
        X_train = np.load("aug_sf/data/stra_sampling/xtrain_subsampled.npy")
        y_train = np.load("aug_sf/data/stra_sampling/ytrain_subsampled.npy")

        unique_classes = np.unique(y_train)
        x_train_classes = {cls: X_train[y_train == cls] for cls in unique_classes}
        print("Unique classes in y_train:", unique_classes)

        x_train_1 = x_train_classes[1]
        print("Shape of x_train_1:", x_train_1.shape)
        y_train_1 = np.ones(x_train_1.shape[0])

        x_train_1 = x_train_1.transpose(0, 2, 1)

        print('permutation')
        paras = [2, 3, 4, 6, 7]
        for para in paras:
            print(para)
            X_aug = aug.permutation(x_train_1, max_segments=para, seg_mode="equal")
            if not os.path.exists(os.path.join(results_path, 'permutation', name, str(para))):
                os.makedirs(os.path.join(results_path, 'permutation', name, str(para)), exist_ok=True)

            np.save(os.path.join(results_path, 'permutation', name, str(para), 'X_train_aug.npy'), \
                    X_aug)


        print('SPAWNER')

        # sigmas = [0.04, 0.045, 0.055, 0.06, 0.065]
        sigmas = [0.055]
        for sigma in sigmas:

            X_aug = aug.spawner(x_train_1, y_train_1, sigma=sigma, verbose=0)
            if not os.path.exists(os.path.join(results_path, 'SPAWNER', name, str(sigma))):
                os.makedirs(os.path.join(results_path, 'SPAWNER', name, str(sigma)), exist_ok=True)

            np.save(os.path.join(results_path, 'SPAWNER', name, str(sigma), 'X_train_aug.npy'), \
                    X_aug)

    except Exception as ex:
        print(ex)
    
    


    
    
    
    
    
    
