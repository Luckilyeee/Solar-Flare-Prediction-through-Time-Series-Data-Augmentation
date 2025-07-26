from sklearn.model_selection import train_test_split
from sklearn.utils import resample
import pandas as pd
from collections import Counter
import numpy as np
import os

data_path = "Solarflare/"
save_path = "aug_sf/data/XvsM/"
def load_ori_data(method):
    inputs = pd.read_pickle(data_path + method + "/inputs.pck")
    labels = pd.read_pickle(data_path + method + "/labels.pck")

    inputs = np.array(inputs)
    labels = np.array(labels)
    inputs = inputs.transpose(0, 2, 1)
    print("the original dataset length")
    print(inputs.shape, labels.shape)

    unique_labels, counts = np.unique(labels, return_counts=True)
    print("Original Label Counts:", dict(zip(unique_labels, counts)))

    # Save data with label 0 to X.npy and label 1 to M.npy
    x_data = inputs[labels == 0]
    m_data = inputs[labels == 1]
    np.save(os.path.join(save_path, "X.npy"), x_data)
    np.save(os.path.join(save_path, "M.npy"), m_data)

    # Combine x and m class samples as original inputs and labels
    x_data = inputs[labels == 0]
    m_data = inputs[labels == 1]
    x_labels = np.ones(len(x_data), dtype=int)
    m_labels = np.zeros(len(m_data), dtype=int)

    all_inputs = np.concatenate([x_data, m_data], axis=0)
    all_labels = np.concatenate([x_labels, m_labels], axis=0)

    # Split into train, test, val
    X_train, X_test, y_train, y_test = train_test_split(
        all_inputs, all_labels, test_size=0.25, stratify=all_labels, random_state=42
    )
    X_train_mean = X_train.mean()
    X_train_std = X_train.std()
    X_train = (X_train - X_train_mean) / X_train_std
    X_test = (X_test - X_train_mean) / X_train_std

    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.25, stratify=y_train, random_state=42
    )

    print("Train shape:", X_train.shape, y_train.shape)
    print("Train class counts:", dict(zip(*np.unique(y_train, return_counts=True))))
    print("Test shape:", X_test.shape, y_test.shape)
    print("Test class counts:", dict(zip(*np.unique(y_test, return_counts=True))))
    print("Val shape:", X_val.shape, y_val.shape)
    print("Val class counts:", dict(zip(*np.unique(y_val, return_counts=True))))

    np.save(save_path + "/train/xtrain.npy", X_train)
    np.save(save_path + "/test/xtest.npy", X_test)
    np.save(save_path + "/train/ytrain.npy", y_train)
    np.save(save_path + "/test/ytest.npy", y_test)
    np.save(save_path + "/val/xval.npy", X_val)
    np.save(save_path + "/val/yval.npy", y_val)
    print("DONE")


method = 'Ori_data'
load_ori_data(method)
