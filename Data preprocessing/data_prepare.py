from sklearn.model_selection import train_test_split
from sklearn.utils import resample
import pandas as pd
from collections import Counter
import numpy as np
import os

data_path = "Solarflare/"
save_path = "aug_sf/data/"
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

    labels = np.where(labels == 0, 1, labels)
    labels = np.where(((labels == 2) | (labels == 3) | (labels == 4)), 0, labels)
    unique_labels, counts = np.unique(labels, return_counts=True)
    print("Binary Label Counts:", dict(zip(unique_labels, counts)))

    X_train, X_test, y_train, y_test = train_test_split(inputs, labels, test_size=0.25, stratify=labels, random_state=42)
    X_train_mean = X_train.mean()
    X_train_std = X_train.std()
    X_train = (X_train - X_train_mean) / X_train_std
    X_test = (X_test - X_train_mean) / X_train_std

    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.25, stratify=y_train, random_state=42)

    np.save(save_path + method + "/train/xtrain.npy", X_train)
    np.save(save_path + method + "/test/xtest.npy", X_test)
    np.save(save_path + method + "/train/ytrain.npy", y_train)
    np.save(save_path + method + "/test/ytest.npy", y_test)
    np.save(save_path + method + "/val/xval.npy", X_val)
    np.save(save_path + method + "/val/yval.npy", y_val)
    print("DONE")

def load_train_test(method):
    X_train = np.load(save_path + method + "/train/xtrain.npy")
    X_test = np.load(save_path + method + "/test/xtest.npy")
    X_val = np.load(save_path + method + "/val/xval.npy")
    y_train = np.load(save_path + method + "/train/ytrain.npy")
    y_test = np.load(save_path + method + "/test/ytest.npy")
    y_val = np.load(save_path + method + "/val/yval.npy")

    print("data distribution")
    print(X_train.shape, X_val.shape, X_test.shape, y_train.shape, y_val.shape, y_test.shape)
    print("train ---- class distribution")
    unique_labels, counts = np.unique(y_train, return_counts=True)
    print("Binary Label Counts:", dict(zip(unique_labels, counts)))
    print("validation ---- class distribution")
    unique_labels, counts = np.unique(y_val, return_counts=True)
    print("Binary Label Counts:", dict(zip(unique_labels, counts)))
    print("test ---- class distribution")
    unique_labels, counts = np.unique(y_test, return_counts=True)
    print("Binary Label Counts:", dict(zip(unique_labels, counts)))
    

def undersampling(method):
    X_train = np.load(save_path + method + "/train/xtrain.npy")
    y_train = np.load(save_path + method + "/train/ytrain.npy")


    unique_labels, counts = np.unique(y_train, return_counts=True)
    print("Binary Label Counts of training before undersampling:", dict(zip(unique_labels, counts)))

    # Separate majority and minority classes
    X_majority = X_train[y_train == 0]
    y_majority = y_train[y_train == 0]
    X_minority = X_train[y_train == 1]
    y_minority = y_train[y_train == 1]

    # Undersample majority class
    X_majority_undersampled, y_majority_undersampled = resample(X_majority,
                                                                y_majority,
                                                                replace=False,  # sample without replacement
                                                                n_samples=len(X_minority),  # match minority class
                                                                random_state=42)  # reproducible results

    # Combine minority class with undersampled majority class
    X_train_balanced = np.vstack((X_majority_undersampled, X_minority))
    y_train_balanced = np.hstack((y_majority_undersampled, y_minority))

    # Shuffle the balanced dataset
    permutation = np.random.permutation(len(X_train_balanced))
    X_train_balanced = X_train_balanced[permutation]
    y_train_balanced = y_train_balanced[permutation]

    print("Original dataset shape:", X_train.shape, y_train.shape)
    print("Balanced dataset shape:", X_train_balanced.shape, y_train_balanced.shape)


    unique_labels, counts = np.unique(y_train_balanced, return_counts=True)
    print("Binary Label Counts of training after undersampling:", dict(zip(unique_labels, counts)))

    np.save(save_path + "/train_under/xtrain.npy", X_train_balanced)
    np.save(save_path + "/train_under/ytrain.npy", y_train_balanced)


def stratify(method):

    X_train = np.load(save_path + method + "/train/xtrain.npy")
    y_train = np.load(save_path + method + "/train/ytrain.npy")


    # Count the number of samples in each class
    unique_classes, class_counts = np.unique(y_train, return_counts=True)
    print("Class distribution:", dict(zip(unique_classes, class_counts)))


    # Define the fraction of the dataset to keep (e.g., 10%)
    subsample_fraction = 0.1

    # Perform stratified sampling
    X_train_subsampled, _, y_train_subsampled, _ = train_test_split(
        X_train, y_train,
        train_size=subsample_fraction,
        stratify=y_train,
        random_state=42
    )

    # Check the class distribution in the subsampled dataset
    unique_classes, class_counts = np.unique(y_train_subsampled, return_counts=True)
    print("Subsampled class distribution:", dict(zip(unique_classes, class_counts)))
    
    # Save the subsampled data (optional)
    np.save(save_path + 'stra_sampling' + "/xtrain_subsampled.npy", X_train_subsampled)
    np.save(save_path + 'stra_sampling' + "/ytrain_subsampled.npy", y_train_subsampled)

method = 'Ori_data'
load_ori_data(method)
load_train_test(method)
undersampling(method)
stratify(method)





