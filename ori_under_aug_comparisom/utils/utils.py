import numpy as np
import pandas as pd
import matplotlib

from utils.constants import ARCHIVE_NAMES as ARCHIVE_NAMES
from utils.constants import ITERATIONS as ITERATIONS
from utils.constants import CLASSIFIERS as CLASSIFIERS
from utils.constants import sizes
from sklearn.metrics import confusion_matrix

import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

matplotlib.use('agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore", message=".*size changed.*", category=RuntimeWarning)


def load_ori_data(method):
    path = "aug_sf/data/"
    X_train = np.load(path + method + "/train/xtrain.npy")
    X_test = np.load(path + method + "/test/xtest.npy")
    X_val = np.load(path + method + "/val/xval.npy")
    y_train = np.load(path + method + "/train/ytrain.npy")
    y_test = np.load(path + method + "/test/ytest.npy")
    y_val = np.load(path + method + "/val/yval.npy")

    return X_train, y_train, X_val, y_val, X_test, y_test

def load_under_data(method):
    path = "aug_sf/data/"
    X_train = np.load(path + method + "/xtrain.npy")
    y_train = np.load(path + method + "/ytrain.npy")
    X_test = np.load(path  +"Ori_data/test/xtest.npy")
    y_test = np.load(path + "Ori_data/test/ytest.npy")
    X_val = np.load(path + "Ori_data/val/xval.npy")
    y_val = np.load(path + "Ori_data/val/yval.npy")

    return X_train, y_train, X_val, y_val, X_test, y_test


def load_stra_data(method):
    path = "aug_sf/data/"
    # training dataset is from the aug_sf/data/train_under folder
    X_train = np.load(path + method + "/xtrain_subsampled.npy")
    y_train = np.load(path + method + "/ytrain_subsampled.npy")
    # test and valitation datasets are from the aug_sf/data/Ori_data/ folder
    X_test = np.load(path  +"Ori_data/test/xtest.npy")
    y_test = np.load(path + "Ori_data/test/ytest.npy")
    X_val = np.load(path + "Ori_data/val/xval.npy")
    y_val = np.load(path + "Ori_data/val/yval.npy")

    return X_train, y_train, X_val, y_val, X_test, y_test

def load_data_aug_onlyx(method):
    print("loading syn data for method:", method)
    # discriminative_guided_warp required all classes to generate synthetic data
    if method == 'discriminative_guided_warp':
        path = "aug_sf/data/"
        x_synthetic = np.load(
            "aug_sf/data/syn/" + method + "/Solarflare/X_train_aug.npy")
        X_train_real = np.load(path + "stra_sampling/xtrain_subsampled.npy")
        print(x_synthetic.shape, X_train_real.shape)
        y_train_real = np.load(path + "stra_sampling/ytrain_subsampled.npy")
        y_synthetic = y_train_real

        x_synthetic_class_1 = x_synthetic[y_synthetic == 1]
        y_synthetic_class_1 = y_synthetic[y_synthetic == 1]

        x_synthetic_class_1 = x_synthetic_class_1.transpose(0, 2, 1)

        X_aug = np.vstack((X_train_real, x_synthetic_class_1))
        y_aug = np.hstack((y_train_real, y_synthetic_class_1))
    else:
        path = "aug_sf/data/"
        x_synthetic = np.load(
            "aug_sf/data/syn/" + method + "/Solarflare/X_train_aug.npy")
        y_synthetic = np.ones(x_synthetic.shape[0])
        X_train_real = np.load(path + "stra_sampling/xtrain_subsampled.npy")
        print(x_synthetic.shape, X_train_real.shape)
        y_train_real = np.load(path + "stra_sampling/ytrain_subsampled.npy")

        x_synthetic = x_synthetic.transpose(0, 2, 1)

        X_aug = np.vstack((X_train_real, x_synthetic))
        y_aug = np.hstack((y_train_real, y_synthetic))



    # Optional: Shuffle the augmented data
    permutation = np.random.permutation(len(X_aug))
    X_aug = X_aug[permutation]
    y_aug = y_aug[permutation]

    unique, counts = np.unique(y_aug, return_counts=True)
    print("Label counts in y_aug:", dict(zip(unique, counts)))

    print("Shape of augmented X:", X_aug.shape)
    print("Shape of augmented y:", y_aug.shape)

    X_test = np.load(path + '/Ori_data' + "/test/xtest.npy")
    X_val = np.load(path + '/Ori_data' + "/val/xval.npy")

    y_test = np.load(path + '/Ori_data' + "/test/ytest.npy")
    y_val = np.load(path + '/Ori_data' + "/val/yval.npy")

    return X_aug, y_aug, X_val, y_val, X_test, y_test



def calculate_metrics(y_true, y_pred, duration, y_true_val=None, y_pred_val=None):
    TN, FP, FN, TP = confusion_matrix(y_true, y_pred).ravel()
    tss = (TP / (TP + FN)) - (FP / (FP + TN))
    hss1 = ((TP + TN) - (FP + TN)) / (TP + FN)
    hss2 = 2 * ((TP * TN) - (FN * FP)) / ((TP + FN) * (FN + TN) + (TP + FP) * (FP + TN))

    res = pd.DataFrame(data=np.zeros((1, 8), dtype=float), index=[0],
                       columns=[ 'Accuracy', 'Precision', 'Recall', 'F1', 'TSS', 'HSS1', 'HSS', 'duration'])
    res['Precision'] = precision_score(y_true, y_pred, average='macro')
    res['Accuracy'] = accuracy_score(y_true, y_pred)
    res['Recall'] = recall_score(y_true, y_pred, average='macro')
    res['F1'] = f1_score(y_true, y_pred, average='macro')
    res['TSS'] = tss
    res['HSS1'] = hss1
    res['HSS'] = hss2
    res['duration'] = duration
    return res

def create_directory(directory_path):
    if os.path.exists(directory_path):
        return None
    else:
        try:
            os.makedirs(directory_path)
        except:
            # in case another machine created the path meanwhile !:(
            return None
        return directory_path


# this is to generate the results of using all of the original dataset
def generate_results_csv_0(output_file_name, root_dir):
    for classifier_name in CLASSIFIERS:
        res = pd.DataFrame(
            columns=['classifier_name', 'dataset_name', 'archive_name', 'accuracy', 'precision', 'recall', 'f1', 'tss',
                     'hss1', 'hss2', 'duration'])
        for archive_name in ARCHIVE_NAMES:
            for it in range(ITERATIONS):
                curr_archive_name = archive_name
                if it != 0:
                    curr_archive_name = curr_archive_name + '_itr_' + str(it)
                output_dir = root_dir + '/results_ori/' + classifier_name + '/' \
                                 + curr_archive_name + '/' + 'solarflare' + '/' + 'df_metrics.csv'
                if not os.path.exists(output_dir):
                    continue

                df_metrics = pd.read_csv(output_dir)
                df_metrics['classifier_name'] = classifier_name
                df_metrics['dataset_name'] = 'solar_flare'
                df_metrics['archive_name'] = archive_name

                res = pd.concat((res, df_metrics), axis=0, sort=False)
        print(root_dir + "/results_ori/" + classifier_name + "/" + output_file_name)

        res.to_csv(root_dir + "/results_ori/" + classifier_name + "/" + output_file_name, index=False)
    return res

def generate_results_csv_1(output_file_name, root_dir):
    for classifier_name in CLASSIFIERS:
        res = pd.DataFrame(
            columns=['classifier_name', 'dataset_name', 'archive_name', 'accuracy', 'precision', 'recall', 'f1', 'tss',
                     'hss1', 'hss2', 'duration'])
        for archive_name in ARCHIVE_NAMES:
            for it in range(ITERATIONS):
                curr_archive_name = archive_name
                if it != 0:
                    curr_archive_name = curr_archive_name + '_itr_' + str(it)
                output_dir = root_dir + '/results_under/' + classifier_name + '/' \
                                 + curr_archive_name + '/' + 'solarflare' + '/' + 'df_metrics.csv'
                if not os.path.exists(output_dir):
                    continue

                df_metrics = pd.read_csv(output_dir)
                df_metrics['classifier_name'] = classifier_name
                df_metrics['dataset_name'] = 'solar_flare'
                df_metrics['archive_name'] = archive_name

                res = pd.concat((res, df_metrics), axis=0, sort=False)
        print(root_dir + "/results_under/" + classifier_name + "/" + output_file_name)

        res.to_csv(root_dir + "/results_under/" + classifier_name + "/" + output_file_name, index=False)
    return res

def generate_results_csv_2(output_file_name, root_dir):
    for classifier_name in CLASSIFIERS:
        res = pd.DataFrame(
            columns=['classifier_name', 'dataset_name', 'archive_name', 'accuracy', 'precision', 'recall', 'f1', 'tss',
                     'hss1', 'hss2', 'duration'])
        for archive_name in ARCHIVE_NAMES:
            for it in range(ITERATIONS):
                curr_archive_name = archive_name
                if it != 0:
                    curr_archive_name = curr_archive_name + '_itr_' + str(it)
                output_dir = root_dir + '/results_stratify/' + classifier_name + '/' \
                                 + curr_archive_name + '/' + 'solarflare' + '/' + 'df_metrics.csv'
                if not os.path.exists(output_dir):
                    continue

                df_metrics = pd.read_csv(output_dir)
                df_metrics['classifier_name'] = classifier_name
                df_metrics['dataset_name'] = 'solar_flare'
                df_metrics['archive_name'] = archive_name

                res = pd.concat((res, df_metrics), axis=0, sort=False)
        print(root_dir + "/results_stratify/" + classifier_name + "/" + output_file_name)

        res.to_csv(root_dir + "/results_stratify/" + classifier_name + "/" + output_file_name, index=False)
    return res

def generate_results_csv_augstra_onlyx(output_file_name, root_dir):
    for classifier_name in CLASSIFIERS:
        res = pd.DataFrame(
            columns=['classifier_name', 'dataset_name', 'archive_name', 'Accuracy', 'Precision', 'Recall', 'F1', 'TSS',
                     'HSS1', 'HSS', 'duration'])
        for archive_name in ARCHIVE_NAMES:
            for it in range(ITERATIONS):
                curr_archive_name = archive_name
                if it != 0:
                    curr_archive_name = curr_archive_name + '_itr_' + str(it)
                output_dir = root_dir + '/results_aug_stra_x/' + classifier_name + '/' \
                                 + curr_archive_name + '/' + 'solarflare' + '/' + 'df_metrics.csv'
                if not os.path.exists(output_dir):
                    continue

                df_metrics = pd.read_csv(output_dir)
                df_metrics['classifier_name'] = classifier_name
                df_metrics['dataset_name'] = 'solar_flare'
                df_metrics['archive_name'] = archive_name

                res = pd.concat((res, df_metrics), axis=0, sort=False)
        print(root_dir + "/results_aug_stra_x/" + classifier_name + "/" + output_file_name)

        res.to_csv(root_dir + "/results_aug_stra_x/" + classifier_name + "/" + output_file_name, index=False)
    return res

def plot_epochs_metric(hist, file_name, metric='loss'):
    plt.figure()
    plt.plot(hist.history[metric])
    plt.plot(hist.history['val_' + metric])
    plt.title('model ' + metric)
    plt.ylabel(metric, fontsize='large')
    plt.xlabel('epoch', fontsize='large')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig(file_name, bbox_inches='tight')
    plt.close()

def save_logs(output_directory, hist, y_pred, y_true, duration, lr=True, y_true_val=None, y_pred_val=None):
    hist_df = pd.DataFrame(hist.history)
    hist_df.to_csv(output_directory + 'history.csv', index=False)

    df_metrics = calculate_metrics(y_true, y_pred, duration, y_true_val, y_pred_val)
    df_metrics.to_csv(output_directory + 'df_metrics.csv', index=False)

    index_best_model = hist_df['loss'].idxmin()
    row_best_model = hist_df.loc[index_best_model]

    df_best_model = pd.DataFrame(data=np.zeros((1, 6), dtype=float), index=[0],
                                 columns=['best_model_train_loss', 'best_model_val_loss', 'best_model_train_acc',
                                          'best_model_val_acc', 'best_model_learning_rate', 'best_model_nb_epoch'])

    df_best_model['best_model_train_loss'] = row_best_model['loss']
    df_best_model['best_model_val_loss'] = row_best_model['val_loss']
    df_best_model['best_model_train_acc'] = row_best_model['accuracy']
    df_best_model['best_model_val_acc'] = row_best_model['val_accuracy']
    if lr == True:
        df_best_model['best_model_learning_rate'] = row_best_model['lr']
    df_best_model['best_model_nb_epoch'] = index_best_model

    df_best_model.to_csv(output_directory + 'df_best_model.csv', index=False)

    # plot losses
    plot_epochs_metric(hist, output_directory + 'epochs_loss.png')

    return df_metrics

def save_test_duration(file_name, test_duration):
    res = pd.DataFrame(data=np.zeros((1, 1), dtype=float), index=[0],
                       columns=['test_duration'])
    res['test_duration'] = test_duration
    res.to_csv(file_name, index=False)

def save_logs(output_directory, hist, y_pred, y_true, duration, lr=True, y_true_val=None, y_pred_val=None):
    hist_df = pd.DataFrame(hist.history)
    hist_df.to_csv(output_directory + 'history.csv', index=False)

    df_metrics = calculate_metrics(y_true, y_pred, duration, y_true_val, y_pred_val)
    df_metrics.to_csv(output_directory + 'df_metrics.csv', index=False)

    index_best_model = hist_df['loss'].idxmin()
    row_best_model = hist_df.loc[index_best_model]

    df_best_model = pd.DataFrame(data=np.zeros((1, 6), dtype=float), index=[0],
                                 columns=['best_model_train_loss', 'best_model_val_loss', 'best_model_train_acc',
                                          'best_model_val_acc', 'best_model_learning_rate', 'best_model_nb_epoch'])

    df_best_model['best_model_train_loss'] = row_best_model['loss']
    df_best_model['best_model_val_loss'] = row_best_model['val_loss']
    df_best_model['best_model_train_acc'] = row_best_model['accuracy']
    df_best_model['best_model_val_acc'] = row_best_model['val_accuracy']
    if lr == True:
        df_best_model['best_model_learning_rate'] = row_best_model['lr']
    df_best_model['best_model_nb_epoch'] = index_best_model

    df_best_model.to_csv(output_directory + 'df_best_model.csv', index=False)
    # plot losses
    plot_epochs_metric(hist, output_directory + 'epochs_loss.png')

    return df_metrics

