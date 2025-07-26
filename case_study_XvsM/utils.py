import numpy as np
import pandas as pd
import matplotlib

from utils.constants import ARCHIVE_NAMES as ARCHIVE_NAMES
from utils.constants import ITERATIONS as ITERATIONS
from utils.constants import CLASSIFIERS as CLASSIFIERS
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

def load_ori_data_XM(method):
    path = "/home/dmlab_a/Peiyu/aug_sf/data/"
    X_train = np.load(path + method + "/train/xtrain.npy")
    X_test = np.load(path + method + "/test/xtest.npy")
    X_val = np.load(path + method + "/val/xval.npy")
    y_train = np.load(path + method + "/train/ytrain.npy")
    y_test = np.load(path + method + "/test/ytest.npy")
    y_val = np.load(path + method + "/val/yval.npy")

    return X_train, y_train, X_val, y_val, X_test, y_test

def load_data_aug_xm(method):
    path_real = "/home/dmlab_a/Peiyu/aug_sf/data/XvsM/"
    path_syn = "/home/dmlab_a/Peiyu/aug_sf/data/XvsM/syn/"
    x_synthetic = np.load(path_syn + method + "/Solarflare/X_train_aug.npy")
    x_synthetic = x_synthetic.transpose(0, 2, 1)
    X_train_real = np.load(path_real + "train/xtrain.npy")
    y_train_real = np.load(path_real + "train/ytrain.npy")
    y_synthetic = y_train_real

    unique_classes, counts = np.unique(y_synthetic, return_counts=True)
    print("Unique classes in y_synthetic:", unique_classes)
    print("Counts of each class in y_synthetic:", counts)

    X_aug = np.vstack((X_train_real, x_synthetic))
    y_aug = np.hstack((y_train_real, y_synthetic))

    # Optional: Shuffle the augmented data
    permutation = np.random.permutation(len(X_aug))
    X_aug = X_aug[permutation]
    y_aug = y_aug[permutation]

    print("Shape of augmented X:", X_aug.shape)
    print("Shape of augmented y:", y_aug.shape)

    '''
    Shape of augmented X: (23968, 33, 60)
    Shape of augmented y: (23968,)
    '''

    X_test = np.load(path_real + "test/xtest.npy")
    X_val = np.load(path_real + "val/xval.npy")

    y_test = np.load(path_real + "test/ytest.npy")
    y_val = np.load(path_real + "val/yval.npy")

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

# this is to generate the results of classifying X and M classes
def generate_results_csv_xm(output_file_name, root_dir):
    for classifier_name in CLASSIFIERS:
        res = pd.DataFrame(
            columns=['classifier_name', 'dataset_name', 'archive_name', 'Accuracy', 'Precision', 'Recall', 'F1', 'TSS',
                     'HSS1', 'HSS', 'duration'])
        for archive_name in ARCHIVE_NAMES:
            for it in range(ITERATIONS):
                curr_archive_name = archive_name
                if it != 0:
                    curr_archive_name = curr_archive_name + '_itr_' + str(it)
                output_dir = root_dir + '/results_ori_xm/' + classifier_name + '/' \
                                 + curr_archive_name + '/' + 'solarflare' + '/' + 'df_metrics.csv'
                if not os.path.exists(output_dir):
                    continue

                df_metrics = pd.read_csv(output_dir)
                df_metrics['classifier_name'] = classifier_name
                df_metrics['dataset_name'] = 'solar_flare'
                df_metrics['archive_name'] = archive_name

                res = pd.concat((res, df_metrics), axis=0, sort=False)
        print(root_dir + "/results_ori_xm/" + classifier_name + "/" + output_file_name)

        res.to_csv(root_dir + "/results_ori_xm/" + classifier_name + "/" + output_file_name, index=False)
    return res

def generate_results_csv_xm_aug(output_file_name, root_dir):
    for classifier_name in CLASSIFIERS:
        res = pd.DataFrame(
            columns=['classifier_name', 'dataset_name', 'archive_name', 'Accuracy', 'Precision', 'Recall', 'F1', 'TSS',
                     'HSS1', 'HSS', 'duration'])
        for archive_name in ARCHIVE_NAMES:
            for it in range(ITERATIONS):
                curr_archive_name = archive_name
                if it != 0:
                    curr_archive_name = curr_archive_name + '_itr_' + str(it)
                output_dir = root_dir + '/results_aug_xm/' + classifier_name + '/' \
                                 + curr_archive_name + '/' + 'solarflare' + '/' + 'df_metrics.csv'
                if not os.path.exists(output_dir):
                    continue

                df_metrics = pd.read_csv(output_dir)
                df_metrics['classifier_name'] = classifier_name
                df_metrics['dataset_name'] = 'solar_flare'
                df_metrics['archive_name'] = archive_name

                res = pd.concat((res, df_metrics), axis=0, sort=False)
        print(root_dir + "/results_aug_xm/" + classifier_name + "/" + output_file_name)

        res.to_csv(root_dir + "/results_aug_xm/" + classifier_name + "/" + output_file_name, index=False)
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
