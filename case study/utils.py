# for case study on four class classificatioin
def calculate_metrics(y_true, y_pred, duration, y_true_val=None, y_pred_val=None):
    cm = confusion_matrix(y_true, y_pred).ravel()
    res = pd.DataFrame(data=np.zeros((1, 5), dtype=float), index=[0],
                       columns=[ 'Accuracy', 'Precision', 'Recall', 'F1', 'duration'])
    res['Precision'] = precision_score(y_true, y_pred, average='macro')
    res['Accuracy'] = accuracy_score(y_true, y_pred)
    res['Recall'] = recall_score(y_true, y_pred, average='macro')
    res['F1'] = f1_score(y_true, y_pred, average='macro')
    res['duration'] = duration
    return res

def load_data_four(method):
    path = "aug_sf/data/"
    X_train = np.load(path + "/four_class/" + "/train/xtrain.npy")
    X_test = np.load(path + "/four_class/" + "/test/xtest.npy")
    y_train = np.load(path + "/four_class/" + "/train/ytrain.npy")
    y_test = np.load(path + "/four_class/" + "/test/ytest.npy")
    X_val = np.load(path + "/four_class/" + "/val/xval.npy")
    y_val = np.load(path + "/four_class/" + "/val/yval.npy")

    return X_train, y_train, X_val, y_val, X_test, y_test

def load_data_four_aug(method):
    path = "aug_sf/data/"
    X_train = np.load(path + "four_class/train_augmented/" + method + "/xtrain.npy")
    y_train = np.load(path + "four_class/train_augmented/" + method + "/ytrain.npy")

    X_test = np.load(path + "/four_class/" + "/test/xtest.npy")
    y_test = np.load(path + "/four_class/" + "/test/ytest.npy")
    X_val = np.load(path + "/four_class/" + "/val/xval.npy")
    y_val = np.load(path + "/four_class/" + "/val/yval.npy")

    return X_train, y_train, X_val, y_val, X_test, y_test


generate_results_csv_four(output_file_name, root_dir):
    for classifier_name in CLASSIFIERS:
        res = pd.DataFrame(
            columns=['classifier_name', 'archive_name', 'Accuracy', 'Precision', 'Recall', 'F1',
                     'HSS', 'duration'])
        for archive_name in ARCHIVE_NAMES:
            for it in range(ITERATIONS):
                curr_archive_name = archive_name
                if it != 0:
                    curr_archive_name = curr_archive_name + '_itr_' + str(it)
                output_dir = root_dir + '/results_four/' + classifier_name + '/' \
                                 + curr_archive_name + '/' + 'solarflare' + '/' + 'df_metrics.csv'
                if not os.path.exists(output_dir):
                    continue

                df_metrics = pd.read_csv(output_dir)
                df_metrics['classifier_name'] = classifier_name
                df_metrics['dataset_name'] = 'solar_flare'
                df_metrics['archive_name'] = archive_name

                res = pd.concat((res, df_metrics), axis=0, sort=False)
        print(root_dir + "/results_four/" + classifier_name + "/" + output_file_name)

        res.to_csv(root_dir + "/results_four/" + classifier_name + "/" + output_file_name, index=False)
    return res

def generate_results_csv_four_aug(output_file_name, root_dir):
    for classifier_name in CLASSIFIERS:
        res = pd.DataFrame(
            columns=['classifier_name', 'archive_name', 'Accuracy', 'Precision', 'Recall', 'F1',
                     'HSS', 'duration'])
        for archive_name in ARCHIVE_NAMES:
            for it in range(ITERATIONS):
                curr_archive_name = archive_name
                if it != 0:
                    curr_archive_name = curr_archive_name + '_itr_' + str(it)
                output_dir = root_dir + '/results_four_aug/' + classifier_name + '/' \
                                 + curr_archive_name + '/' + 'solarflare' + '/' + 'df_metrics.csv'
                if not os.path.exists(output_dir):
                    continue

                df_metrics = pd.read_csv(output_dir)
                df_metrics['classifier_name'] = classifier_name
                df_metrics['dataset_name'] = 'solar_flare'
                df_metrics['archive_name'] = archive_name

                res = pd.concat((res, df_metrics), axis=0, sort=False)
        print(root_dir + "/results_four_aug/" + classifier_name + "/" + output_file_name)

        res.to_csv(root_dir + "/results_four_aug/" + classifier_name + "/" + output_file_name, index=False)
    return res
