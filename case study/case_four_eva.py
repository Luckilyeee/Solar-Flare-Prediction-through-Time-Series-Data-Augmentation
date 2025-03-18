import numpy as np
from sklearn.metrics import classification_report, accuracy_score

methods = ['fcn', 'resnet']
path0 = "results_four/"
def res(method):
    print(method)
    y_true = np.load(
        path0 + method + "/SF_four_class/solarflare/y_true.npy")
    y_preds = []
    for i in range(10):
        if i == 0:
            path = path0 + method + "/SF_four_class/solarflare/y_pred.npy"
        else:
            path = path0 + method + "/SF_four_class_itr_" + str(
                i) + "/solarflare/y_pred.npy"

        y_pred = np.load(path)
        y_preds.append(y_pred.argmax(axis=1))

    # Initialize a dictionary to store metrics for each class
    class_metrics = {0: {'F1': [], 'Precision': [], 'Recall': []},
                     1: {'F1': [], 'Precision': [], 'Recall': []},
                     2: {'F1': [], 'Precision': [], 'Recall': []},
                     3: {'F1': [], 'Precision': [], 'Recall': []},
                     }

    accuracy_scores = []

    # Loop through each run and collect metrics for each class
    for y_pred in y_preds:
        report = classification_report(y_true, y_pred, digits=4, output_dict=True)

        accuracy = accuracy_score(y_true, y_pred)
        accuracy_scores.append(accuracy)

        for class_label in class_metrics.keys():
            class_metrics[class_label]['F1'].append(report[str(class_label)]['f1-score'])
            class_metrics[class_label]['Precision'].append(report[str(class_label)]['precision'])
            class_metrics[class_label]['Recall'].append(report[str(class_label)]['recall'])

    # Calculate mean and standard deviation for each class
    mean_std_per_class = {}
    # print(accuracy_scores)
    for class_label, metrics in class_metrics.items():
        mean_std_per_class[class_label] = {
            'mean_f1': np.mean(metrics['F1']),
            'std_f1': np.std(metrics['F1']),
            'mean_precision': np.mean(metrics['Precision']),
            'std_precision': np.std(metrics['Precision']),
            'mean_recall': np.mean(metrics['Recall']),
            'std_recall': np.std(metrics['Recall']),
        }

    # Calculate mean and standard deviation for accuracy
    mean_accuracy = np.mean(accuracy_scores)
    std_accuracy = np.std(accuracy_scores)
    class_labels = {0: "X", 1: "M", 2: "C/B", 3: "Q"}

    print(f"Total Accuracy Metrics:")
    print(f"Mean Accuracy: {mean_accuracy:.4f} $\pm$ {std_accuracy:.4f}")

    # Print or use the mean and standard deviation for each class and accuracy as needed
    for class_label, metrics in mean_std_per_class.items():
        print(f"Class {class_labels[class_label]} Metrics:")
        print(f" F1-score: {metrics['mean_f1']:.4f} $\pm$ {metrics['std_f1']:.4f}")
        print(f" Precision: {metrics['mean_precision']:.4f} $\pm$ {metrics['std_precision']:.4f}")
        print(f" Recall: {metrics['mean_recall']:.4f} $\pm$ {metrics['std_recall']:.4f}")
        print()

for method in methods:
    res(method)
