import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score
import os

plt.rcParams["pdf.use14corefonts"] = True

path_xm_ori = '/aug_sf/results_ori_xm/'
path_xm_aug = '/aug_sf/results_aug_xm/'

path_figs = '/aug_sf/figs_final/xm_vs_aug/'

aug_methods = ['ww', 'ws', 'wdba', 'tw', 'spawner', 'rgw', 'mw', 'dgw', 'permutation', 'scaling', 'jitter', 'rotation']
labels = ['WW', 'WS', 'WDBA', 'TW', 'SPAWNER', 'RGW', 'MW', 'DGW', 'PM', 'SC', 'JIT', 'ROT']

def compute_gini_scores(model, method, aug_method=None):
    y_test = np.load("/home/dmlab_a/Peiyu/aug_sf/data/XvsM/test/ytest.npy")
    labels, counts = np.unique(y_test, return_counts=True)
    # for label, count in zip(labels, counts):
    #     print(f"Label: {label}, Count: {count}")
    base_dir = f"/home/dmlab_a/Peiyu/aug_sf/{method}/{model}/"

    if method == "results_ori_xm":
        run_dirs = ["SF_real_ori_XvsM", "SF_real_ori_XvsM_itr_1", "SF_real_ori_XvsM_itr_2", "SF_real_ori_XvsM_itr_3", "SF_real_ori_XvsM_itr_4",
                    "SF_real_ori_XvsM_itr_5", "SF_real_ori_XvsM_itr_6", "SF_real_ori_XvsM_itr_7", "SF_real_ori_XvsM_itr_8", "SF_real_ori_XvsM_itr_9"]
    elif method == "results_aug_xm":
        run_dirs = [
            f"SF_aug_{aug_method}",
            f"SF_aug_{aug_method}_itr_1",
            f"SF_aug_{aug_method}_itr_2",
            f"SF_aug_{aug_method}_itr_3",
            f"SF_aug_{aug_method}_itr_4",
            f"SF_aug_{aug_method}_itr_5",
            f"SF_aug_{aug_method}_itr_6",
            f"SF_aug_{aug_method}_itr_7",
            f"SF_aug_{aug_method}_itr_8",
            f"SF_aug_{aug_method}_itr_9"

        ]
    else:
        raise ValueError("Unknown method type")

    gini_scores = []
    for run in run_dirs:
        pred_path = os.path.join(base_dir, run, "solarflare/y_pred.npy")
        y_pred = np.load(pred_path)
        if y_pred.ndim == 2 and y_pred.shape[1] == 2:
            y_pred = y_pred[:, 1]
        auc = roc_auc_score(y_test, y_pred)
        gini_scores.append(2 * auc - 1)
    return gini_scores

def plot(model, metric):
    print(f"Processing: {model}, Metric: {metric}")
    df_ori = pd.read_csv(path_xm_ori + model + "/res.csv")
    df_aug = pd.read_csv(path_xm_aug + model + "/res.csv")
    aug_list = df_aug['archive_name'].unique()

    if metric != "Gini":
        # Get ori mean metric
        stratified_df = df_ori[df_ori['archive_name'] == 'SF_real_ori_XvsM']
        mean_val = stratified_df[metric].mean()

        # Prepare augmentation data
        data = [df_aug[df_aug['archive_name'] == aug][metric] for aug in aug_list]

    else:
        # Compute Gini scores for original XM baseline
        ori_scores = compute_gini_scores(model, 'results_ori_xm')
        mean_val = np.mean(ori_scores)

        # Compute Gini scores for each augmentation
        data = [compute_gini_scores(model, 'results_aug_xm', aug) for aug in aug_methods]

    # Sort by mean
    sorted_data, sorted_labels = zip(*sorted(zip(data, labels), key=lambda x: np.mean(x[0])))

    fig, ax = plt.subplots(figsize=(5, 4))
    bplot = ax.boxplot(sorted_data, vert=True, patch_artist=True, labels=sorted_labels)

    colors = ['lightblue', 'lightgreen', 'yellow', 'orange', 'tan', 'salmon', 'plum', 'cyan', 'magenta', 'lime', 'gold', 'coral']
    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)

    ax.axhline(mean_val, color='red', linestyle='--', label='Original baseline')
    ax.set_xticklabels(sorted_labels, rotation=90, fontsize=12, fontweight='bold')
    ax.set_xlabel("Augmentation Methods", fontsize=14, fontweight='bold')
    ax.set_ylabel(metric, fontsize=14, fontweight='bold')
    ax.legend()
    ax.yaxis.grid(True)

    plt.tight_layout()
    plt.savefig(path_figs + f"{model}_{metric}_comparison.pdf", format='pdf', dpi=300)
    plt.show()

models = ['fcn', 'lstm_fcn', 'resnet']
metrics = ['Recall', 'TSS', 'HSS', 'Gini']  # Add Gini here
for model in models:
    for metric in metrics:
        plot(model, metric)
