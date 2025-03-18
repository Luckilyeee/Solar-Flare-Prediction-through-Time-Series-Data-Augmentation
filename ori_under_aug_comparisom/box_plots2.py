
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
warnings.resetwarnings()

plt.rcParams["pdf.use14corefonts"] = True

# model = 'resnet'
path_stra= 'results_stratify/'
path_stra_aug = 'results_aug_stra_x/'
path_figs = 'figs/stra_vs_aug_x/'


def plot(model, metric):
    print(model, metric)
    df_stra = pd.read_csv(path_stra + model + '/' + "res.csv")
    df_aug = pd.read_csv(path_stra_aug + model + '/' + "res.csv")

    aug_methods = df_aug['archive_name'].unique()

    # get the ori stratified results
    filtered_df = df_stra[df_stra['archive_name'] == 'SF_real_stra']

    # Calculate the mean value of each metric
    mean_metrics = filtered_df.groupby('archive_name').agg({
        'Accuracy': 'mean',
        'Precision': 'mean',
        'Recall': 'mean',
        'F1': 'mean',
        'TSS': 'mean',
        'HSS': 'mean'
    }).reset_index()

    median_metrics = filtered_df.groupby('archive_name').agg({
        'Accuracy': 'median',
        'Precision': 'median',
        'Recall': 'median',
        'F1': 'median',
        'TSS': 'median',
        'HSS': 'median'
    }).reset_index()


    ww = df_aug[df_aug['archive_name'] == aug_methods[0]][metric]
    ws = df_aug[df_aug['archive_name'] == aug_methods[1]][metric]
    wdba = df_aug[df_aug['archive_name'] == aug_methods[2]][metric]
    tw = df_aug[df_aug['archive_name'] == aug_methods[3]][metric]
    spawner = df_aug[df_aug['archive_name'] == aug_methods[4]][metric]
    rgw = df_aug[df_aug['archive_name'] == aug_methods[5]][metric]
    mw = df_aug[df_aug['archive_name'] == aug_methods[6]][metric]
    dgw = df_aug[df_aug['archive_name'] == aug_methods[7]][metric]
    permutation = df_aug[df_aug['archive_name'] == aug_methods[8]][metric]
    scaling = df_aug[df_aug['archive_name'] == aug_methods[9]][metric]
    jitter = df_aug[df_aug['archive_name'] == aug_methods[10]][metric]
    rotation = df_aug[df_aug['archive_name'] == aug_methods[11]][metric]


    data = [ww, ws, wdba, tw, spawner, rgw, mw, dgw, permutation, scaling, jitter, rotation]
    labels = ['WW', 'WS', 'WDBA', 'TW', 'SPAWNER', 'RGW', 'MW', 'DGW', 'PM', 'SC', 'JIT', 'ROT']

    sorted_data, sorted_labels = zip(*sorted(zip(data, labels), key=lambda x: np.mean(x[0])))

    fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(5, 4))

    # Rectangular box plot with sorted data and labels
    bplot1 = ax1.boxplot(sorted_data,
                         vert=True,  # vertical box alignment
                         patch_artist=True,  # fill with color
                         tick_labels=sorted_labels)  # will be used to label x-ticks

    # Tilt x-labels at a specified angle (e.g., 45 degrees)
    ax1.set_xticklabels(sorted_labels, rotation=90, fontsize=12, fontweight='bold')

    # Fill with colors
    colors = ['lightblue', 'lightgreen', 'yellow', 'orange', 'tan', 'salmon', 'plum', 'cyan', 'magenta', 'lime', 'gold', 'coral']
    # Set facecolor of the boxes
    for patch, color in zip(bplot1['boxes'], colors):
        patch.set_facecolor(color)

    # Adding horizontal grid lines
    ax1.yaxis.grid(True)

    # Add a horizontal line for the mean value from mean_metrics
    ax1.axhline(mean_metrics[metric].values[0], color='red', linestyle='--', label='Original Stratified Sampling (baseline)')

    # Add legend
    ax1.legend(prop={'weight': 'bold'})
    ax1.set_xlabel("Augmentation methods", fontsize=15, fontweight='bold')
    ax1.set_ylabel(metric, fontsize=15, fontweight='bold')

    plt.tight_layout()
    plt.savefig(path_figs + model + "_" + metric + '_comparison.pdf', dpi=300)
    plt.show()

models = ['fcn', 'lstm_fcn', 'resnet']
metrics = ['Recall',  'TSS', 'HSS']
for model in models:
    for metric in metrics:
        plot(model, metric)

