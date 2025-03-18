import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
warnings.resetwarnings()

plt.rcParams["pdf.use14corefonts"] = True
models = ['lstm_fcn', 'fcn', 'resnet']
metrics = ['Recall', 'TSS', 'HSS']
path = "aug_sf_updated/results_6x/"
save_path = "aug_sf_updated/figs/ratio_study/"

def load_data(model):
    df = pd.read_csv(path + model + '/res.csv')
    return df

def plot(df, model, metric):
    # Filter the DataFrame to select only the relevant columns
    filtered_df = df[['size', metric]]

    # Create a box plot using Seaborn
    plt.figure(figsize=(5, 4))
    sns.boxplot(x='size', y=metric, data=filtered_df)
    plt.grid(True)

    # Customize the plot
    # plt.title(model + '_' + metric)
    plt.xlabel('Syn/real ratio', fontsize=15, fontweight='bold')
    plt.ylabel(metric.upper(), fontsize=15, fontweight='bold')
    plt.tight_layout()
    # Show the plot
    plt.savefig(save_path + model + "_" + metric + '.pdf', dpi=300)
    plt.show()

for model in models:
    for metric in metrics:
        df = load_data(model)
        plot(df, model, metric)

