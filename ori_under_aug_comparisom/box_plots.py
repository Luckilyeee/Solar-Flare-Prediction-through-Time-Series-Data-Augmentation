import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
warnings.resetwarnings()

plt.rcParams["pdf.use14corefonts"] = True

path_ori = 'aug_sf/results_ori/'
path_under = 'aug_sf/results_under/'
path_figs_orivsunder = 'aug_sf/figs/results_ori_vs_under/'

def plot_group(model):
  print(model)
  df1 = pd.read_csv(path_ori + model + "/res.csv")
  df2 = pd.read_csv(path_under + model + "/res.csv")

  metrics = ['Recall', 'TSS', 'HSS']
  ori_method = df1['archive_name'].unique()
  under_method = df2['archive_name'].unique()
  print(ori_method, under_method)

  ori = df1[df1['archive_name'] == ori_method[0]]
  under = df2[df2['archive_name'] == under_method[0]]

  # Combine 'ori' and 'under' DataFrames into a single DataFrame with a 'Category' column
  ori['Category'] = 'Ori'
  under['Category'] = 'Undersampled'
  combined_data = pd.concat([ori, under], ignore_index=True)

# Transpose the data to have metrics as columns
  combined_data = combined_data.melt(id_vars=['classifier_name', 'Category'], value_vars=metrics,
                                   var_name='Metrics', value_name='Values')

# Create the box plot
  plt.figure(figsize=(5, 4))
  sns.boxplot(data=combined_data, x='Metrics', y='Values', hue='Category', palette='Set2')
  plt.grid(True)
  plt.xlabel('Metrics', fontsize=15, fontweight='bold')
  plt.ylabel('Values', fontsize=15, fontweight='bold')


  plt.xticks(rotation=45, fontsize=12, fontweight='bold')
  plt.legend(loc='best', prop={'weight': 'bold'})
  plt.tight_layout()
  plt.savefig(path_figs_orivsunder + model + '_comparison.pdf', format='pdf', dpi=300)
  plt.show()

models = ['fcn', 'resnet', 'lstm_fcn']
for model in models:
  plot_group(model)

