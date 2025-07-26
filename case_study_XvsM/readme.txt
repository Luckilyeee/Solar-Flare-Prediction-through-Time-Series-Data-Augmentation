generate XvsM baseline data: python3 data_XvsM.py
Augment XvsM: python3 aug_XvsM.py

XvsM baseline: python3 predict_XM.py run_all   python3 predict_XM.py generate_results_csv
XvsM augmentation: python3 predict_XM_aug.py run_all  python3 predict_XM_aug.py generate_results_csv

After the csv files have been generated,  plot the results using box plots: python3 plots_final.py
