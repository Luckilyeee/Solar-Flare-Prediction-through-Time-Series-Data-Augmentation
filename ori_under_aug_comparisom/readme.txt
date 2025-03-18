for original data: python3 predict.py run_all   python3 predict.py generate_results_csv
for undersampling: python3 predict1.py run_all  python3 predict1.py generate_results_csv
for stratified sampling:  python3 predict2.py run_all  python3 predict2.py generate_results_csv
for augmentation based on the stratified sampling: python3 predict3.py run_all   python3 predict3.py generate_results_csv


After the csv files have been generated,  plot the results using box plots
plot ori vs under: python3 box_plots.py
plot stratified vs aug_on stratified: python3 box_plots1.py

