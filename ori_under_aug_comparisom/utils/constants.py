# for original binary
ITERATIONS = 5 # nb of random runs for random initializations

DATASET_NAMES = ['solarflare']
CLASSIFIERS = ['lstm_fcn', 'fcn', 'resnet']
ARCHIVE_NAMES = ['SF_real_ori']
DATA_FOLDERS = {'Ori_data'}

data_folders_for_archive = {'SF_real_ori': 'Ori_data'}

# for undersampling binary
# ITERATIONS = 5 
# DATASET_NAMES = ['solarflare']
# # 'fcn', 'resnet' exchanged in logs
# CLASSIFIERS = ['fcn', 'lstm_fcn', 'resnet']
# ARCHIVE_NAMES = ['SF_real_under']
# DATA_FOLDERS = {'train_under'}
#
# data_folders_for_archive = {'SF_real_under': 'train_under'}



