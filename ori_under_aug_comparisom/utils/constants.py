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


# for stratified samppling binary
# ITERATIONS = 5 # nb of random runs for random initializations
#
# DATASET_NAMES = ['solarflare']
# CLASSIFIERS = ['fcn', 'lstm_fcn', 'resnet']
# ARCHIVE_NAMES = ['SF_stra']
# DATA_FOLDERS = {'stra_sampling'}
#
# data_folders_for_archive = {'SF_stra': 'stra_sampling'}

# for augmentation on stratified data
# ITERATIONS = 5
# DATASET_NAMES = ['solarflare']
# CLASSIFIERS = ['resnet']
# ARCHIVE_NAMES = ['SF_aug_ww', 'SF_aug_ws', 'SF_aug_wdba', 'SF_aug_tw', 'SF_aug_spawner', 'SF_aug_rgw', 'SF_aug_mw', 'SF_aug_dgw', 'SF_aug_permutation', 'SF_aug_scaling', 'SF_aug_jitter', 'SF_aug_rotation']
# DATA_FOLDERS = {'discriminative_guided_warp', 'random_guided_warp', 'time_warp',
#                'window_slice', 'magnitude_warp', 'SPAWNER', 'wdba', 'window_warp', 'permutation', 'scaling', 'jitter', 'rotation'}

# data_folders_for_archive = {'SF_aug_ww': 'window_warp', 'SF_aug_wdba': 'wdba', 'SF_aug_spawner': 'SPAWNER',
# 'SF_aug_ws': 'window_slice', 'SF_aug_rgw': 'random_guided_warp', 'SF_aug_mw': 'magnitude_warp',
# 'SF_aug_tw': 'time_warp', 'SF_aug_dgw': 'discriminative_guided_warp',
# 'SF_aug_permutation': 'permutation', 'SF_aug_scaling': 'scaling', 'SF_aug_jitter': 'jitter', 'SF_aug_rotation': 'rotation'}




