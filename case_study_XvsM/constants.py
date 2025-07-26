# for original X vs M
ITERATIONS = 10 # nb of random runs for random initializations
DATASET_NAMES = ['solarflare']
CLASSIFIERS = ['lstm_fcn', 'resnet', 'fcn']
ARCHIVE_NAMES = ['SF_real_ori_XvsM']
DATA_FOLDERS = {'XvsM'}
data_folders_for_archive = {'SF_real_ori_XvsM': 'XvsM'}

# for augmentation on X vs M
ITERATIONS = 10
DATASET_NAMES = ['solarflare']
CLASSIFIERS = ['lstm_fcn', 'fcn', 'resnet']
ARCHIVE_NAMES = ['SF_aug_ww', 'SF_aug_ws', 'SF_aug_wdba', 'SF_aug_tw', 'SF_aug_spawner', 'SF_aug_rgw', 'SF_aug_mw', 'SF_aug_dgw', 'SF_aug_permutation', 'SF_aug_scaling', 'SF_aug_jitter', 'SF_aug_rotation']
DATA_FOLDERS = {'discriminative_guided_warp', 'random_guided_warp', 'time_warp',
               'window_slice', 'magnitude_warp', 'SPAWNER', 'wdba', 'window_warp', 'permutation', 'scaling', 'jitter', 'rotation'}

data_folders_for_archive = {'SF_aug_ww': 'window_warp', 'SF_aug_wdba': 'wdba', 'SF_aug_spawner': 'SPAWNER',
'SF_aug_ws': 'window_slice', 'SF_aug_rgw': 'random_guided_warp', 'SF_aug_mw': 'magnitude_warp',
'SF_aug_tw': 'time_warp', 'SF_aug_dgw': 'discriminative_guided_warp',
'SF_aug_permutation': 'permutation', 'SF_aug_scaling': 'scaling', 'SF_aug_jitter': 'jitter', 'SF_aug_rotation': 'rotation'}
