# for ratio study
ITERATIONS = 5
DATASET_NAMES = ['solarflare']
sizes = [0, 1, 2, 3, 4, 5, 6]

CLASSIFIERS = ['fcn']
ARCHIVE_NAMES = ['SF_aug_permutation']
DATA_FOLDERS = {'permutation'}
data_folders_for_archive = {'SF_aug_permutation': 'permutation'}

CLASSIFIERS = ['resnet']
ARCHIVE_NAMES = ['SF_aug_permutation']
DATA_FOLDERS = {'permutation'}
data_folders_for_archive = {'SF_aug_permutation': 'permutation'}

CLASSIFIERS = ['lstm_fcn']
ARCHIVE_NAMES = ['SF_aug_spawner']
DATA_FOLDERS = {'SPAWNER'}
data_folders_for_archive = {'SF_aug_spawner': 'SPAWNER'}
