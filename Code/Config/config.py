PROJECT_ROOT = '/home/licaizi/Projects/iSeg2019/'

file_paths = {
    'src_train_data_path': PROJECT_ROOT + 'Dataset/src/iSeg-2019-Training',
    'src_val_data_path': PROJECT_ROOT + 'Dataset/src/iSeg-2019-Validation',
    'src_test_data_path': PROJECT_ROOT + 'Dataset/src/iSeg-2019-Testing',
    'src_train_hdf5_path': PROJECT_ROOT + 'Dataset/hdf5_iseg_data',
    'src_val_hdf5_path': PROJECT_ROOT + 'Dataset/hdf5_iseg_val_data',
    'src_test_hdf5_path': PROJECT_ROOT + 'Dataset/hdf5_iseg_test_data',
    'train_list': PROJECT_ROOT + 'Code/Data_preprocessing/train_list.txt',
    'test_list': PROJECT_ROOT + 'Code/Data_preprocessing/test_list.txt',
    'val_list': PROJECT_ROOT + 'Code/Data_preprocessing/val_list.txt',
}


model_params = {
    'half_bn': 0.6,
    'model_ver': 'ver1_bnin_ver1',
    'batch_size': 4,
    'patch_size': [64, 64, 64],
    'nb_classes': 4,    # channels of output
    'growth_rate': 16,
    'dropout': 0.2,
    'first_output': 32,
    'reduction': 0.5,
    'N': [4, 4, 4, 4],
}

train_params = {
    'nb_iters': 35000,
    'lr_rate': 2e-4,
    'lr_rate_d': 1e-4,
    'lr_gamma': 0.1,
    'lr_step_size': 10000,
    'weight_decay': 0.0005,
    'momentum': 0.97,
    'nb_accu_iters': 1,
    'lambda_adv_target': 0.001,
    'model_snapshot_path': PROJECT_ROOT + 'Code/snapshot',
    'snapshot_step_size': 5,
    'log_path': PROJECT_ROOT + 'Code/logs',
    'resume_path': None,
    'resume_path_d': None,
}