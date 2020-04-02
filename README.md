# iSeg-2019

## Introduction
This project is the implementation of our method(Lequan Yu, Caizi Li) for MICCAI Grand Challenge on 6-month Infant Brain MRI Segmentation from Multiple Sites(http://iseg2019.web.unc.edu/). Our Code is based on 3D_DenseSeg(https://github.com/tbuikr/3D_DenseSeg) and ADVENT(https://github.com/valeoai/ADVENT). 

## Requirements
PyTorch 1.2
Python 3.5
Ubuntu 16.04
Cuda 10.0
PyCharm 2019.3.3 (Community Edition)
batchgenerators(https://github.com/MIC-DKFZ/batchgenerators)
GeForce RTX 2080Ti

## Usage

Step 1: 
Change the root directory 'PROJECT_ROOT' into your owns in Config.config

Step 2:
Put the training data, validation data and testing data in 'PROJECT_ROOT/Dataset/src/iSeg-2019-Training', 'PROJECT_ROOT/Dataset/src/iSeg-2019-Validation' and 'PROJECT_ROOT/Dataset/src/iSeg-2019-Testing', respectively.

Step 3:
Enter directory 'Data_preprocessing', generate hdf5 files for data of 'Step 2' by 'prepare_hdf5_cutedge.py', 'prepare_hdf5_cutedge_valdata.py' and 'prepare_hdf5_cutedge_testdata.py'. The hdf5 files for training, validation and testing will be found in directory 'PROJECT_ROOT/Dataset/hdf5_iseg_data', 'PROJECT_ROOT/Dataset/hdf5_iseg_val_data' and 'PROJECT_ROOT/Dataset/hdf5_iseg_test_data', respectively.

Step 4:
Enter directory 'Main', run 'train.py' and 'test.py' for training and testing.


