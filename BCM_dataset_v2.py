# Pytorch dataset class

import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import h5py
from sklearn.preprocessing import OneHotEncoder
import torch.utils.data as data
from sklearn import datasets


class bcmDataset(Dataset):
    """BCM dataset"""

    def __init__(self, file_path, class_id, window_size = 3, stride = 3, MFCC_stride = 0.032, transform=None):
        """
        Args:
        ----------
            file_path : str
                Path to the h5 file.
            class_id : int
                The class of the speaker (each file is a speaker, and each file creates a dataset that can then be concatenated).   
            window_size : int, optional
                The size of the window in seconds. The default is 3.
            stride : int, optional
                The stride of the window in seconds. The default is 1.
            MFCC_stride : float, optional
                The stride of the MFCC in seconds. The default is 0.01.
        """
        self.transform = transform
        self.file_path = file_path
        self.window_size = window_size
        self.stride = stride
        self.MFCC_stride = MFCC_stride        
        self.mfccs_pr_window = int(window_size/MFCC_stride)
        self.mfccs_pr_stride = int(stride/MFCC_stride)

        #Empty list to store the data
        self.data = np.load(file_path)

        y = np.zeros((len(self.data),5))
        y[:,class_id] = 1
        self.y = y


    def __len__(self):
        return int((len(self.data) - self.mfccs_pr_window) / self.mfccs_pr_stride)


    def __getitem__(self, idx):
        position = idx * self.mfccs_pr_stride
        x = self.data[position : position + self.mfccs_pr_window]
        y = self.y[position : position + self.mfccs_pr_window]

        return torch.from_numpy(x).float(), torch.from_numpy(y[0]).float()
    
def concat_train_test_datasets(path, window_size = 3, stride = 0.032, MFCC_stride = 0.032): # Uses all files  in folder to concatenate test and train datasets
    # os walk to get all files in folder
    training_set_list = []
    val_set_list = []
    
    print('Validation set')
    for subdir, dirs, files in sorted(os.walk(f'{path}/validation')):
        for i, file in enumerate(sorted(files)):
            val_set_list.append(bcmDataset(f'{path}/validation/{file}',class_id = i, window_size = window_size, stride = stride, MFCC_stride = MFCC_stride))
            print(f'{path}/validation/{file}')
    
    print('Training set')
    for subdir, dirs, files in sorted(os.walk(f'{path}/train')):
        for i, file in enumerate(sorted(files)):
            training_set_list.append(bcmDataset(f'{path}/train/{file}',class_id = i, window_size = window_size, stride = stride, MFCC_stride = MFCC_stride))
            print(f'{path}/train/{file}')
    
    
    return data.ConcatDataset(training_set_list), data.ConcatDataset(val_set_list)
            
            
    
    

# Print dataset version
print("Dataset version:", 2.0)

#Testing 
if False:
    dataset_train, dataset_val = concat_train_test_datasets('data/bcm')
    print(len(dataset_train))
    print(len(dataset_val))
    print(dataset_train[0])

    pass
