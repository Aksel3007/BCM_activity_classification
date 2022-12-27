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

    def __init__(self, file_path, class_id, window_size = 3, stride = 3, MFCC_stride = 0.032, transform=None, occlusion = 0, fractional = 1, temporal_cutout = False):
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
        self.occlusion = occlusion
        self.fractional = fractional
        
        # Check if file_path contains the word spectrogram
        if 'spectrogram' in file_path:
            self.spectrogram = True
        else:
            self.spectrogram = False

        #Empty list to store the data
        self.data = np.load(file_path)
        
        # Occlusion
        if occlusion:
            for i in range(len(occlusion)):
                if occlusion[i]:
                    self.data[:,i] = 0

        y = np.zeros((len(self.data),5))
        y[:,class_id] = 1
        self.y = y
        self.data_len_multiplier = 1
        if temporal_cutout:                
            self.data_len_multiplier += 1
            if "noise" in file_path:
                self.data_len_multiplier = 1
            if "pitch" in file_path:
                self.data_len_multiplier = 1
            if "speed" in file_path:
                self.data_len_multiplier = 1
            
            
        self.length_of_data = int(((len(self.data) - self.mfccs_pr_window) / self.mfccs_pr_stride)*self.fractional)


    def __len__(self):
        return self.length_of_data*self.data_len_multiplier


    def __getitem__(self, idx):
        if idx > self.length_of_data:
            idx = idx - self.length_of_data
            cutout = True
        else:
            cutout = False
            
        position = idx * self.mfccs_pr_stride
        x = self.data[position : position + self.mfccs_pr_window]
        y = self.y[position : position + self.mfccs_pr_window]
        if self.spectrogram: # If the data is a spectrogram, we normalize it
            x = (x - np.mean(x)) / np.std(x)
        
        if cutout:
            # Temporal cutout
            # Random number between 0 and 1
            cutout_len = (np.random.random()*self.mfccs_pr_window/self.window_size)*0.2 # Multiplied with the desired cutout length in seconds 
            cutout_pos = np.random.random()*self.mfccs_pr_window
            # Replace the cutout with zeros
            x_cutout = x.copy()
            x_cutout[int(cutout_pos):int(cutout_pos+cutout_len)] = 0
            return torch.from_numpy(x_cutout).float(), torch.from_numpy(y[0]).float()
        return torch.from_numpy(x).float(), torch.from_numpy(y[0]).float()

    
def concat_train_test_datasets(path, window_size = 3, stride = 0.032, MFCC_stride = 0.032, occlusion = 0, fractional = 1, temporal_cutout = False): # Uses all files  in folder to concatenate test and train datasets
    # os walk to get all files in folder
    training_set_list = []
    val_set_list = []
    print('_____________________')
    print(type(path))
    print('_____________________')
    if type(path) == str:
        print('Validation set')
        for subdir, dirs, files in sorted(os.walk(f'{path}/validation')):
            for i, file in enumerate(sorted(files)):
                val_set_list.append(bcmDataset(f'{path}/validation/{file}',class_id = i, window_size = window_size, stride = stride, MFCC_stride = MFCC_stride, occlusion = occlusion, fractional = fractional, temporal_cutout = False))
                print(f'{path}/validation/{file}')
        
        print('Training set')
        for subdir, dirs, files in sorted(os.walk(f'{path}/train')):
            for i, file in enumerate(sorted(files)):
                training_set_list.append(bcmDataset(f'{path}/train/{file}',class_id = i, window_size = window_size, stride = stride, MFCC_stride = MFCC_stride, occlusion = occlusion, fractional = fractional, temporal_cutout = temporal_cutout))
                print(f'{path}/train/{file}')
    
    else:
        print('Validation set')
        for paradigm_file in path[1]:
            for subdir, dirs, files in sorted(os.walk(f'{paradigm_file}')):
                for i, file in enumerate(sorted(files)):
                    class_id = int(file.split('.')[0][-1])
                    val_set_list.append(bcmDataset(f'{paradigm_file}/{file}',class_id = class_id, window_size = window_size, stride = stride, MFCC_stride = MFCC_stride, occlusion = occlusion, fractional = fractional, temporal_cutout = False))
                    print(f'{paradigm_file}/{file}')
        
        print('Training set')
        for paradigm_file in path[0]:
            for subdir, dirs, files in sorted(os.walk(f'{paradigm_file}')):
                for i, file in enumerate(sorted(files)):
                    class_id = int(file.split('.')[0][-1])
                    training_set_list.append(bcmDataset(f'{paradigm_file}/{file}',class_id = class_id, window_size = window_size, stride = stride, MFCC_stride = MFCC_stride, occlusion = occlusion, fractional = fractional, temporal_cutout = temporal_cutout))
                    print(f'{paradigm_file}/{file}')
        
    
    return data.ConcatDataset(training_set_list), data.ConcatDataset(val_set_list)

            
    
    

# Print dataset version
print("Dataset version:", 5)

#Testing 
if False:
    ds_list = [['bcm_behaviour_data_multi_subject/subject1/2022-09-20_14-58-39','bcm_behaviour_data_multi_subject/subject1/2022-09-20_15-18-27'],['bcm_behaviour_data_multi_subject/subject1/2022-09-20_15-38-11']]
    
    dataset_train, dataset_val = concat_train_test_datasets(ds_list, temporal_cutout = True)
    print(len(dataset_train))
    print(len(dataset_val))
    
    
    print(*dataset_train[0][0], sep = '\n')
    print("________________________")
    print(*dataset_train[-3][0], sep = '\n')
    print("________________________")
    print(*dataset_train[-3][0], sep = '\n')
    print("________________________")
    print(*dataset_train[-3][0], sep = '\n')
    print("________________________")
    print(*dataset_train[-3][0], sep = '\n')
    print("________________________")
    print(*dataset_train[-3][0], sep = '\n')



    pass
