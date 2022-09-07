# Pytorch dataset class

import os
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import h5py
from sklearn.preprocessing import OneHotEncoder

class BCMDataset(Dataset):
    """BCM dataset"""

    def __init__(self, file_path, window_size = 3, stride = 1, MFCC_stride = 0.005):
        """
        Args:
        ----------
            file_path : str
                Path to the h5 file.
            window_size : int, optional
                The size of the window in seconds. The default is 3.
            stride : int, optional
                The stride of the window in seconds. The default is 1.
            MFCC_stride : float, optional
                The stride of the MFCC in seconds. The default is 0.01.
        """
        self.window_size = window_size
        self.stride = stride
        self.MFCC_stride = MFCC_stride
        self.mfccs_pr_window = int(window_size/MFCC_stride)
        self.mfccs_pr_stride = int(stride/MFCC_stride)

        #Empty list to store the data
        self.data = np.load(file_path)

        y = np.zeros(len(self.data))
        i = 0
        while i < len(self.data):
            # Generate random number
            n = int(500 + np.random.rand()*500)
            y[i:i+n] = int(np.random.rand()*5)
            i += n

        # Use sklearn one hot encoding
        onehot_encoder = OneHotEncoder(sparse=False)
        y = y.reshape(len(y), 1)
        y = onehot_encoder.fit_transform(y)
        self.y = y


    def __len__(self):
        return int((len(self.data) - self.mfccs_pr_window) / self.mfccs_pr_stride)


    def __getitem__(self, idx):

        position = idx * self.mfccs_pr_stride
        x = self.data[position : position + self.mfccs_pr_window]
        y = self.y[position : position + self.mfccs_pr_window]
        pass
        return torch.from_numpy(x).float().cpu(), torch.from_numpy(y).float().cpu()
    
    

# Print dataset version
print("Dataset version:", 0.8)

#Testing 
if False:
    dataset = BCMDataset('data/mfcc_array2.npy')
    print(len(dataset))
    print(dataset[0])

    pass
