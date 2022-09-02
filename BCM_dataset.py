# Pytorch dataset class

import os
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import h5py
from sklearn.preprocessing import OneHotEncoder

class BCMDataset(Dataset):
    """BCM dataset"""

    def __init__(self, file_path):
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
        return len(self.data)


    def __getitem__(self, idx):
        
        return self.data[idx], self.y[idx]
    
    


#Testing 
if True:
    dataset = BCMDataset('data/mfcc_array2.npy')
    pass
