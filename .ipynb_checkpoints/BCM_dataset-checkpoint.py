# Pytorch dataset class

import os
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import h5py
from sklearn.preprocessing import OneHotEncoder

class BCMDataset(Dataset):
    """BCM dataset"""

    def __init__(self, file_path, window_size = 3, stride = 1, MFCC_stride = 0.005, transform=None):
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
        self.transform = transform
        self.file_path = file_path
        self.window_size = window_size
        self.stride = stride
        self.MFCC_stride = MFCC_stride
        

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
        return int(len(self.data))


    def __getitem__(self, idx):
        return torch.from_numpy(self.data[idx]).float().cpu(), torch.from_numpy(self.y[idx]).float().cpu()
    
    def concat_train_test_datasets(path): # Uses all files in folder to concatenate test and train datasets
        # os walk to get all files in folder
        training_set_list = []
        val_set_list = []
        file_count = 0
        for file in sorted(os.walk(path)):
            # Add every fourth file to validation set
            if file_count % 4 == 0:
                val_set_list.append(BCMDataset(file))
            else:
                training_set_list.append(BCMDataset(file))
            file_count += 1
            
            
    
    

# Print dataset version
print("Dataset version:", 0.12)

#Testing 
if False:
    dataset = BCMDataset('data/mfcc_array2.npy')
    print(len(dataset))
    print(dataset[0])

    pass
