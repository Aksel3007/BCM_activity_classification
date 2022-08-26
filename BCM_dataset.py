# Pytorch dataset class

import os
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import h5py

class BCMDataset(Dataset):
    """BCM dataset"""

    def __init__(self, root_dir):
        #Empty list to store the data
        self.data = []
        # OS walk through the directory to find the files
        for subdir, dirs, files in sorted(os.walk(root_dir)):
            for file in files:
                if "hdf5" in file:
                    # Load the hdf5 file, and append to the list
                    self.data.append(h5py.File(os.path.join(subdir, file), 'r'))
                    
                    # Print the filename
                    print(f'{subdir}/{file}')
                    print("debug")
                


    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return 1
    





#Testing 
if True:

    data_dir = '//uni.au.dk/dfs/Tech_EarEEG/Students/Msc2022_BCM_AkselStark'
    dataset = BCMDataset(data_dir)
