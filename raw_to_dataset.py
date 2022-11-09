#!python3 -m pip install python_speech_features # Install PSF specifically on python3.
import os
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import h5py
from python_speech_features import mfcc, fbank
import matplotlib.pyplot as plt
import tracemalloc
from SerialTriggerDecoder import SerialTriggerDecoder
from scipy import signal
# Import fft
from scipy.fftpack import fft

def raw_to_dataset(path):
    
    # Load the data from the .h5 file
    file = h5py.File(os.path.join(path), 'r')
    data_full = np.array(file['DAQ970A']['data'])
    data_bcm = []
    labels_bcm = []
    
    # Get the relevant data from the .h5 file
    data_bcm.append(data_full[:,0])
    data_bcm.append(data_full[:,1])
    labels_bcm.append(file['DAQ970A']['data'][:,3])
    
    # Make a list of the indices where the triggers are
    last_decoded = -500000
    label_index_list = [] # List of the indices of the labels
    for i, j in enumerate(np.rint(labels_bcm[0])):
    
        if int(j) and (i > last_decoded+50000*8):
            #print(manchester_decode(np.rint(labels_bcm[0][i-900:i+100000][0::2480])))
            print(f"i: {i}")
            print(f"Time: {i/50000} s")
            last_decoded = i
            
            label_index_list.append(i)
    
    
    index_labels = [] # A list of labels for the indexes in label_index_list

    for i in range(6):
        index_labels.extend([0,0,0,1,1,1,2,2,2,-1])

    for i in range(18): index_labels.append(3)
    index_labels.append(-1)

    for i in range(18): index_labels.append(4)
    index_labels.append(-1)

    # New list with negative values removed list conprehension
    index_labels_new = [x for x in index_labels if x != -1]
    
    # Add the sections to 5 lists corresponding to the 5 different labels/classes
    ''' 
    Classes:
        Breathing: 0
        Snoring: 1
        Hold_breath: 2
        Chewing: 3
        Talking: 4
    '''
    fs = 50000
    nested_class_list = [[],[],[],[],[]]


    for datastream in data_bcm: # Loop through the data 
        for i, j in enumerate(label_index_list):
            if index_labels[i]>=0:
                nested_class_list[index_labels[i]].append(datastream[j:j+fs*10]) # Append 
                
    class_label_list = ['Breathing', 'Snoring', 'Hold_breath', 'Chewing', 'Talking']


    # create a folder with the same name as the path
    path = path.split('.')[0]
    if not os.path.exists(path):
        os.mkdir(path)
    
    
    # Concatenate the arrays, calc mfccs and save them to files
    for i, j in enumerate(nested_class_list):
        stacked_array = np.hstack(j)
        
        # Plot the stacked array
        '''fig = plt.figure(figsize=(30, 5))
        plt.plot(stacked_array, label=f'Data for label {i}')
        plt.xlabel('Samples')
        plt.ylabel('Amplitude')
        plt.legend()
        plt.grid(which = 'minor')
        plt.show()'''
        
        # Create and save mfccs
        data_mfcc = mfcc(stacked_array, samplerate = fs, nfft = 1600, winlen=0.032, winstep=0.032, numcep=16) # Sample rate is important when using mel scale
        
        #np.save(f'data/bcm_alt_3/train/{i}.npy', data_mfcc)
        
        np.save(f'{path}/{i}.npy', data_mfcc)
    
    
    
#raw_to_dataset("bcm_activity_dataset/2022-09-20_14-58-39.hdf5")