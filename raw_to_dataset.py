#!python3 -m pip install python_speech_features # Install PSF specifically on python3.
import os
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import h5py
from python_speech_features import mfcc, fbank
import matplotlib.pyplot as plt
import tracemalloc
from scipy import signal
# Import fft
from scipy.fftpack import fft
from librosa.effects import pitch_shift

def plot_mfcc(data_mfcc, name = 'test!!!'):
    fig = plt.figure(figsize=(30, 5))
    plt.pcolormesh((data_mfcc),shading='auto')
    plt.ylabel('Mel Cepstrum Coefficients')
    plt.xlabel('Time [sec]')
    plt.show()
    #plt.savefig(name)



def raw_to_dataset(path, new_file_name, da_noise = False, da_speed = False, mfcc_stride = 0.032, mfcc_window = 0.032, da_pitch = False):
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
    path = f'{path}{new_file_name}'
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
        
        
        if da_noise:
            # Add gaussian noise to the stacked array
            noise = np.random.normal(0,0.02783*0.0002,len(stacked_array)) # 0.02783 is the standard deviation of the data, hardcoded to makes sure noise is similar across classes
            stacked_array = stacked_array + noise

        if da_speed:
            # Add random number between -5000 and 5000
            fs = 50000 + np.random.randint(-10000,10000)
            # Changing the sampling rate effectively changes the speed of the audio, so the speed is randomly changed by up to 10%.
            
        if da_pitch:
            tones = (np.random.random())-0.5 # Pitch shift between -2 and 2 semitones
            print(f'Tones: {tones}')
            stacked_array = pitch_shift(stacked_array, sr=50000, n_steps=tones)
            
            
        # Create and save mfccs
        data_mfcc = mfcc(stacked_array, samplerate = fs, nfft = int(np.round(fs*mfcc_window)), winlen=mfcc_window, winstep = mfcc_stride, numcep=16)
        pass
        # Plot the mfccs
        #plot_mfcc(np.transpose(data_mfcc[0:1000]), f'debugging_sanity_check')
        
        
        np.save(f'{path}/{i}.npy', data_mfcc)
        
    
    
print('Version 7')    
#raw_to_dataset("bcm_activity_dataset/2022-09-20_15-18-27.hdf5")

if False: # Test the function
    raw_to_dataset("bcm_behaviour_data_multi_subject/subject1/2022-09-20_14-58-39.hdf5", "test!", da_speed = False, mfcc_window = 0.064)