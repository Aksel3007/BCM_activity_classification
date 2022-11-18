import glob
import os
import numpy as np


def create_loso_datasets():
    loso_datasets = []
    p = list(np.load('bcm_behaviour_data_multi_subject/paths.npy'))

    loso_datasets.append([p[4:],p[:4]])
    loso_datasets.append([p[8:]+p[:4],p[4:8]])
    loso_datasets.append([p[:8],p[8:]])
    print('\nloso_datasets:')
    print('\ntrain\n')
    for i in range(3):
        print(i)
        print(*loso_datasets[i][0],sep='\n')
    
    print('\nval\n')
    for i in range(3):
        print(i)
        print(*loso_datasets[i][1],sep='\n')
    
    return loso_datasets



def create_indiv_datasets():
    p = list(np.load('bcm_behaviour_data_multi_subject/paths.npy'))

    indiv_datasets = []


    indiv_datasets.append([[p[1],p[2],p[3]],[p[0]]])
    indiv_datasets.append([[p[0],p[2],p[3]],[p[1]]])
    indiv_datasets.append([[p[0],p[1],p[3]],[p[2]]])
    indiv_datasets.append([[p[0],p[1],p[2]],[p[3]]])
                        
    indiv_datasets.append([[p[5],p[6],p[7]],[p[4]]])
    indiv_datasets.append([[p[4],p[6],p[7]],[p[5]]])
    indiv_datasets.append([[p[4],p[5],p[7]],[p[6]]])
    indiv_datasets.append([[p[4],p[5],p[6]],[p[7]]])

    indiv_datasets.append([[p[9],p[10],p[11]],[p[8]]])
    indiv_datasets.append([[p[8],p[10],p[11]],[p[9]]])
    indiv_datasets.append([[p[8],p[9],p[11]],[p[10]]])
    indiv_datasets.append([[p[8],p[9],p[10]],[p[11]]])
    
    
    print('\nIndividual datasets\n')
    print(indiv_datasets[0][0][0])

    print('\n\n\n')

    for dataset in indiv_datasets:
        print(*dataset[0],sep='\n')
        print(dataset[1])
        print('\n')

    return indiv_datasets


def create_full_datasets():
    p = list(np.load('bcm_behaviour_data_multi_subject/paths.npy'))
    full_datasets = []
    print('\nFull datasets\n')
    for i in range(12):
        p = list(np.load('bcm_behaviour_data_multi_subject/paths.npy'))
        val = [p[i]]
        del p[i]
        full_datasets.append([p,val])
    
    for dataset in full_datasets:
        print(*dataset[0],sep='\n')
        print(dataset[1])
        print('\n')
    
    return full_datasets


def create_ft_datasets():
    # Create datasets for finetuning

    loso_datasets = []
    ft_datasets = []
    p = list(np.load('bcm_behaviour_data_multi_subject/paths.npy'))
    loso_datasets.append([p[4:],p[:4]])
    loso_datasets.append([p[8:]+p[:4],p[4:8]])
    loso_datasets.append([p[:8],p[8:]])

    print('\nFint-tuning datasets\n')
    for i in range(4):
    
        for loso_set in loso_datasets:
            ft_val = loso_set[1]
            ft_train = ft_val[i]
            #del ft_val[i]
            ft_val = np.delete(ft_val,i)
            ft_set = []
            ft_set.append(loso_set[0])
            ft_set.append(ft_val)
            ft_set.append([ft_train])
            ft_datasets.append(ft_set)

        
    print('\ntrain\n')
    for i in range(12):
        print(i)
        print(*ft_datasets[i][0],sep='\n')
    
    print('\nval\n')
    for i in range(12):
        print(i)
        print(*ft_datasets[i][1],sep='\n')
    
    print('\nFinetune\n')
    for i in range(12):
        print(i)
        print(*ft_datasets[i][2],sep='\n')
    
    return ft_datasets