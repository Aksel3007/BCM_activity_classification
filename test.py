import numpy as np
#load numpy file
new = np.load('bcm_activity_dataset/2022-09-20_14-58-39/0.npy')
old = np.load('data/bcm/validation/0.npy')

print(new[0])
print(old[0])

print(new[1])
print