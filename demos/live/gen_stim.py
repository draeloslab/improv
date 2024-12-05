import numpy as np
from itertools import product

x1 = np.linspace(0,330,num=12) 
x2 = np.array([0.02, 0.04, 0.06, 0.08, 0.10, 0.12]) 
x3 = np.array([list(product(np.arange(205,1525,100), np.arange(625,1285,100)))]).squeeze()
x4 = np.linspace(10,900, num=10).astype(int)
x5 = np.linspace(10, 1800, num=10).astype(int)
x6 = np.array([0,1])
x7 = np.linspace(1,330, num=12).astype(int)

# labels = ['angle', 'vel']
labels = ['angle', 'vel', 'init_pos', 'length', 'width', 'shape', 'frequency']
# labels = ['direction', 'spatial_freq', 'speed', 'contrast']

# np.save('stimuli.npy', np.array([x1,x2], dtype=object)) #,x3,x4])

np.save('stimuli.npy', np.array([x1,x2,x3,x4,x5,x6,x7], dtype=object)) #,x3,x4])
np.save('labels.npy', labels)

