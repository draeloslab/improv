import numpy as np
from itertools import product

x1 = np.arange(0, 331, 30)
x2 = np.array([0.02, 0.04, 0.06, 0.08, 0.10, 0.12]) 
x3 = np.arange(405,1525,100)
x4 = np.arange(625,1285,100)
x5 = np.arange(20, 401, 40)
x6 = np.arange(20, 801, 80)
x7 = np.array([0,1])
x8 = np.linspace(1,120, num=12).astype(int)

# labels = ['angle', 'vel']
labels = ['angle', 'vel', 'init_posx', 'init_posy', 'length', 'width', 'shape', 'frequency']
# labels = ['direction', 'spatial_freq', 'speed', 'contrast']

# np.save('stimuli.npy', np.array([x1,x2], dtype=object)) #,x3,x4])

np.save('stimuli.npy', np.array([x1,x2,x3,x4,x5,x6,x7,x8], dtype=object)) #,x3,x4])
np.save('labels.npy', labels)

