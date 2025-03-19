import numpy as np

x1 = np.arange(0, 331, 30)
x2 = np.array([0.02, 0.04, 0.06, 0.08, 0.10, 0.12]) 
x3 = np.arange(405,1525,100)
x4 = np.arange(625,1285,100)
x5 = np.arange(20, 401, 40)
x6 = np.arange(20, 801, 80)
x7 = np.array([0,1])
x8 = np.linspace(1,120, num=12).astype(int)

labels = ['angle', 'vel', 'init_posx', 'init_posy', 'length', 'width', 'shape', 'frequency']
stim = np.array([x1,x2,x3,x4,x5,x6,x7,x8], dtype=object)

np.save('stimuli.npy', stim)
np.save('labels.npy', labels)