import numpy as np

x1 = np.linspace(0, 360, num=9, dtype=int)[:-1] #np.arange(0, 331, 30, dtype=int)
x2 = np.linspace(50, 400, num=5, dtype=int)
x3 = np.array([0.02, 0.04, 0.06, 0.08, 0.10, 0.12]) 
x4 = np.array([1, 3, 10, 20], dtype=int) #np.insert(np.arange(4, 101,8, dtype=int),0,1)

labels = ['angle', 'size', 'vel', 'freq'] 

np.save('stimuli.npy', np.array([x1,x2, x3, x4], dtype=object))
np.save('labels.npy', labels)