import numpy as np

x1 = np.arange(0, 331, 30)
x2 = np.linspace(50, 400, num=15)
# x2 = np.array([0.02, 0.04, 0.06, 0.08, 0.10, 0.12]) #np.linspace(0.02,1,num=5)

labels = ['angle', 'size'] 

np.save('stimuli.npy', np.array([x1,x2], dtype=object))
np.save('labels.npy', labels)