import numpy as np
from itertools import product
import random

class StimulusSpace():
    def __init__(self): # how to make this more modular (i.e. will user have to come in and change these parameters in the class? 

        x1 = np.arange(0, 331, 30)
        x2 = np.array([0.02, 0.04, 0.06, 0.08, 0.10, 0.12]) 
        # x3 = np.arange(405,1525,100)
        # x4 = np.arange(625,1285,100)
        # x5 = np.arange(20, 401, 40)
        # x6 = np.arange(20, 801, 80)
        # x7 = np.array([0,1])
        # x8 = np.linspace(1,120, num=12).astype(int)

        labels = ['angle', 'vel']
        stim = np.array([x1, x2], dtype=object)
        # labels = ['angle', 'vel', 'init_posx', 'init_posy', 'length', 'width', 'shape', 'frequency']
        # stim = np.array([x1,x2,x3,x4,x5,x6,x7,x8], dtype=object)

        # example of initializing 6 random stimuli
        self.initial_stim_count = 6
        initial_stim = []
        for _ in range(self.initial_stim_count):
            random_stim = []
            for param in stim:
                rand_stim = random.choice(param)
                random_stim.append(np.argwhere(rand_stim == param)[0][0])
            initial_stim.append(random_stim)

        # put stimuli, labels, and initial stim in dictionary
        self.stim_space = {
            'stimuli': stim,
            'labels': labels,
            'initial_stim': initial_stim,
        }

    def param_to_idx(self, stimuli):
        # translation from parameter space to index space
        indices = []
        for d, stim in enumerate(stimuli):
            indices.append(np.argwhere(stim == self.stim_space['stimuli'][d])[0][0])
        
        return indices

    def idx_to_param(self, indices):
        # translation from index space to parameter spacez
        parameters = []
        for d,idx in enumerate(indices):
            parameters.append(self.stim_space['stimuli'][d][idx])
        
        return parameters