import numpy as np
from itertools import product
import random

class StimulusSpace():
    def __init__(self): 

        # x1 = np.array([0, 45, 90, 135, 180, 225, 270, 315]) #np.arange(0, 331, 30)
        x1 = np.array([0, 90, 180, 270])
        # x1 = np.array([0, 45, 90, 135, 180, 225, 270, 315]) #np.arange(0, 331, 30)
        x1 = np.array([0, 90, 180, 270, 360])
        x2 = np.array([0.02, 0.04, 0.06, 0.08, 0.10, 0.12]) 
        x3 = np.array([50, 137, 225, 312, 400])
        x4 = np.array([1, 3, 10, 20])

        # x3 = np.arange(405,1525,100)
        # x4 = np.arange(625,1285,100)
        # x6 = np.arange(20, 801, 80)
        # x7 = np.array([0,1])
        # x8 = np.linspace(1,120, num=12).astype(int)
        # stim_list = [x1, x2, x3, x4]

        labels = ['angle', 'vel', 'size', 'frequency']
        stim = np.array([x1, x2, x3, x4], dtype=object)

        ## 
        self.initial_stim_count = 10
        initial_stim = self.initial_stim(stim, initial_type='baseline')

        total_stim_time = 10 # duration of stimuli (in sec)
        # put stimuli, labels, and initial stim in dictionary
        self.stim_space = {
            'stimuli': stim,
            'labels': labels,
            'initial_stim': initial_stim,
            'total_stim_time': total_stim_time
        }

    def initial_stim(self, stim, initial_type):

        np.random.seed(42)
        if initial_type == 'baseline':
            # This initial stim set will only be varying 1st param (directionality (angle))
            initial_stim = []
            np.random.seed(42)
            shuffled_stim_0 = stim[0].copy()
            np.random.shuffle(shuffled_stim_0)
            shuffled_stim_0_idx = [np.where(stim[0] == value)[0][0] for value in shuffled_stim_0]

            for i in range(self.initial_stim_count):
                idx = i % len(shuffled_stim_0_idx)
                initial_stim.append([shuffled_stim_0_idx[idx], 0, 1,0])
            # initial_stim = [[i, 0, 1, 0] for i in shuffled_stim_0_idx]
            
        else:
            # This initial stim set will varying all parameters 
            shuffled_stim_list = [x.copy() for x in stim.tolist()]
            np.random.seed(42)
            for x in shuffled_stim_list:
                np.random.shuffle(x)
        
            initial_stim = []
            for i in range(self.initial_stim_count):
                ind = []
                for shuffle, original in zip(shuffled_stim_list, stim.tolist()):
                    idx = i % len(shuffle)
                    ind.append(np.argwhere(original == shuffle[idx])[0][0])
                initial_stim.append(ind)

        return initial_stim

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