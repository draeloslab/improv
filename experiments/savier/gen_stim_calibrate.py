import numpy as np
from itertools import product
import random

class StimulusSpace():
    def __init__(self): 

        x1 = np.array([0, 45, 90, 135, 180, 225, 270, 315]) #np.arange(0, 331, 30)
        x2 = np.array([0.0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12]) 
        x3 = np.array([50, 137, 225, 312, 400])
        x4 = np.array([1, 3, 10, 20])
        x5 = np.array([800, 850, 900])
        x6 = np.array([950, 1000, 1050])
        x7 = np.array([0,1])

        labels = ['angle', 'velocity', 'size', 'frequency', 'center_x', 'center_y', 'shape']
        stim = np.array([x1, x2, x3, x4, x5, x6, x7], dtype=object)

        ## 
        
        # initial_stim = self.initial_stim(stim, initial_type='baseline')

        drift_gratings = [[i, 2, 0, 0, 0, 0, 1] for i in range(len(stim[0]))]
        stationary_spots = [[0,0,0,0,1,1,0], [0,0,0,0,0,0,0], [0,0,0,0,0,2,0], [0,0,0,0,2,2,0], [0,0,0,0,2,0,0]]
        moving_spots = [[0,2,0,1,0,0,0], [0,2,0,1,0,2,0], [0,2,0,1,2,2,0], [0,2,0,1,2,0,0], [0,2,0,1,1,1,0]]

        initial_stim = stationary_spots 
        # initial_stim = drift_gratings + stationary_spots + moving_spots
        # random.shuffle(initial_stim)

        self.initial_stim_count = len(initial_stim)

        total_stim_time = 10 # duration of stimuli (in sec)
        hold_after = 5 # hold after period  (in sec)
        stationary_t = 0 # stationary time in the beginning (in sec)

        # put stimuli, labels, and initial stim in dictionary
        self.stim_space = {
            'stimuli': stim,
            'labels': labels,
            'initial_stim': initial_stim,
            'total_stim_time': total_stim_time,
            'hold_after': hold_after,
            'stat_t': stationary_t, 
        }

    def initial_stim(self, stim, initial_type):

        np.random.seed(42)
        if initial_type == 'baseline':
            np.random.shuffle(stim[0])
            initial_stim = [[i, 0, 1, 0] for i in range(len(stim[0]))]
            
        else:
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