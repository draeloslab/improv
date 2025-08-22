import numpy as np
from itertools import product
import random
import logging; logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class StimulusSpace():
    def __init__(self): 

        x1 = np.array([0, 45, 90, 135, 180, 225, 270, 315]) #np.arange(0, 331, 30)
        # x1 = np.array([0, 90, 180, 270])
        x2 = np.array([0.02, 0.04, 0.06, 0.08, 0.10, 0.12]) 
        x3 = np.array([50, 137, 225, 312, 400])
        x4 = np.array([1, 3, 10, 20])
        x5 = np.array([0, 50, 100])

        # x3 = np.arange(405,1525,100)
        # x4 = np.arange(625,1285,100)
        # x6 = np.arange(20, 801, 80)
        # x7 = np.array([0,1])
        # x8 = np.linspace(1,120, num=12).astype(int)
        # stim_list = [x1, x2, x3, x4]

        labels = ['angle', 'velocity', 'size', 'frequency', 'contrast']
        stim = np.array([x1, x2, x3, x4, x5], dtype=object)
        # logger.info("what is stim: {}".format(stim))

        ## 
        self.initial_stim_count = 8
        initial_stim = self.initial_stim(stim, initial_type='baseline')

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
        initial_stim = []
        np.random.seed(42)
        if initial_type == 'baseline':
            scramle_dim1_param = stim[0].copy()
            np.random.shuffle(scramle_dim1_param)
            scramle_dim1_idx = [np.where(stim[0] == value)[0][0] for value in scramle_dim1_param]
            for i in range(self.initial_stim_count):
                idx = i % len(scramle_dim1_idx)
                idx1 = i % len(stim[4])
                initial_stim.append([scramle_dim1_idx[idx], 0, 1, 0, 0])  # use high contrast #idx1])
            # initial_stim = [[i, 0, 1, 0] for i in scramle_dim1_idx]
            
        else:
            shuffled_stim_list = [x.copy() for x in stim.tolist()]
            # np.random.seed(42)
            for x in shuffled_stim_list:
                np.random.shuffle(x)
        
            # initial_stim = []
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