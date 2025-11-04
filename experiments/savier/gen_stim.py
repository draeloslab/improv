import numpy as np
from itertools import product
import random
import logging; logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class StimulusSpace():
    def __init__(self): 

        x1 = np.array([0, 45, 90, 135, 180, 225, 270, 315]) 
        x2 = np.array([0.0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12]) 
        x3 = np.array([50, 137, 225, 312, 400])
        x4 = np.array([1, 3, 10, 20])
        x5 = np.array([800, 850, 900])
        x6 = np.array([950, 1000, 1050])
        x7 = np.array([0, 50, 100])
        x8 = np.array([0,1])

        labels = ['angle', 'speed', 'size', 'frequency', 'center_x', 'center_y', 'contrast', 'shape']
        stim = np.array([x1, x2, x3, x4, x5, x6, x7, x8], dtype=object)
        stim_optim = np.array([x1, x2, x3, x4, x7], dtype=object)


        param_space_grid = np.meshgrid(*stim, indexing='ij')
        self.param_space = np.stack(param_space_grid, axis=-1).reshape(-1, len(stim)) 
        self.param_space_size = self.param_space.shape[0]

        optim_param_space_grid = np.meshgrid(*stim_optime, indexing='ij')
        self.optim_param_space = np.stack(optim_param_space_grid, axis=-1).reshape(-1, len(stim)) 
        self.optim_param_space_size = self.optim_param_space.shape[0]

        #NOTE: to find the corresponding index for a given stim (np.argwhere((stim == param_space).all(axis=1))[0][0]) 

        calibration_stim, self.calibration_stim_count = self.calibration_stim(stim)

        self.initial_stim_count = 8
        initial_stim = self.initial_stim(stim_optim, initial_type='baseline')

        total_stim_time = 10 # duration of stimuli (in sec)
        hold_after = 5       # hold after period  (in sec)
        stationary_t = 0     # stationary time in the beginning (in sec)

        # put stimuli, labels, and initial stim in dictionary
        self.stim_space = {
            'stimuli': stim,
            'stimuli_optim': stim_optim,
            'labels': labels,
            'calibration_stim': calibration_stim,
            'initial_stim': initial_stim,
            'total_stim_time': total_stim_time,
            'hold_after': hold_after,
            'stat_t': stationary_t, 
        }

    def calibration_stim(self, stim):

        drift_gratings = [[i, 2, 0, 2, 0, 0, 1, 1] for i in range(len(stim[0]))]
        stationary_spots = [[0,0,0,0,1,1,1,0], [0,0,0,0,0,0,1,0], [0,0,0,0,0,2,1,0], [0,0,0,0,2,2,1,0], [0,0,0,0,2,0,1,0]]
        moving_spots = [[0,2,0,1,0,0,1,0], [0,2,0,1,0,2,1,0], [0,2,0,1,2,2,1,0], [0,2,0,1,2,0,1,0], [0,2,0,1,1,1,1,0]]

        calibration_stim = drift_gratings + stationary_spots + moving_spots
        calibration_stim_count = len(calibration_stim)

        return calibration_stim, calibration_stim_count

    def initial_stim(self, stim, initial_type):
        initial_stim = []
        np.random.seed(42)
        if initial_type == 'baseline':
            scramle_dim1_param = stim[0].copy()
            np.random.shuffle(scramle_dim1_param)
            scramle_dim1_idx = [np.where(stim[0] == value)[0][0] for value in scramle_dim1_param]
            for i in range(self.initial_stim_count):
                idx = i % len(scramle_dim1_idx)
                # idx1 = i % len(stim[4])
                initial_stim.append([scramle_dim1_idx[idx], 1, 1, 0, 0])  # use high contrast #idx1])
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
    
    def param_to_ridx(self, stimuli, set_type):

        if set_type == 'calibration': # is this going to overcomplicate things? but if the param spaces are separate? how would that work for the analyis? 
            row_index = np.argwhere((stimuli == self.param_space).all(axis=1))[0][0]
        else:
            row_index = np.argwhere((stimuli == self.optim_param_space).all(axis=1))[0][0]

        return row_index

    def ridx_to_param(self, row_index, set_type):
        
        if set_type == 'calibration':
            stimuli = self.param_space[row_index]
        else:
            stimuli = self.optim_param_space[row_index]

        return stimuli