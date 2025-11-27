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
        x5 = np.array([250, 850, 1450])
        x6 = np.array([600, 1000, 1300])
        x7 = np.array([0, 50, 100])
        x8 = np.array([0,1])

        labels = ['angle', 'speed', 'size', 'frequency', 'center_x', 'center_y', 'contrast', 'shape']
        self.stim = np.array([x1, x2, x3, x4, x5, x6, x7, x8], dtype=object)
        self.stim_optim = np.array([x1, x2, x3, x4, x7], dtype=object)

        param_space_grid = np.meshgrid(*self.stim, indexing='ij')
        param_space = np.stack(param_space_grid, axis=-1).reshape(-1, len(self.stim)) 
        # replace size, center_x, center_y params with -99 for the drift gratings stimuli (NOTE: tried to replace with NaN but it was having trouble with np.where)
        mask = param_space[:, 7] == 1
        for i in [2, 4, 5]:
            param_space[mask, i] = np.nan 
        temp = param_space.copy()
        mask = np.isnan(temp)
        temp[mask] = -99
        param_space = np.unique(temp, axis=0)
        # param_space[param_space == -99] = np.nan
        self.param_space = param_space

        self.param_space_size = self.param_space.shape[0]

        #NOTE: to find the corresponding index for a given stim (np.argwhere((stim == param_space).all(axis=1))[0][0]) 

        calibration_stim, self.calibration_stim_count = self.calibration_stim(self.stim)

        self.initial_stim_count = 8
        initial_stim = self.initial_stim(self.stim)

        total_stim_time = 10 # duration of stimuli (in sec)
        hold_after = 5       # hold after period  (in sec)
        stationary_t = 0     # stationary time in the beginning (in sec)

        # put stimuli, labels, and initial stim in dictionary (FIXME: consider reorganzing this dictionary (less stuff))
        self.stim_space = {
            'stimuli': self.stim,
            'stimuli_optim': self.stim_optim,
            'labels': labels,
            'calibration_stim': calibration_stim,
            'initial_stim': initial_stim,
            'total_stim_time': total_stim_time,
            'hold_after': hold_after,
            'stat_t': stationary_t, 
        }

    #FIXME: need to put these functions into index space 
    def calibration_stim(self, stim):
        
        calibration_stim = []
        drift_grating = self.param_space[(self.param_space[:,7] == 1) & (self.param_space[:,1] == 0.02) & (self.param_space[:,3] == 3) & (self.param_space[:,6] == 50)]
        for param in drift_grating:
            row_index = self.param_to_ridx(param)
            calibration_stim.append(row_index)
        
        stationary_spots = [[0,0,0,0,1,1,1,0], [0,0,0,0,0,2,1,0], [0,0,0,0,2,2,1,0], [0,0,0,0,2,0,1,0], [0,0,0,0,0,0,1,0]]
        moving_spots = [[i,2,0,1,0,0,1,0] for i in range(0,6,2)]
        spots = stationary_spots + moving_spots

        for params in spots:
            param = self.idx_to_param(params)
            row_index = self.param_to_ridx(param)
            calibration_stim.append(row_index)

        return calibration_stim, len(calibration_stim)

    
    def initial_stim(self, stim):
        initial_stim = []
        np.random.seed(42)
        scramle_dim1_param = stim[0].copy()
        np.random.shuffle(scramle_dim1_param)
        for i in range(self.initial_stim_count):
            i_stim = [scramle_dim1_param[i], stim[1][1], stim[2][1], stim[3][0], stim[4][0], stim[5][0], stim[6][1], stim[7][0]]
            r_idx = self.param_to_ridx(i_stim)
            initial_stim.append(r_idx)

        return initial_stim

    def param_to_idx(self, stimuli):
        # translation from parameter space to index space
        indices = []
        for d, stim in enumerate(stimuli):
            indices.append(np.argwhere(stim == self.stim[d])[0][0])
        
        return indices

    def idx_to_param(self, indices):
        # translation from index space to parameter spacez
        parameters = []
        for d,idx in enumerate(indices):
            parameters.append(self.stim[d][idx])
        
        return parameters
    
    def param_to_ridx(self, stimuli):

        row_index = np.argwhere((stimuli == self.param_space).all(axis=1))[0][0]

        return row_index

    def ridx_to_param(self, row_index):
        
        stimuli = self.param_space[row_index]

        return stimuli