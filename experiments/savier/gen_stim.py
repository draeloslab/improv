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
        
        self.stim_optim_space = np.array([x1, x2[1:], x3, x4, x5, x6, x7, x8], dtype=object) # Yes, it's 8D but needs it for analysis to find the right row index in the full space? (#TODO: keep checking this)
        self.stim_optim = np.array([x1, x2[1:], x3, x4, x7], dtype=object)  # 86543

        param_space_grid = np.meshgrid(*self.stim, indexing='ij')
        param_space_full = np.stack(param_space_grid, axis=-1).reshape(-1, len(self.stim)) 

        index_grids = np.meshgrid(*(np.arange(len(s)) for s in self.stim), indexing='ij')
        param_index_space_full = np.stack(index_grids, axis=-1).reshape(-1, len(self.stim))

        # replaces size, center_x, center_y params with -99 for the drift gratings stimuli (NOTE: tried to replace with NaN but it was having trouble with np.where)
        mask = param_space_full[:, 7] == 1
        for i in [2, 4, 5]:
            param_space_full[mask, i] = np.nan 
        temp = param_space_full.copy()
        mask_nan = np.isnan(temp)
        temp[mask_nan] = -99

        param_space, idx = np.unique(temp, axis=0, return_index=True)
        self.param_space = param_space
        self.param_space_size = self.param_space.shape[0]
        self.param_index_space = param_index_space_full[idx]

        # param_space_optim (this will not include drift gratings or flashing spots, only moving dots)
        mask_dg = np.any(param_space == -99, axis=1)
        mask_fs = param_space[:, 1] == float(0)
        mask1 = mask_dg | mask_fs

        param_space_optim = self.param_space[~mask1]
        mask_center_x = param_space_optim[:,4] == 850
        mask_center_y = param_space_optim[:,5] == 1000
        mask_shape = param_space_optim[:,7] == 0
        mask2 = mask_center_x & mask_center_y & mask_shape
        self.param_space_optim = param_space_optim[mask2]

        calibration_stim, self.calibration_stim_count = self.calibration_stim(self.stim)

        self.initial_stim_count = 8
        # initial_stim = self.initial_stim(self.stim)
        initial_stim = self.initial_stim(self.stim_optim_space)

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
            row_index = self.param_to_ridx(param, tag = 'calibration')
            calibration_stim.append(row_index)
        
        stationary_spots = [[0,0,0,0,1,1,1,0], [0,0,0,0,0,2,1,0], [0,0,0,0,2,2,1,0], [0,0,0,0,2,0,1,0], [0,0,0,0,0,0,1,0]]
        moving_spots = [[i,2,0,1,0,0,1,0] for i in range(0,6,2)]
        spots = stationary_spots + moving_spots

        for params in spots:
            param = self.idx_to_param(params, tag = 'calibration')
            row_index = self.param_to_ridx(param, tag='calibration')
            calibration_stim.append(row_index)

        return calibration_stim, len(calibration_stim)

    
    def initial_stim(self, stim):
        initial_stim = []
        np.random.seed(42)
        scramle_dim1_param = stim[0].copy()
        np.random.shuffle(scramle_dim1_param)
        for i in range(self.initial_stim_count):
            # various orientation, 0.02 speed, 137 size, 1 frequency, black contrast
            i_stim = [scramle_dim1_param[i], stim[1][0], stim[2][1], stim[3][0], stim[4][1], stim[5][1], stim[6][0], stim[7][0]]
            # logger.info('initial stim {}: {}'.format(i, i_stim))
            r_idx = self.param_to_ridx(i_stim, tag='initial')
            initial_stim.append(r_idx)

        return initial_stim

    def param_to_idx(self, stimuli, tag):
        # translation from parameter space to index space
        indices = []
        if tag == 'optim' or tag == 'initial':
            space = self.stim_optim_space
        else:
            space = self.stim

        for d, stim in enumerate(stimuli):
            stim_space = space[d]
            matches = np.where(stim_space == stim)[0]

            if matches.size == 0:
                indices.append(np.nan)
            else:
                indices.append(int(matches[0]))
            # indices.append(np.argwhere(stim == self.stim[d])[0][0])
        
        return indices

    def idx_to_param(self, indices, tag):
        # translation from index space to parameter spacez
        parameters = []
        if tag == 'optim' or tag == 'initial':
            space = self.stim_optim_space
        else:
            space = self.stim
        for d,idx in enumerate(indices):
            parameters.append(space[d][idx])
        
        return parameters
    
    def param_to_ridx(self, stimuli, tag):

        if tag == 'optim' or tag == 'initial':
            if isinstance(stimuli, np.ndarray) and stimuli.shape[0] == 5:
                extended_stim = np.concatenate([stimuli[:4], [850, 1000], stimuli[4:], [0]])
                row_index = np.argwhere((extended_stim == self.param_space_optim).all(axis=1))[0][0]

            else:
                row_index = np.argwhere((stimuli == self.param_space_optim).all(axis=1))[0][0]
        else:
            row_index = np.argwhere((stimuli == self.param_space).all(axis=1))[0][0]

        return row_index

    def ridx_to_param(self, row_index, tag):
        
        if tag == 'optim' or tag == 'initial':
            stimuli = self.param_space_optim[row_index]
        else:
            stimuli = self.param_space[row_index]

        return stimuli

    
    def param_space_shrinking(self, stim_set):
        '''
        "Shrinking" the param space from 8D to 5D for optimization. 
        Specifically removing the 4th, 5th, and 7th dimensions that correspond to center_x, center_y, and shape, respectively.
        '''

        stim_set_copy = stim_set.copy()

        stimuli_optim = np.delete(stim_set_copy, [4,5,7], axis=0) 

        return stimuli_optim
