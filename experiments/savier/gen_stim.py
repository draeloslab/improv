import numpy as np
from itertools import product
import random
import logging; logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class StimulusSpace():
    '''
    StimulusSpace is a class designed to create parameter spaces of visual stimuli.
    It defines the parameter space based on each dimension (x_):
       - param_space: Full parameter space, shape (C, d) where C is the total number of unique combinations of stim and d is the number of dimensions
       - param_index_space: Full parameter space in index space, shape (C, d)

    The class also defines different translation functions to translate between index and parameter spaces, and also between the full parameter space and subspace sets. 
    '''
    
    def __init__(self): 

        x1 = np.array([0, 45, 90, 135, 180, 225, 270, 315]) 
        x2 = np.array([0.0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12]) 
        x3 = np.array([50, 137, 225, 312, 400])
        x4 = np.array([1, 7, 39, 95])
        x5 = np.array([150, 800, 1450]) #1450])
        x6 = np.array([500, 800, 1100])
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
        self.param_index_space = param_index_space_full[idx]  # this is the full index space

        self.translation()

        # logger.info(f"param_space and r2_param same? {np.array_equal(self.param_space_optim, self.r2_params)}")

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
        logger.info(f"stim_space: {self.stim_space}")

    def calibration_stim(self, stim):
        ''' Creates a set of calibration stim (drift gratings (8), flashing spots (5), and moving dots (3))
            The set is in index space'''
        
        calibration_stim = []
        drift_grating = self.param_space[(self.param_space[:,7] == 1) & (self.param_space[:,1] == 0.02) & (self.param_space[:,3] == 7) & (self.param_space[:,6] == 50)]
        for param in drift_grating:
            row_index = self.param_to_ridx(param, tag = 'calibration')
            calibration_stim.append(row_index)
        
        stationary_spots = [[0,0,0,0,1,1,1,0], [0,0,0,0,0,2,1,0], [0,0,0,0,2,2,1,0], [0,0,0,0,2,0,1,0], [0,0,0,0,0,0,1,0]]
        moving_spots = [[i,2,0,1,1,1,1,0] for i in range(0,6,2)]
        spots = stationary_spots + moving_spots

        for params in spots:
            param = self.idx_to_param(params, tag = 'calibration')
            row_index = self.param_to_ridx(param, tag='calibration')
            calibration_stim.append(row_index)
            # logger.info(f"{self.map_full_to_r1[row_index]}")
            # if self.map_full_to_r1[row_index] >= 0:
            #     logger.info(f"params is {params}, row_idx is {row_index}, remapped? is {self.map_full_to_r1[row_index]}, {self.r1_params[self.map_full_to_r1[row_index]]}")
            
        return calibration_stim, len(calibration_stim)

    
    def initial_stim(self, stim):
        ''' Creates a set of initial stimuli (moving dots (8))
            The set is in index space'''

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
        if tag == 'optim' or tag == 'initial' or tag == "calibration_initial" or tag == 'random' or tag == 'grid':
            space = self.stim_optim_space
        else:  # calibration 
            space = self.stim

        for d, stim in enumerate(stimuli):
            stim_space = space[d]
            matches = np.where(stim_space == stim)[0]

            if matches.size == 0:
                indices.append(np.nan)
            else:
                indices.append(int(matches[0]))
        
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

        if tag == 'optim' or tag == 'initial' or tag == 'random' or tag == 'grid':
            if isinstance(stimuli, np.ndarray) and stimuli.shape[0] == 5:
                # logger.info(f"stimuli before extending: {stimuli}")
                extended_stim = np.concatenate([stimuli[:4], [850, 1000], stimuli[4:], [0]])
                # logger.info(f"stimuli after extending: {extended_stim}")
                row_index = self.idx_r2_to_r1[np.argwhere((extended_stim == self.r2_params).all(axis=1))[0][0]]
                # # self.idx_r2_to_r1
                # extended_stim = np.concatenate([stimuli[:4], [850, 1000], stimuli[4:], [0]])
                # row_index = np.argwhere((extended_stim == self.param_space_optim).all(axis=1))[0][0]  # TODO: translation happens here

            else:
                row_index = np.argwhere((stimuli == self.r1_params).all(axis=1))[0][0]
                # row_index = np.argwhere((stimuli == self.param_space_optim).all(axis=1))[0][0]  # TODO: translation happens here
        elif tag == "calibration_initial":
            row_index = np.argwhere((stimuli == self.param_space).all(axis=1))[0][0]
            row_index = self.map_full_to_r1[row_index]
            # logger.info(f"remapping HAPPENING the new row_index is {self.r1_params[row_index]} and index of {self.r1_coords[row_index]}")
        else:  # calibration
            row_index = np.argwhere((stimuli == self.param_space).all(axis=1))[0][0]
            # if self.map_full_to_r1[row_index] >= 0:
            #     # row_index = np.argwhere((stimuli == self.r1_params).all(axis=1))[0][0]
            #     logger.info(f"remapping IMAGING the new row_index is {self.r1_params[self.map_full_to_r1[row_index]]} and index of {self.map_full_to_r1[row_index]}")
                
        return row_index

    def ridx_to_param(self, row_index, tag):
        
        if tag == 'optim':
            # extended_row_index = np.concatenate([row_index[:4], [1, 1], row_index[4:], [0]])
            stimuli = self.r2_params[row_index]
        elif tag == 'initial' or tag == 'random' or tag == 'grid':
            stimuli = self.r1_params[row_index]
            # stimuli = self.param_space_optim[row_index] # TODO: translation happens here
        elif tag == 'calibration_initial':
            # logger.info("blurrrrrr")
            stimuli = self.r1_params[self.map_full_to_r1[row_index]]
            # stimuli = self.r1_params[row_index]
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
    
    def translation(self):
        '''
        "Translating" between full space and 2 subspaces (R1 and R2). 
        R1 is the subspace containing all moving dots (with shape 25920,8)
        R2 is ths subspace containing all moving dots with fixed center_x and center_y (with shape 2880, 8)
        '''

        # mapping from full space to R1 (all moving dots)
        mask_r1 = (self.param_index_space[:, 7] != 1) & (self.param_index_space[:, 1] != 0)

        # r1 to full
        self.idx_r1_to_full = np.where(mask_r1)[0]
        self.r1_coords = self.param_index_space[mask_r1].copy()
        self.r1_coords[:, 1] -= 1  # to account for the speed dimension
        self.r1_params = self.param_space[mask_r1].copy()  # all subspace?

        # r2 to r1
        mask_r2_within_r1 = (self.r1_coords[:, 4] == 1) & (self.r1_coords[:, 5] == 1)
        self.idx_r2_to_r1 = np.where(mask_r2_within_r1)[0]
        self.r2_coords = self.r1_coords[self.idx_r2_to_r1].copy()
        self.r2_params = self.r1_params[self.idx_r2_to_r1].copy()  # 2880, 8


        # full to r1 mapping
        self.map_full_to_r1 = np.full(len(self.param_index_space), -1)
        self.map_full_to_r1[self.idx_r1_to_full] = np.arange(len(self.idx_r1_to_full))
        
        # r1 to r2 mapping
        self.map_r1_to_r2 = np.full(len(self.r1_coords), -1)
        self.map_r1_to_r2[self.idx_r2_to_r1] = np.arange(len(self.idx_r2_to_r1))