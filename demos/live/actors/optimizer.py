# TODO: make a BayesOpt actor (take optimizer class from Stimulus actor)
import time
import numpy as np
import random
import zmq
from improv.actor import Actor
from queue import Empty
from scipy.stats import norm
import random
from itertools import product
import yaml
from datetime import datetime as dt

from BayesOpt.model.config import Config
from BayesOpt.model.optimizer import Optimizer

from gen_stim import StimulusSpace
# from gen_stim_calibrate import StimulusSpace

import logging; logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class BayesOptimizer(Actor):
    def __init__(self, *args, stimuli=None, param_file=None, **kwargs):
        super().__init__(*args, **kwargs)

        # Stimulus Space information (loading from stimulus class)
        self.stimuli_space = StimulusSpace()
        self.stim_space = self.stimuli_space.stim_space
        # logger.info("what is stimuli_space {}; what is stim_space {}".format(self.stimuli_space, self.stim_space))
        # self.stimuli = self.stim_space['stimuli']
        self.stimuli = np.array([np.sort(stim) for stim in self.stim_space['stimuli']], dtype=object)
        # logger.info('reading in stim: {}'.format(self.stimuli))
        self.total_stim_time = self.stim_space['total_stim_time']
        self.d = self.stimuli.shape[0]
        self.initial_length = self.stimuli_space.initial_stim_count
        logger.info('Stimuli info: Num of Stimuli Parameters: {}, Num of Initial Stim: {}'.format(self.d, self.initial_length))
        logger.info("Stim space specification: {}".format(self.stim_space))

        # -----------------------------------------------------------------------------

        self.param_file = param_file
        self.init_params = yaml.safe_load(open(self.param_file, 'r'))

        init_T = self.init_params['General']['init_T']
        self.seed = self.init_params['General']['seed']
        self.maxT = self.init_params['General']['max_tests']

        self.config = Config(self.param_file)
        self.stim_choice = self.config.stim_choice
        self.GP_stimuli = self.config.exs

        self.stopping_crit = float(self.init_params['Optimizer']['optim_1']['stopping_crit']) #3.0e-4
        kernels = self.init_params['Optimizer']['optim_1']['kernel']
        self.optim = Optimizer(self.config, kernels) #gamma[:self.d], var, nu, eta, self.config.x_star)

        # List of all stimuli combinations
        xs = np.meshgrid(*self.stimuli, indexing='ij') #,x3,x4])
        x_star = np.empty(xs[0].shape + (self.d,))
        for i in range(self.d):
            x_star[...,i] = xs[i]

        self.stim_star = x_star.reshape(-1, self.d)
        logger.info('stim_star: {}'.format(self.stim_star))

        # before proceeding, check if dimensions (in gen_stim & bayesopt.yaml) matched
        stimuli_length = [len(i) for i in self.stimuli]
        if len(self.stimuli) != len(self.stim_choice) or stimuli_length != self.stim_choice:
            # logger.error("MISMATCH DIMENSION!!! STIM LENGTH FROM YAML {}; VS FROM StimulusSpace {}".format(self.stim_choice, stimuli_length))
            raise ValueError(f"MISMATCH DIMENSION!!! Expect {stimuli_length} from StimulusSpace, got {self.stim_choice} from yaml")

        self.X0 = np.zeros((self.d, init_T))
        self.X = self.X0.copy()
        self.y0 = None
        self.nID = None

        self.optimized_n = []
        self.goback_neurons = []
        self.stopping_list = []
        self.peak_list = []
        self.optim_f_list = []

        self.total_times_update = []
        self.total_times = []

        self.saved_GP_est = []
        self.saved_GP_unc = []


    def setup(self):
    
        self.stop_sending = False
        self.initial = True
        self.newN = False
        self.counter = 0
        self.timer = time.time()

        self.stim_ind = None

        

    def stop(self):

        np.save('output/optimized_neurons.npy', np.array(self.optimized_n))
        np.save('output/stopping_list.npy', np.array(self.stopping_list))
        np.save('output/peak_list.npy', np.array(self.peak_list))
        np.save('output/optim_f_list.npy', np.array(self.optim_f_list))

        try:
            np.savetxt('output/timing/optimizer_time.txt', np.array(self.total_times))
            np.savetxt('output/timing/optimizer_time_udpates.txt', self.total_times_update, fmt="%s")
        except Exception as e:
            logger.error("Trouble saving optimizer timings: {}".format(e))
            pass
        logger.info('Optimizer complete, avg time per frame: {}'.format(np.mean(self.total_times)))

    def runStep(self):
        t = time.time()
        try:
            ids = self.q_in.get(timeout=0.0001)
            X = self.client.get(ids[0])
            Y = self.client.get(ids[1])

            # logger.info('X, Y: {}, {}'.format(X, Y))

            tmpX = np.squeeze(np.array(X)).T
            # logger.info(f'{tmpX.shape}, {len(Y)}----------------------------------------------------')
            sh = len(tmpX.shape)
            if sh > 1:
                self.X = tmpX.copy()
                if tmpX.shape[1] > 4:
                    self.X = tmpX[:, -tmpX.shape[1]:]

            try:
                b = np.zeros([len(Y),len(max(Y,key = lambda x: len(x)))])
                for i,j in enumerate(Y):
                    b[i][:len(j)] = j
                self.y0 = b.T
            except:
                pass
            
        except Empty as e:
            pass
        except Exception as e:
            print('Error in optimizer get: {}'.format(e))
        
        if self.stop_sending:
            pass

        elif self.initial: 
            # displays initial stimulus 
            # internally counts to make sure that we only send correct number of initial stim
            flag = False
            if self.stim_ind is None:
                logger.info("what is the current counter: {}".format(self.counter))
                self.stim_ind = self.stim_space['initial_stim'][self.counter-1] ## FIXME: counter started with 1 (somehow)
                # self.stim_ind, flag = self.stimuli_space.initial_stim(self.stimuli, self.counter)

            if (time.time() - self.timer) >= self.total_stim_time:
                self.links['stim_ind_out'].put(self.stim_ind)
                self.stim_ind = None
                self.counter += 1  # FIXME: counter started with 1 (somehow)
                # logger.info("self.counter just added by 1!")
                self.timer = time.time()
            
            if self.counter-1 >= self.stimuli_space.initial_stim_count:  # FIXME: counter started with 1 (somehow)
                flag = True
            
            if flag:
                logger.info('Done with initial frames...')
                self.initial = False
                self.newN = True
            
    
        elif self.newN:

            nonopt = np.array(list(set(np.arange(self.y0.shape[0]))-set(self.optimized_n)))
            logger.info('nonopt is {}, number of neurons '.format(nonopt,self.y0.shape[0]))

            if len(nonopt) >= 1 or len(self.goback_neurons)>=1:
                if len(nonopt) >= 1:
                    self.nID = nonopt[np.argmax(np.mean(self.y0[nonopt,:], axis=1))]
                    logger.info('selecting most responsive neuron: {}'.format(self.nID))
                    self.optimized_n.append(self.nID)
                    self.saved_GP_est = []
                    self.saved_GP_unc = []
                elif len(self.goback_neurons)>=1:
                    self.nID = self.goback_neurons.pop(0)
                    logger.info('Trying again with neuron {}'.format(self.nID))
                    self.optimized_n.append(self.nID)
                
                print(self.y0.shape, self.X.shape, self.X0.shape)
                if self.X.shape[1] < self.y0.shape[1]:
                    self.optim.initialize_GP(self.X[:, :].T, self.y0[self.nID, -self.X.shape[1]:].T)
                elif self.y0.shape[1] < self.maxT:
                    self.optim.initialize_GP(self.X[:, -self.y0.shape[1]:].T, self.y0[self.nID, -self.y0.shape[1]:].T)
                else:
                    self.optim.initialize_GP(self.X[:, :].T, self.y0[self.nID, :].T)
                self.test_count = 0
                self.newN = False
                self.stopping = np.zeros(self.maxT)

                curr_unc = np.diagonal(self.optim.sigma).reshape((self.stim_choice))
                curr_est = self.optim.f.reshape((self.stim_choice))
                self.saved_GP_unc.append(curr_unc)
                self.saved_GP_est.append(curr_est)

                ids = []
                ids.append(self.nID)
                ids.append(self.client.put(curr_est)) #, 'est'))
                ids.append(self.client.put(curr_unc)) # 'unc'))
                self.q_out.put(ids)
        
        else:
            # need to update the GP
            t_update = time.time()
            if self.stim_ind is None: 
                X = np.zeros(self.d) 
                for i in range(self.d):
                    X[i] = self.GP_stimuli[i][int(self.X[i,-1])]

                logger.info('optim {} , update GP with {}, {}'.format( self.nID, X, self.y0[self.nID, -1]))
                self.optim.update_GP(np.squeeze(X), self.y0[self.nID,-1])

                curr_unc = np.diagonal(self.optim.sigma).reshape((self.stim_choice))
                curr_est = self.optim.f.reshape((self.stim_choice))
                self.saved_GP_unc.append(curr_unc)
                self.saved_GP_est.append(curr_est)

                ids = []
                ids.append(self.nID)
                ids.append(self.client.put(curr_est)) #, 'est'))
                ids.append(self.client.put(curr_unc)) #, 'unc'))
                self.q_out.put(ids)

                stopCrit, PI = self.optim.stopping()
                logger.info('----------- stopCrit: {}'.format(stopCrit))
                self.stopping[self.test_count] = stopCrit
                self.test_count += 1

                self.total_times_update.append([dt.now(), time.time() - t_update])

                if stopCrit < self.stopping_crit: 
                    peak = self.stim_star[np.argmax(self.optim.f)]
                    logger.info('Satisfied with this neuron, moving to next. Est peak: {}'.format(peak))
                    # self.nID += 1
                    self.newN = True
                    self.stopping_list.append(self.stopping)
                    self.peak_list.append(peak)
                    self.optim_f_list.append(self.optim.f)

                    np.save('output/saved_GP_est_'+str(self.nID)+'.npy', np.array(self.saved_GP_est))
                    np.save('output/saved_GP_unc_'+str(self.nID)+'.npy', np.array(self.saved_GP_unc))

                elif self.test_count >= self.maxT:
                    logger.info('exceeded test count')
                    self.goback_neurons.append(self.nID)
                    self.newN = True
                    self.stopping_list.append(self.stopping)
                    peak = self.stim_star[np.argmax(self.optim.f)]
                    self.peak_list.append(peak)
                    self.optim_f_list.append(self.optim.f)

                else:
                    ind, xt_1 = self.optim.max_acq()
                    logger.info('suggest next stim: {}, {}, {}'.format(ind, xt_1, xt_1.T[...,None].shape))
                    next_ind = []
                    for i in range(self.d):
                        next_ind.append(np.where(self.stimuli[i] == self.stim_star[ind][i])[0][0])
                    self.stim_ind = next_ind

            # Need to send ind to stimulus actor to create this stim request ??
            if (time.time() - self.timer) >= self.total_stim_time:
                self.links['stim_ind_out'].put(self.stim_ind)
                self.stim_ind = None
                self.timer = time.time()
                
        self.total_times.append(time.time() - t)

class RandomSampler(Actor):
    def __init__(self, *args, stimuli=None, param_file=None, **kwargs):
        super().__init__(*args, **kwargs)

        ''' RandomSampler displays a initial stimuli set, and then proceeds to send random stimuli requests. '''

        # Stimulus Space information (loading from stimulus class)
        self.stimuli_space = StimulusSpace()
        self.stim_space = self.stimuli_space.stim_space
        self.stimuli = self.stim_space['stimuli']
        self.total_stim_time = self.stim_space['total_stim_time']
        self.d = self.stimuli.shape[0]
        self.initial_length = self.stimuli_space.initial_stim_count
        logger.info('Stimuli info: Num of Stimuli Parameters: {}, Num of Initial Stim: {}'.format(self.d, self.initial_length))

        # ----------------------------------------------------------------------------
        # List of all stimuli combinations
        xs = np.meshgrid(*self.stimuli, indexing='ij') #,x3,x4])
        x_star = np.empty(xs[0].shape + (self.d,))
        for i in range(self.d):
            x_star[...,i] = xs[i]

        self.stim_star = x_star.reshape(-1, self.d)
        logger.info('stim_star: {}'.format(self.stim_star))

        self.total_times = []

    def setup(self):
    
        self.stop_sending = False
        self.initial = True
        self.newN = False
        self.counter = 0
        self.timer = time.time()

        self.stim_ind = None

    def stop(self):
        pass

    def runStep(self):
        t = time.time()
        # Listening to the analysis actor (currently commented out for RandomSampler)
        # try:
        #     ids = self.q_in.get(timeout=0.0001)

        #     # X, Y, stim, _ = self.client.get(ids)
        #     X = self.client.get(ids[0])
        #     Y = self.client.get(ids[1])
        #     # frame_num = self.client.get(ids[2]) # maybe (sometimes this try block "fails" and so the frame num isn't recorded?)

        #     # logger.info('X, Y: {}, {}'.format(X, Y))

        #     tmpX = np.squeeze(np.array(X)).T
        #     # logger.info(f'{tmpX.shape}, {len(Y)}----------------------------------------------------')
        #     sh = len(tmpX.shape)
        #     if sh > 1:
        #         self.X = tmpX.copy()
        #         if tmpX.shape[1] > 4:
        #             self.X = tmpX[:, -tmpX.shape[1]:]
        #         # print('self.X DIRECT from analysis is ', X, 'and self.X is ', self.X[:,-1])
        #     # print(self.X)

        #     try:
        #         b = np.zeros([len(Y),len(max(Y,key = lambda x: len(x)))])
        #         for i,j in enumerate(Y):
        #             b[i][:len(j)] = j
        #         self.y0 = b.T
        #     except:
        #         pass
            

        # except Empty as e:
        #     pass
        # except Exception as e:
        #     print('Error in optimizer get: {}'.format(e))

        if self.initial: 
            # displays initial stimulus 
            # internally counts to make sure that we only send correct number of initial stim
            flag = False
            if self.stim_ind is None:
                self.stim_ind = self.stim_space['initial_stim'][self.counter]

            if (time.time() - self.timer) >= self.total_stim_time:
                logger.info('Displaying initial stimuli....')
                self.links['stim_ind_out'].put(self.stim_ind)
                self.stim_ind = None
                self.counter += 1
                self.timer = time.time()
            
            if self.counter >= self.stimuli_space.initial_stim_count:
                flag = True
            
            if flag:
                logger.info('Done with initial frames... starting random sampler')
                self.initial = False
                self.newN = True
                self.counter = 0
            
        elif self.newN:
            if self.stim_ind is None:
                logger.info('random')
                np.random.shuffle(self.stim_star)
                grid = self.stim_star[self.counter]
                
                #FIXME: This is a manual method (need to fix to make it more flexible)
                param0 = np.argwhere(int(grid[0]) == self.stimuli[0])[0][0]
                param1 = np.argwhere(grid[1] == self.stimuli[1])[0][0]
                param2 = np.argwhere(int(grid[2]) == self.stimuli[2])[0][0]
                param3 = np.argwhere(int(grid[3]) == self.stimuli[3])[0][0]

                self.stim_ind = [param0, param1, param2, param3]

            if (time.time() - self.timer) >= self.total_stim_time:
                self.links['stim_ind_out'].put(self.stim_ind)
                self.stim_ind = None
                self.counter += 1
                self.newN = True
                self.timer = time.time()
        
        self.total_times.append(time.time() -t)
        
