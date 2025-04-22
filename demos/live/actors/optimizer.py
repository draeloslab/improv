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

# from BayesOpt.model.config import Config
# from BayesOpt.model.optimizer import Optimizer

from gen_stim_test import StimulusSpace

import logging; logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class BayesOpt(Actor):
    def __init__(self, *args, stimuli=None, config_file=None, **kwargs):
        super().__init__(*args, **kwargs)

        # self.stimuli = np.load(stimuli, allow_pickle=True)
        self.stimuli_space = StimulusSpace()
        self.stim_space = self.stimuli_space.stim_space
        self.stimuli = self.stim_space['stimuli']
        self.total_stim_time = self.stim_space['total_stim_time']
        self.d = self.stimuli.shape[0]
        self.initial_length = self.stimuli_space.initial_stim_count
        logger.info('Stimuli info: Num of Stimuli Parameters: {}, Num of Initial Stim: {}'.format(self.d, self.initial_length))

        # ----------------------------------------------------------------------------
        # this section maybe in the config file already???
        self.stim_choice = []
        self.GP_stimuli = []
        self.stim_sets = []
        for i, s in enumerate(self.stimuli):
            self.stim_choice.append(s.shape[0])

            indices = np.arange(s.shape[0])
            random.shuffle(indices)

            self.GP_stimuli.append(np.arange(s.shape[0]))
            self.stim_sets.append(s[indices].tolist())
        self.stim_choice = np.array(self.stim_choice) 
        
        logger.info('Number of possible stimuli is: {}'.format(self.stim_choice[0]))
        # -----------------------------------------------------------------------------

        self.config_file = config_file

        self.init_params = yaml.safe_load(open(self.config_file, 'r'))
        # self.config = Config(self.params)
        
        var = self.init_params['Optimizer']['var'] #1e-1
        nu = self.init_params['Optimizer']['nu'] #1 #0.5 #1e-1
        eta = self.init_params['Optimizer']['eta'] #5e-2
        maxS = self.stim_choice #np.array([l[-1] for l in self.stim_choice])
        gamma = maxS #(1 / maxS) / 2

        # -----------------------------------------------------------------------------------------------
        # Can delete this section when actually using Bayes Opt repo config and optimizer 
        gp_copy = self.GP_stimuli.copy()
        xs = np.meshgrid(*gp_copy, indexing='ij') #,x3,x4])
        x_star = np.empty(xs[0].shape + (self.d,))
        for i in range(self.d):
            x_star[...,i] = xs[i]
        self.x_star = x_star.reshape(-1, self.d) 
        logger.info('Number of possible test points to optimize over: {}'.format(self.x_star.shape[0]))
        # -----------------------------------------------------------------------------------------------

        self.optim = Optimizer(gamma[:self.d], var, nu, eta, self.x_star)

        init_T = self.init_params['General']['init_T']
        self.seed = self.init_params['General']['seed']
        self.maxT = self.init_params['General']['max_tests']
        self.stopping_crit = self.init_params['General']['stopping_crit'] #3.0e-4

        self.X0 = np.zeros((self.d, init_T))
        self.X = self.X0.copy()
        self.y0 = None
        self.nID = None

        self.optimized_n = []
        self.goback_neurons = []
        self.stopping_list = []
        self.peak_list = []
        self.optim_f_list = []
        self.total_times = []

        self.saved_GP_est = []
        self.saved_GP_unc = []

        xs = np.meshgrid(*self.stimuli, indexing='ij') #,x3,x4])
        x_star = np.empty(xs[0].shape + (self.d,))
        for i in range(self.d):
            x_star[...,i] = xs[i]

        self.stim_star = x_star.reshape(-1, self.d)
        logger.info('stim_star: {}'.format(self.stim_star))


    def setup(self):
    
        self.stop_sending = False
        self.initial = True
        self.newN = False
        self.counter = 0
        self.timer = time.time()

        self.stim_ind = None

        # logger.info('Optimizer Links: {}'.format(self.getLinks()))
        

    def stop(self):
        '''Triggered at Run
        '''
        np.save('output/optimized_neurons.npy', np.array(self.optimized_n))
        # print(self.stopping_list)
        np.save('output/stopping_list.npy', np.array(self.stopping_list))
        # print(self.peak_list)
        np.save('output/peak_list.npy', np.array(self.peak_list))
        # print(self.optim_f_list)
        np.save('output/optim_f_list.npy', np.array(self.optim_f_list))

        # TODO: need to start adding timestamps and save here 

        logger.info('Stimulus complete, avg time per frame: {}'.format(np.mean(self.total_times)))
        # logger.info('Stim got through {} frames'.format(self.frame_num))

    def runStep(self):
        # self.timer = time.time()
        try:
            ids = self.q_in.get(timeout=0.0001)

            # X, Y, stim, _ = self.client.get(ids)
            X = self.client.get(ids[0])
            Y = self.client.get(ids[1])
            # frame_num = self.client.get(ids[2]) # maybe (sometimes this try block "fails" and so the frame num isn't recorded?)

            # logger.info('X, Y: {}, {}'.format(X, Y))

            tmpX = np.squeeze(np.array(X)).T
            # logger.info(f'{tmpX.shape}, {len(Y)}----------------------------------------------------')
            sh = len(tmpX.shape)
            if sh > 1:
                self.X = tmpX.copy()
                if tmpX.shape[1] > 4:
                    self.X = tmpX[:, -tmpX.shape[1]:]
                # print('self.X DIRECT from analysis is ', X, 'and self.X is ', self.X[:,-1])
            # print(self.X)

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
            print('Error in stimulus_spots get: {}'.format(e))
        
        if self.stop_sending:
            pass

        # if self.counter >= self.initial_length:
        #     self.initial = False
        #     self.newN = True

        elif self.initial: # NOTE: consider moving initial stimuli in stimulus actor since it's just reading from a list from gen_stim
            # displays initial stimulus 
            # internally counts to make sure that we only send correct number of initial stim
            flag = False
            if self.stim_ind is None:
                self.stim_ind, flag = self.stimuli_space.initial_stim(self.stimuli, self.counter)
            # logger.info('delta time in optimizer: {}'.format(time.time() - self.timer))
            if (time.time() - self.timer) >= self.total_stim_time:
                self.links['stim_ind_out'].put([dt.now(), self.stim_ind])
                self.stim_ind = None
                self.counter += 1
                self.timer = time.time()
                # logger.info('timer reset')
            
            if flag:
                logger.info('Done with initial frames...')
                self.initial = False
                self.newN = True
                
            
    
        elif self.newN:
            # skipping random flag? as that will be it's separate optimizer actor

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
            # if self.prepared_frame is None: ??
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

                stopCrit = self.optim.stopping()
                logger.info('----------- stopCrit: {}'.format(stopCrit))
                self.stopping[self.test_count] = stopCrit
                self.test_count += 1

                if stopCrit < self.stopping_crit: #self.config.stopping_crit:
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
                self.links['stim_ind_out'].put([dt.now(), self.stim_ind])
                self.stim_ind = None
                self.timer = time.time()

class Optimizer():
    def __init__(self, gamma, var, nu, eta, x_star):
        self.gamma = gamma
        self.variance = var
        self.nu = nu
        self.eta = eta
        self.x_star = x_star        

        self.d = self.x_star.shape[1]

        self.f = None
        self.sigma = None       ## Note: this is actually sigma squared
        self.X_t = None
        self.K_t = None
        self.k_star = None
        self.y = None
        self.A = None

        self.t = 0

    def initialize_GP(self, X, y):
        ## X is a matrix (T,d) of initial T measurements we have results for

        self.X_t = X
        self.y = y

        T = self.X_t.shape[0]
        a = self.x_star.shape[0]

        self.test_count = np.zeros(a)

        self.K_t = kernel(self.X_t, self.X_t, self.variance, self.gamma)
        self.k_star = kernel(self.X_t, self.x_star, self.variance, self.gamma)

        self.A = np.linalg.inv(self.K_t + self.eta**2 * np.eye(T))
        self.f = self.k_star.T @ self.A @ self.y
        self.sigma = self.variance * np.eye(a) - self.k_star.T @ self.A @ self.k_star
        ### TODO: rewrite sigma computation to be every a not matrix mult
        # self.sigma = np.diagonal(self.sigma)

        self.t = T

    def update_obs(self, x, y):
        self.y_t1 = np.array([y])
        self.x_t1 = x[None,...]
       
    def update_GP(self, x, y):
        self.update_obs(x, y)

        ## Can't do internally due to out of memory / invalid array errors from numpy
        self.k_t, self.u, self.phi, f_upd, sigma_upd = update_GP_ext(self.X_t, self.x_t1, self.A, self.x_star, self.eta, self.y, self.y_t1, self.k_star, self.variance, self.gamma)
        self.f = self.f + f_upd
        # self.sigma = self.sigma + np.diagonal(sigma_upd)
        # self.f = self.k_star.T @ self.A @ self.y
        self.sigma = self.variance * np.eye(self.x_star.shape[0]) - self.k_star.T @ self.A @ self.k_star
        # self.sigma = np.diagonal(sigma)

        self.iterate_vars()

    def iterate_vars(self):
        self.y = np.append(self.y, self.y_t1)
        self.X_t = np.append(self.X_t, self.x_t1, axis=0)
        self.k_star = np.append(self.k_star, kernel(self.x_t1, self.x_star, self.variance, self.gamma), axis=0)

        ## update for A
        self.A = self.A + self.phi * np.outer(self.u, self.u)
        self.A = np.vstack((self.A, -self.phi*self.u.T))
        right = np.append(-self.phi*self.u, self.phi)
        self.A = np.column_stack((self.A, right))

        self.t += 1

    def max_acq(self):
        test_pt = np.argmax(self.ucb())
        
        if self.test_count[test_pt] > 3:
            test_pt = np.random.choice(np.arange(self.x_star.shape[0]))
            print('choosing random stim instead')
        self.test_count[test_pt] += 1

        return test_pt, self.x_star[test_pt]

    def ucb(self):
        tau = self.d * np.log(self.t + 1e-16)
        # import pdb; pdb.set_trace()
        sig = self.sigma
        if np.any(sig < 0):
            sig = np.clip(sig, 0, np.max(sig))
        fcn = self.f + np.sqrt(self.nu * tau) * np.sqrt(np.diagonal(sig))
        return fcn

    def stopping(self):
        val = self.f - np.max(self.f) - 1e-4
        # PI = np.max(norm.cdf((val) / (np.diagonal(self.sigma))))
        # using expected improvement
        sig = np.diagonal(self.sigma)
        EI = np.max(val * norm.cdf(val / sig) + sig * norm.pdf(val))
        return EI


def kernel(x, x_j, variance, gamma):

    # ## x shape: (T, d) (# tests, # dimensions)
    # K = np.zeros((x.shape[0], x_j.shape[0]))
    # # period = 24 ##FIXME

    # for i in range(x.shape[0]):
    #     # K[:,i] = self.variance * rbf_kernel(x[:,i], x_j[:,i], gamma = self.gamma[i])
    #     for j in range(x_j.shape[0]):
    #         ## first dimension is direction
    #         # dist = np.abs(x[i,0] - x_j[j,0])
    #         # # print(dist)
    #         # # if dist > 12:
    #         # #     dist = 24 - dist
    #         # # print(dist)
    #         # K[i,j] = np.exp(-gamma[0]*((dist)**2))
    #         # K[i,j] *= variance * np.exp(-gamma[1:].dot((x[i,1:]-x_j[j,1:])**2))

    #         ## binocular
    #         # dist1 = np.sin(np.pi * np.abs(x[i,0] - x_j[j,0]) / period)
    #         # dist2 = np.sin(np.pi * np.abs(x[i,1] - x_j[j,1]) / period)

    #         dist1 = np.abs(x[i,0] - x_j[j,0])
    #         dist2 = np.abs(x[i,1] - x_j[j,1])

    #         K[i,j] = np.exp(-gamma[0]*(dist1**2))
    #         K[i,j] *= variance * np.exp(-gamma[1]*(dist2**2))

    # new ways to compute kernel
    dist = x[:, None, :] - x_j[None, :, :]
    ws_dist = np.sum(gamma * (dist**2), axis =2)
    K = variance *np.exp(-ws_dist)
            
    return K

def update_GP_ext(X_t, x_t1, A, x_star, eta, y, y_t1, k_star, variance, gamma):

    k_t = kernel(X_t, x_t1, variance, gamma)
    u = A @ k_t
    k_t1 = kernel(x_t1, x_t1, variance, gamma)
    k_star_t1 = kernel(x_t1, x_star, variance, gamma)
    phi = np.linalg.inv(k_t1 + eta**2 - k_t.T.dot(u))
    kuk = k_star.T @ u - k_star_t1.T
    f = np.squeeze(phi * kuk * (y.dot(u) - y_t1))
    sigma = phi * (kuk**2)
    # import pdb; pdb.set_trace()

    return k_t, u, phi, f, sigma 