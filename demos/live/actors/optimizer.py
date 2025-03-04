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

from BayesOpt.model.config import Config
from BayesOpt.model.optimizer import Optimizer

import logging; logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class BayesOpt(Actor):
    def __init__(self, *args, stimuli=None, config_file=None, **kwargs):
        super().__init__(*args, **kwargs)

        self.stimuli = np.load(stimuli, allow_pickle=True)
        self.d = self.stimuli.shape[0]

        self.config_file = config_file

        self.params = yaml.safe_load(open(self.config_file, 'r'))
        config = Config(self.params)

        self.optim = Optimizer(config.gamma[:config.d], config.var, config.eta, config.x_star)

        self.optimized_n = []
        self.goback_neurons = []
        self.nID = None


    def setup(self):
    
        self.stop_sending = False
        self.initial = True
        self.counter = 0
        self.initial_length = 8

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
        logger.info('Stim got through {} frames'.format(self.frame_num))

    def runStep(self):

        try:
            ids = self.q_in.get(timeout=0.0001) # aquiring from Analysis actor 

            X = self.client.get(ids[0])
            Y = self.client.get(ids[1])
            # logger.info('Y is receiving: {}'.format(len(Y)))
            tmpX = np.squeeze(np.array(X)).T
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
            except Exception as e:
                logger.info('X, Y shapes: {}, {}'.format(self.X.shape, self.y0.shape))
                pass
        
        except Empty as e:
            pass
        except Exception as e:
            logger.info('Error in BayesOpt Actor get: {}'.format(e))
        
        if self.stop_sending:
            pass

        if self.counter > self.initial_length:
            self.initial = False
            logger.info('Done with initial stimuli, starting optimization')

        elif self.initial:
            # displays initial stimulus 
            # counts to make sure that we only send correct number of initial stim
            
            initial_ind = self.stimuli['initial_stim']
            self.links['stim_out'].put(initial_ind)
            
        
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
                    logger.info('Trying again with neuron', self.nID)
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

                curr_unc = np.diagonal(self.optim.sigma.reshape((72,72))).reshape((12,6))
                curr_est = self.optim.f.reshape((12,6))
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
            X = np.zeros(self.d) 
            for i in range(self.d):
                X[i] = self.GP_stimuli[i][int(self.X[i,-1])]
            # X[1] = self.GP_stimuli[1][int(self.X[1,-1])]

            logger.info('optim {} , update GP with {}, {}'.format( self.nID, X, self.y0[self.nID, -1]))
            self.optim.update_GP(np.squeeze(X), self.y0[self.nID,-1])

            curr_unc = np.diagonal(self.optim.sigma.reshape((72,72))).reshape((12,6)) # FIXME
            curr_est = self.optim.f.reshape((12,6))
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

            if stopCrit < 3.0e-4: #FIXME: add to yaml file
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

                # Need to send ind to stimulus actor to create this stim request ??
                self.links['stim_out'].put(ind)

                ## OR
                # id = self.client.put(ind)
                # self.stim_out.put(id)

