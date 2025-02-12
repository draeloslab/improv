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

from BayesOpt.model.config import Config
from BayesOpt.model.optimizer import Optimizer

import logging; logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class BayesOpt(Actor):
    def __init__(self, *args, ip=None, port=None, seed=1234, stimuli = None, **kwargs):
        super().__init__(*args, **kwargs)

    def setup(self):
        context = zmq.Context()
        
        print('Starting setup')
        self._socket = context.socket(zmq.PUB)
        send_IP =  self.ip
        send_port = self.port
        self._socket.bind('tcp://' + str(send_IP)+":"+str(send_port))
        self.stimulus_topic = 'stim'
        print('Done setup VisStim')

        config = Config()

    def stop(self):
        np.save('output/optimized_neurons.npy', np.array(self.optimized_n))
        # print(self.stopping_list)
        np.save('output/stopping_list.npy', np.array(self.stopping_list))
        # print(self.peak_list)
        np.save('output/peak_list.npy', np.array(self.peak_list))
        # print(self.optim_f_list)
        np.save('output/optim_f_list.npy', np.array(self.optim_f_list))

    def runStep(self):

        #this is will contain the logic from teh stimulus actor when asked to initialize/update the GP