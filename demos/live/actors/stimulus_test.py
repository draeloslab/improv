import time
import numpy as np
import random
import zmq
from improv.actor import Actor
from queue import Empty
from scipy.stats import norm
import random
from itertools import product

import logging; logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class VisualStimulus(Actor):

    def __init__(self, *args, ip=None, port=None, seed=1234, stimuli = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.ip = ip
        self.port = port
        self.frame_num = 0
        self.displayed_stim_num = 0
        self.stop_sending = False

        self.params = []
        
        self.stimuli = np.load(stimuli, allow_pickle=True)
        np.save('output/generated_stimuli.npy', self.stimuli)

    def setup(self):
        context = zmq.Context()
        
        print('Starting setup')
        self._socket = context.socket(zmq.PUB)
        send_IP =  self.ip
        send_port = self.port
        self._socket.bind('tcp://' + str(send_IP)+":"+str(send_port))
        self.stimulus_topic = 'stim'
        print('Done setup VisStim')

        self.timer = time.time()
        self.total_times = []
        self.timestamp = []
        self.stimmed = []
        self.frametimes = []
        self.framesendtimes = []
        self.stimsendtimes = []
        self.tailsendtimes = []
        self.tails = []

    def stop(self):
        '''Triggered at Run
        '''
        
        # Need to save list of stimuli requested and when

        logger.info('Stimulus complete, avg time per frame: {}'.format(np.mean(self.total_times)))
        logger.info('Stim got through {} frames'.format(self.frame_num))
        
    def runStep(self):
        
        # Listen to request from Optimizer actor? 
        try: 
            # time benchmark
            indices = self.links['stim_in'].get(timeout=0.0001)
        except Empty as e:
            pass
        except Exception as e:
            logger.info('Error in receiving stimulus indices from optimizer: {}'.format(e))

        
        # translate to parameter space
        parameters = self.stimuli[indices] # use the stimulus translator class
        stim = self.create_frame(parameters)
        self.send_frame(stim)

        # need to log stimuli requests
       

    def send_frame(self, stim):

        # logger.info('PARMS INSIDE SEND FRAME {}'.format(params))

        if stim is not None:

            if self.params[7] == 0: #shape is ellipse
                text = {'texture_size': 1600,
                        'frequency': int(self.params[8]),
                        'center_x': int(self.params[3]),
                        'center_y': int(self.params[4]),
                        'width': int(self.params[6]),
                        'length': int(self.params[5]),
                        'texture_name': 'gray_ellipse',
                        'bg_intensity': 200,
                        'fg_intensity': 50,
                        }
            
            if self.params[7] == 1: #shape is a rectangle
                text = {'texture_size': 1600,
                        'frequency': int(self.params[8]),
                        'center_x': int(self.params[3]),
                        'center_y': int(self.params[4]),
                        'width': int(self.params[6]),
                        'length': int(self.params[5]),
                        'texture_name': 'gray_rectangle',
                        'bg_intensity': 200,
                        'fg_intensity': 50,
                        }
                
            stimulus = {'stimulus': stim, 'texture': text}
            # TODO: add timestamp (includes time and stimulus request)
            self._socket.send_string(self.stimulus_topic, zmq.SNDMORE)
            self._socket.send_pyobj(stimulus)
            self.timer = time.time()
            logger.info('Number of stimuli requested: {}'.format(self.displayed_stim_num))
            self.displayed_stim_num += 1
        else:
            logger.error('Tried to send a None frame')

    def create_frame(self, params):
        self.params = params
        stat_t = 0
        stim_t = stat_t + 5
        self.total_stim_time = stim_t
    
        if self.params[7] == 0: #ellipse:
            stim = {
                    'stim_name': 'moving_gray_ellipse',
                    'angle': self.params[0],
                    'velocity': self.params[1],
                    'stationary_time': stat_t,
                    'duration': stim_t,
                    'hold_after': float(stat_t),
                        }
        elif self.params[7] == 1: #rectangle:
            stim = {
                    'stim_name': 'moving_gray_rectangle',
                    'angle': self.params[0],
                    'velocity': self.params[1],
                    'stationary_time': stat_t,
                    'duration': stim_t,
                    'hold_after': float(stim_t),
                        }

        
        self.timer = time.time()
        return stim 
