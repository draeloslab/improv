import time
import numpy as np
import random
import zmq
from improv.actor import Actor
from queue import Empty
from scipy.stats import norm
import random
from itertools import product
from datetime import datetime as dt

from gen_stim import StimulusSpace
# from gen_stim_calibrate import StimulusSpace

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
        
        # self.stimuli = np.load(stimuli, allow_pickle=True)
        self.stimuli_space = StimulusSpace()
        self.stim_space = self.stimuli_space.stim_space
        # logger.info('stim: {}'.format(self.stim_space['stimuli']))
        self.stimuli = np.array([np.sort(stim) for stim in self.stim_space['stimuli']], dtype=object)
        # logger.info('reading in stim: {}'.format(self.stimuli))
        self.total_stim_time = self.stim_space['total_stim_time']
        np.save('output/generated_stimuli.npy', self.stim_space['stimuli'])


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
        self.requested_stim = []

    def stop(self):

        np.savetxt('output/timing/stimulus_frame_time.txt', np.array(self.total_times))
        # np.savetxt('output/requested_stimuli.txt', self.requested_stim, fmt="%s")

        logger.info('Stimulus complete, avg time per frame: {}'.format(np.mean(self.total_times)))
        logger.info('Stim got through {} frames'.format(self.frame_num))
        
    def runStep(self):
        
        # Listen to request from Optimizer actor? 
        try: 
            t = time.time()
            indices = self.links['stim_ind_in'].get(timeout=0.0001)
            logger.info('indices: {}'.format(indices))
            parameters = self.stimuli_space.idx_to_param(indices) 
            logger.info('parameters: {}'.format(parameters))
            stim = self.create_frame(parameters)

            self.send_frame(stim)

            self.requested_stim.append([dt.now(), parameters])
            self.total_times.append(time.time() - t)
        
        except Empty as e:
            pass
        except Exception as e:
            logger.error('Error in receiving stimulus indices from optimizer: {}'.format(e))

        # need to log stimuli requests
       

    def send_frame(self, stim):

        # logger.info('PARMS INSIDE SEND FRAME {}'.format(params))

        if stim is not None:

            # text = {'texture_size': 1600,
            #             'circle_center': (865,955),
            #             'circle_radius': 25,
            #             'bg_intensity': 200,
            #             'texture_name': 'calibration_dots',
            #             }

            text = {'texture_size': 1600,
                        'frequency': int(self.frequency),
                        'center_x': 900,
                        'center_y': 1000,
                        'width': int(self.size), 
                        'length': int(self.size),
                        'texture_name': 'gray_ellipse',
                        'bg_intensity': 200,
                        'fg_intensity': 50,
                        }

            # text = {'texture_size': 1600,
            #             'frequency': int(self.frequency),
            #             'center_x': 800,
            #             'center_y': 1000,
            #             'width': int(self.size), #int(self.size), 
            #             'length': int(self.size), #int(self.size),
            #             'texture_name': 'gray_ellipse',
            #             'bg_intensity': 200,
            #             'fg_intensity': 50,
            #             }

            # if self.shape == 0:
            #     text = {'texture_size': 1600,
            #             'frequency': int(self.frequency),
            #             'center_x': int(self.x_pos),
            #             'center_y': int(self.y_pos),
            #             'width': int(self.size), #int(self.size), 
            #             'length': int(self.size), #int(self.size),
            #             'texture_name': 'gray_ellipse',
            #             'bg_intensity': 200,
            #             'fg_intensity': 50,
            #             }
            # else:
            #     text = {'texture_size': 1600,
            #             'frequency': int(5),
            #             'texture_name': 'grating_gray',
            #             'light_value': 200,
            #             'dark_value': 50,
            #             }

                
            stimulus = {'stimulus': stim, 'texture': text}
            # logger.info('Stimuli requested: {}'.format(stimulus))
            # TODO: add timestamp (includes time and stimulus request)
            self._socket.send_string(self.stimulus_topic, zmq.SNDMORE)
            self._socket.send_pyobj(stimulus)
            self.timer = time.time()
            logger.info('Number of stimuli requested: {}'.format(self.displayed_stim_num))
            self.displayed_stim_num += 1
        else:
            logger.error('Tried to send a None frame')

    def create_frame(self, params):
        # self.params = params
        stat_t = 0
        stim_t = stat_t + self.total_stim_time 

        self.size = params[2]
        self.frequency = params[3]
        # self.x_pos = params[4]
        # self.y_pos = params[5]
        # self.shape = params[6]

        # stim = {
        #         'stim_name': 'gray_ellipse',
        #         'angle': int(0),
        #         'velocity': float(0),
        #         'stationary_time': stat_t,
        #         'duration': self.total_stim_time,
        #         'hold_after': float(stim_t),
        #             }

        stim = {
                'stim_name': 'gray_ellipse',
                'angle': int(params[0]),
                'velocity': params[1],
                'stationary_time': stat_t,
                'duration': self.total_stim_time,  #TODO: make the hold time modular
                'hold_after': float(stim_t-5),
                    }

        self.timer = time.time()
        return stim 
