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

        self.stimuli_space = StimulusSpace()
        self.stim_space = self.stimuli_space.stim_space
        self.stimuli = self.stim_space['stimuli']
        self.total_stim_time = self.stim_space['total_stim_time']
        self.hold_after = self.stim_space['hold_after']
        self.stat_t = self.stim_space['stat_t']
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
        # logger.info('Stim got through {} frames'.format(self.frame_num))
        
    def runStep(self):
        
        # Listen to request from Optimizer actor? 
        try: 
            t = time.time()
            row_index, tag = self.links['stim_ind_in'].get(timeout=0.0001) #TODO: need to confirm if this row_index makes sense
            logger.info('stim index: {}'.format(row_index))
            if tag == 'optim':
                parameters = self.stimuli_space.ridx_to_param(row_index, tag='optim') 
            else: 
                parameters = self.stimuli_space.ridx_to_param(row_index, tag='non_optim') 
            logger.info('parameters: {}'.format(parameters))
            stim = self.create_frame(parameters, tag=tag)

            self.send_frame(stim)

            self.requested_stim.append([dt.now(), parameters])
            self.total_times.append(time.time() - t)
        
        except Empty as e:
            pass
        except Exception as e:
            logger.error('Error in receiving stimulus indices from optimizer: {}'.format(e))


    def send_frame(self, stim):

        # logger.info('PARMS INSIDE SEND FRAME {}'.format(params))

        if stim is not None:

            if self.speed == float(0):
                center_x = self.center_x
                center_y = self.center_y
            else:
            #NOTE: this is hardcoded for specific directions and their most visible "center" point on the visible grid (this will move to experiments folder)
                if self.angle == 45:
                    center_x = 80
                    center_y = 1400
                elif self.angle == 135:
                    center_x = 750
                    center_y = 1500
                elif self.angle == 225:
                    center_x = 350
                    center_y = 1000
                elif self.angle == 315:
                    center_x = 1400
                    center_y = 1500
                else:
                    center_x = 850
                    center_y = 1000

            if self.shape == 0:
                texture_name = 'gray_ellipse'
                
                text = {'texture_size': 1600,
                        'frequency': int(self.frequency),
                        'center_x': center_x,
                        'center_y': center_y,
                        'width': int(self.size), 
                        'length': int(self.size),
                        'texture_name': texture_name,
                        'bg_intensity': 200,
                        'fg_intensity': int(self.contrast),
                        }

            else:
                texture_name = 'grating_gray'
                text = {'texture_size': 1600,
                        'frequency': int(self.frequency),
                        'texture_name': texture_name,
                        'light_value': 200,
                        'dark_value': int(self.contrast),
                        }

            
            stimulus = {'stimulus': stim, 'texture': text}
            # logger.info('stimulus: {}'.format(stimulus))
            
            # TODO: add timestamp (includes time and stimulus request)
            self._socket.send_string(self.stimulus_topic, zmq.SNDMORE)
            self._socket.send_pyobj(stimulus)
            self.timer = time.time()
            logger.info('Number of stimuli requested: {}'.format(self.displayed_stim_num))
            self.displayed_stim_num += 1
        else:
            logger.error('Tried to send a None frame')

    def create_frame(self, parameters, tag):
        stim_t = self. stat_t + self.total_stim_time 

        # NOTE: this is creating self.<param> based on the labels defined in gen_stim
        for label, param in zip(self.stim_space['labels'], parameters):
                setattr(self, label, param)

        # NOTE: this is a hardcoded param dict mapping for default values, in case the parameters aren't predefined (hopefully can get rid of later)
        # This allows for some flexibility when adding/removing parameters in gen_stim
        default_params = {
            'angle': 45, 
            'speed': 0.02,
            'size': 50, 
            'frequency': 1, 
            'contrast': 50,
            'shape': 0,
        }
        for key, default_param in default_params.items():
            if not hasattr(self, key):
                setattr(self, key, default_param)

        if self.shape == 0:
            stim_name = 'gray_circle'
        else:
            stim_name = 'grating_gray'

        stim = {
                'stim_name': stim_name,
                'angle': int(self.angle),
                'velocity': self.speed,
                'stationary_time': self.stat_t,
                'duration': self.total_stim_time, 
                'hold_after': float(stim_t-self.hold_after),
                'note': tag
                    }

        self.timer = time.time()
        return stim 
