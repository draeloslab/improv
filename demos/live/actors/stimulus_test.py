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

        self.seed = 42 #1337 #81 #1337 #7419 #FIXME
        np.random.seed(self.seed)

        self.prepared_frame = None
        self.random_flag = False 
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

            if self.params[4] == 0: #shape is ellipse
                text = {'texture_size': 1600,
                        'frequency': int(self.params[5]),
                        'center_x': int(self.params[0]),
                        'center_y': int(self.params[1]),
                        'width': int(self.params[3]),
                        'length': int(self.params[2]),
                        'texture_name': 'gray_ellipse',
                        'bg_intensity': 200,
                        'fg_intensity': 50,
                        }
            
            if self.params[4] == 1: #shape is a rectangle
                text = {'texture_size': 1600,
                        'frequency': int(self.params[5]),
                        'center_x': int(self.params[0]),
                        'center_y': int(self.params[1]),
                        'width': int(self.params[3]),
                        'length': int(self.params[2]),
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

    # def send_move(self, z):
    #     self._socket.send_string('move', zmq.SNDMORE)
    #     self._socket.send_pyobj(z)
    #     logger.info('sent move command')

    # def create_chosen_stim(self, ind):
    #     xt = self.stim_star[ind]
    #     angle = xt[0]
    #     vel = xt[1]
    #     posx = xt[2]
    #     posy = xt[3]
    #     length = xt[4]
    #     width = xt[5]
    #     shape = xt[6]
    #     freq = xt[7]

    #     logger.info('create chosen stim xt: '.format(xt))
    #     stim = self.create_frame(angle, vel, [posx, posy, length, width, shape, freq])
    #     return stim

    def create_frame(self, angle, vel, params):
        self.params = params
        stat_t = 0
        stim_t = stat_t + 5
        self.total_stim_time = stim_t
    
        if self.params[4] == 0: #ellipse:
            stim = {
                    'stim_name': 'moving_gray_ellipse',
                    'angle': int(angle),
                    'velocity': vel,
                    'stationary_time': stat_t,
                    'duration': stim_t,
                    'hold_after': float(stat_t),
                        }
        elif self.params[4] == 1: #rectangle:
            stim = {
                    'stim_name': 'moving_gray_rectangle',
                    'angle': int(angle),
                    'velocity': vel,
                    'stationary_time': stat_t,
                    'duration': stim_t,
                    'hold_after': float(stim_t),
                        }

        
        self.timer = time.time()
        return stim 

    # def initial_frame(self):
    #     # logger.info('self.inital_pos {}'.format(self.initial_pos))
    #     if self.which_angle%8 == 0:
    #         random.shuffle(self.initial_angles)
    #         random.shuffle(self.initial_vel)
    #         random.shuffle(self.initial_posx)
    #         random.shuffle(self.initial_posy)
    #         random.shuffle(self.initial_len)
    #         random.shuffle(self.initial_width)
    #         random.shuffle(self.initial_shape)
    #         random.shuffle(self.initial_frequency)
    #     angle = self.initial_angles[self.which_angle%8] #self.stim_sets[0][self.which_angle%len(self.stim_sets[0])]
    #     vel = self.initial_vel[self.which_angle%6]
    #     posx = self.initial_posx[self.which_angle%len(self.initial_posx)]
    #     posy = self.initial_posy[self.which_angle%len(self.initial_posy)]
    #     length = self.initial_len[self.which_angle%len(self.initial_len)]
    #     width = self.initial_width[self.which_angle%len(self.initial_width)]
    #     shape = self.initial_shape[self.which_angle%len(self.initial_shape)]
    #     freq = self.initial_frequency[self.which_angle%len(self.initial_frequency)]

        
    #     self.which_angle += 1
    #     if self.which_angle >= self.initial_length: 
    #         self.initial = False
    #         self.stop_sending = False
    #         self.newN = True
    #         self.which_angle = 0
    #         logger.info('Done with initial frames, starting random set')
        
    #     stim = self.create_frame(angle, vel, [posx, posy, length, width, shape, freq]) #, 0.14)
    #     self.timer = time.time()
    #     return stim


    # def random_frame(self):
    #     ## grid choice
    #     snum = int(self.stim_choice[0] / 2)
    #     grid = np.argwhere(self.grid_choice==self.grid_ind[self.displayed_stim_num%(snum**2)])[0] #self.which_angle%24 #self.which_angle%24 #np.argwhere(self.grid_choice==self.grid_ind[self.displayed_stim_num%(36*36)])[0]
    #     angle = self.stimuli[0][grid[0]] #self.all_angles[grid] #self.stimuli[0][grid[0]]
    #     # angle2 = self.stimuli[0][grid[1]] #TODO: i need to change these? 

    #     stim = self.create_frame(angle) #, angle2)
    #     self.timer = time.time()
    #     return stim

