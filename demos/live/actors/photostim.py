import time
import os
import h5py
import struct
import numpy as np
import random
import ipaddress
import zmq
import json
from pathlib import Path
from improv.actor import Actor, RunManager
from queue import Empty
from scipy.stats import norm
import random

import logging; logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class PhotoStimulus(Actor):

    def __init__(self, *args, ip=None, port=None, red_chan_image=None, seed=1234, stimuli=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.ip = ip
        self.port = port
        self.red_chan_image = red_chan_image
        self.frame_num = 0
        self.displayed_stim_num = 0
        self.counter_stim = 0
        self.selected_neuron = None
        self.rep_count = 5
        self.img_size = 400

        self.seed = 1337 #81 #1337 #7419
        np.random.seed(self.seed)

        self.prepared_frame = None
        self.initial=True
        self.total_stim_time = 15
        self.wait_initial = 60*8.5 #*1000

        self.stopping_list = []
        self.peak_list = []
        self.goback_neurons = []
        self.optim_f_list = []

        self.photo_frames = []
        self.photostimmed_neurons = []

    def setup(self):
        context = zmq.Context()
        
        self.socket = context.socket(zmq.PUB)
        send_IP =  self.ip
        send_port = self.port
        self.socket.bind('tcp://' + str(send_IP)+":"+str(send_port))
        self.stimulus_topic = 'stim'

        self.stimmed_neurons = []

        ### Using red channel info to help select neurons for photostimulation
        logger.error('Waiting to collect red_channel files')
        while not os.path.exists(self.red_chan_image):
            time.sleep(1)
        logger.error('The red channel image file {} now exists'.format(self.red_chan_image))
        rc = np.load(self.red_chan_image, allow_pickle=True)
        self.red_chan = 255*(rc-rc.min()) / (rc-rc.min()).max()
        logger.info('Size of red chan iamge: {}'.format(self.red_chan.shape))

        self.timer = time.time()
        self.total_times = []
        self.timestamp = []
        self.stimmed = []
        self.frametimes = []
        self.framesendtimes = []
        self.stimsendtimes = []
        self.whole_timer = time.time()

    def stop(self):
        print('Stimulus complete, avg time per frame: ', np.mean(self.total_times))
        print('Stim got through ', self.frame_num, ' frames')
        np.save('output/photostims.npy', np.array(self.photo_frames))
        
    def runStep(self):
        ### Get data from analysis actor
        x = None
        neurons = None
        com = None
        try:
            ids = self.q_in.get(timeout=0.0001)
            # logger.info('1')
            (Cx, C, Cpop, tune, color, coords, allStims, tc_list) = self.client.getList(ids[:-1])

            # logger.info('2')
            if coords is not None:
                # logger.info('com is {}'.format(com))
                neurons = [o['neuron_id']-1 for o in coords]
                com = np.array([o['CoM'] for o in coords])
            self.frame_num = ids[-1]
            # logger.info('3')
            # self.initial = False
            ### initial run ignores signals and just sends 8 basic stimuli
            if self.initial:
                # logger.info('4')
                if (time.time() - self.timer) >= self.total_stim_time and (time.time() - self.whole_timer) >= self.wait_initial:
                    x, y, r1, r2, tuning = self.pick_stim_neuron(neurons, com, tune[0], tc_list)
                    # logger.error('returned {}, {}, {}, {}'.format(x, y, r1, r2))
                    # logger.info('5')
                    if x is not None:
                        # logger.info('6')
                        self.prepared_frame = self.send_frame(x, y, r1, r2)
                        self.send_generate()
                        self.photo_frames.append(np.array([self.frame_num, self.displayed_stim_num, self.selected_neuron, x, y]))
                        self.prepared_frame = None
                        self.timer = time.time()
                        self.counter_stim += 1
                        logger.info('sent a photostim for neuron tuning  {}'.format(tuning))
                        self.photostimmed_neurons.append(self.selected_neuron)
                    else:
                        logger.error('No neurons able to be selected')

        except Empty as e:
            pass
        # except Exception as e:
        #     logger.error('Error in stimulus get: {}'.format(e))

    def send_generate(self):
        dest = "scanner2"
        type = "MC"
        src = "improv"
        id = "333"
        message = 'wf:generate'

        self.socket.send_multipart(
            [
                "dest".encode(), dest.encode(),
                "type".encode(), type.encode(),
                "src".encode(), src.encode(),
                "id".encode(), id.encode(),
                "message".encode(), message.encode(),
            ])
    
    def send_frame(self, x, y, r1, r2):
        dest = "scanner2"
        type = "MC"
        src = "improv"
        id = "333"
        xyp = '[['+str(x)+', '+str(y)+']]' #[x, y] 
        rintext = '[['+str(r1)+', '+str(r2)+']]'
        axes = "[1,1]"
        dsize = "[2,"+str(self.img_size)+","+str(self.img_size)+"]"
        xy_range = "{\"min\":{\"x\":-4,\"y\":-4},\"max\":{\"x\":4,\"y\":4}}"

        message = 'wf:roi={type=ellipse; xy_pixels='+xyp+'; rint_ext_pixels='+rintext+'; axes_dir='+axes+'; data_size='+dsize+'; xy_limits='+xy_range+';}'

        self.socket.send_multipart(
            [
                "dest".encode(), dest.encode(),
                "type".encode(), type.encode(),
                "src".encode(), src.encode(),
                "id".encode(), id.encode(),
                "xy_pixels".encode(), xyp.encode(),
                "axes dir".encode(), axes.encode(),
                "data_size".encode(), dsize.encode(),
                "xy_limits".encode(), xy_range.encode(),
                "message".encode(), message.encode(),
            ])

        self.timer = time.time()
        logger.info('Number of neurons targeted: {}, {}'.format(self.displayed_stim_num, self.frame_num))
        logger.info('Neuron selected for stim {} at {}, {}'.format(self.selected_neuron, x, y))
        self.displayed_stim_num += 1
        self.q_out.put([self.selected_neuron, x, y, self.frame_num])


    def pick_stim_neuron(self, neurons, com, tune, tc_list):

        ## compute rc intensity for all neurons in rough area
        ## TODO: using coords? that isn't the area however
        rc = self.red_chan
        rg = 3
        rcI = []
        iList = []
        for nn in range(com.shape[0]):
            try:
                if nn not in self.photostimmed_neurons:
                    cx = int(com[nn][0])
                    cy = int(com[nn][1])
                    inten = np.sum(rc[cx-rg:cx+rg, cy-rg:cy+rg])
                    if inten > 400: 
                        rcI.append(nn)
                        iList.append(inten)
            except:
                pass
        # try:
        #     logger.info('rcI list {}'.format(rcI))
        # except: pass
        # maxI = np.argmax(np.array(rcI))
        possibles = None
        try:
            ll = np.array(tc_list)
            ll[ll==np.inf] = 0
            ll[ll<0] = 0
            left = np.array([-1,-1,1,0])
            right = np.array([1,-1,-1,0])
            dotted = ll.dot(left) #left)
            possibles = np.squeeze(np.argwhere(np.abs(dotted - dotted.max()) < 75)) #50)
        except:
            pass
        # candidate = np.argmax(ll.dot(left))
        if len(self.photostimmed_neurons) > 0 and possibles is not None and self.photostimmed_neurons[0] is not None:
            # logger.info('{}'.format(possibles))
            # logger.info('{}'.format(self.photostimmed_neurons))
            # print(np.append(possibles, self.photostimmed_neurons))
            pp = np.unique(np.append(possibles, self.photostimmed_neurons)).tolist()
            # print(pp)
        else: pp = possibles.tolist()

        if pp is None: pp = []

        # try:
        if self.counter_stim >= self.rep_count or self.displayed_stim_num < 1:
            try:
                pp.extend(rcI)
            except:
                try:
                    pp = [pp].extend(rcI)
                except:
                    pass
            index = random.choice(pp) #rcI) #neurons) # maxI #
            self.selected_neuron = index
            self.counter_stim = 0
        else:
            index = self.selected_neuron

        x, y = com[index][1], com[index][0]         ##THIS IS WHERE X AND Y ARE SWAPPED CORRECTLY
        # except:
        #     x, y = 350, 100

        r1 = 0
        r2 = 5

        return x, y, r1, r2, ll[index]


## old neuron selection criteria
# logger.info('got to pici stim neuron')
        # # ll = np.array(tc_list)
        # # ll[ll==np.inf] = 0
        # # ll[ll<0] = 0
        # # left = np.array([-1,-1,1,0])
        # # candidate = np.argmax(ll.dot(left))
    
        # # if self.counter_stim >= self.rep_count or self.displayed_stim_num < 1:
        # #     index = candidate #random.choice(neurons) # candidate #.shape[0])
        # #     # print('tuning curve is ', tune[candidate], 'candidate ', candidate, ' color ', ll[candidate])
        # #     self.selected_neuron = index
        # #     self.counter_stim = 0
        # # else:
        # #     index = self.selected_neuron
        # # # print('Selecting neuron # '+str(index)+' for stimulation')
        # # # print('com is ', com[index])
        # # x, y = com[index][0], com[index][1]
        # r1 = [1, 3]
        # r2 = [1, 3]

        ################# 
        # ll = np.array(tc_list)
        # ll[ll==np.inf] = 0
        # ll[ll<0] = 0
        # left = np.array([-1,-1,1,0])
        # right = np.array([1,-1,-1,0])
        # dotted = ll.dot(right) #left)
        # possibles = np.squeeze(np.argwhere(np.abs(dotted - dotted.max()) < 75)) #50)
        # # candidate = np.argmax(ll.dot(left))
        # if len(self.photostimmed_neurons) > 0:
        #     # print(possibles)
        #     # print(self.photostimmed_neurons)
        #     # print(np.append(possibles, self.photostimmed_neurons))
        #     pp = np.unique(np.append(possibles, self.photostimmed_neurons))
        #     # print(pp)
        # else: pp = possibles