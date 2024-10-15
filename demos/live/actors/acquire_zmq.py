import os
import time
import h5py
import numpy as np
import zmq
import json
import pathlib
from pathlib import Path
from improv.actor import Actor, RunManager
from ast import literal_eval as make_tuple
import matplotlib.pyplot as plt
import pickle 
import re
import ast
from datetime import datetime as dt

import logging; logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# logging.basicConfig(
#     format='%(asctime)s - %(levelname)s - %(message)s',
#     datefmt='%Y-%m-%d %H:%M:%S',
#     level=logging.INFOss
# )

class ZMQAcquirer(Actor):

    def __init__(self, *args, ip=None, ports=None, output=None, red_chan_image=None, init_filename=None, init_frame=60, **kwargs):
        super().__init__(*args, **kwargs)
        print("init")
        self.ip = ip
        self.ports = ports
        self.frame_num = 0
        self.stim_count = 0
        self.initial_frame_num = init_frame     # Number of frames for initialization
        self.init_filename = init_filename 
        self.red_chan_image = red_chan_image
        
        self.output_folder = str(output)
        pathlib.Path(output).mkdir(exist_ok=True) 
        pathlib.Path(output+'timing/').mkdir(exist_ok=True)

    def setup(self):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.SUB)
        for port in self.ports:
            self.socket.connect("tcp://"+str(self.ip)+":"+str(port))
            logger.info('Connected to '+str(self.ip)+':'+str(port))
        self.socket.connect("tcp://localhost:5010")
        # logger.info('Connected to '+str(self.ip)+':'+str(port))
        self.socket.setsockopt(zmq.SUBSCRIBE, b'')

        self.saveArray = []
        self.saveArrayRedChan = []
        self.save_ind = 0
        self.fullStimmsg = []
        self.total_times_frame = []
        self.total_times_pstim = []
        self.timestamp_frame = []
        self.timestamp_pstim = []
        self.stimmed = []
        self.frametimes = []
        self.framesendtimes = []
        self.stimsendtimes = []
        self.tailsendtimes = []
        self.tails = []
        self.photostims = []

        self.tailF = False
        self.stimF = False
        self.frameF = False
        self.align_flag = True

        if not os.path.exists(self.init_filename):

            ## Save initial set of frames to output/initialization.h5
            self.kill_flag = False
            while self.frame_num < self.initial_frame_num:
                self.runStep()

            self.imgs = np.array(self.saveArray)
            f = h5py.File(self.init_filename, 'w', libver='earliest')
            f.create_dataset("default", data=self.imgs)
            f.close()

        # if not os.path.exists(self.red_chan_image):
        #     if len(self.saveArrayRedChan) > 1:
        #         mean_red = np.mean(np.array(self.saveArrayRedChan), axis=0)
        #         np.save(self.red_chan_image, mean_red)

        self.frame_num = 0
        self.track = 0

        self.kill_flag = True

        # ## reconnect socket
        # self.socket.close()
        # self.socket = context.socket(zmq.SUB)
        # for port in self.ports:
        #     self.socket.connect("tcp://"+str(self.ip)+":"+str(port))
        #     print('RE-Connected to '+str(self.ip)+':'+str(port))
        # self.socket.setsockopt(zmq.SUBSCRIBE, b'')


    def stop(self):
        logger.info('Acquire ZMQ stopping procedure --')
        self.imgs = np.array(self.saveArray)
        logger.info('Trying to save 1')
        f = h5py.File('output/sample_stream_end.h5', 'w', libver='earliest')
        logger.info('Trying to save 2')
        f.create_dataset("default", data=self.imgs)
        logger.info('Trying to save 3')
        f.close()
        logger.info('Trying to save 4')

        np.savetxt('output/stimmed.txt', np.array(self.stimmed))
        np.savetxt('output/photostimmed_msgs.txt', np.array(self.photostims))
        np.save('output/tails.npy', np.array(self.tails))
        np.savetxt('output/timing/frametimes.txt', np.array(self.frametimes))
        np.savetxt('output/timing/framesendtimes.txt', np.array(self.framesendtimes), fmt="%s")
        np.savetxt('output/timing/stimsendtimes.txt', np.array(self.stimsendtimes), fmt="%s")
        np.savetxt('output/timing/tailsendtimes.txt', np.array(self.tailsendtimes), fmt="%s")
        np.savetxt('output/timing/acquire_frame_time.txt', self.total_times_frame, fmt="%s")
        np.savetxt('output/timing/acquire_pstim_time.txt', self.total_times_pstim, fmt="%s")
        np.savetxt('output/timing/acquire_frame_timestamp.txt', self.timestamp_frame, fmt="%s")
        np.savetxt('output/timing/acquire_pstim_timestamp.txt', self.timestamp_pstim, fmt="%s")
        np.save('output/fullstim.npy', self.fullStimmsg)

        logger.info('Acquisition complete, avg time per frame: {}'.format(np.mean(self.total_times_frame)))
        logger.info('Acquire got through {} frames'.format(self.frame_num))

    def runStep(self):
        try:
            self.get_message()
        except zmq.Again:
            # No messages available
            pass 
        except Exception as e:
            print('error: {}'.format(e))

    def get_message(self, timeout=0.001):

        #  try receiving microscope message: 
        try:
            msg = self.socket.recv_pyobj(flags=0)
            if isinstance(msg, dict):
                msg_dict = msg
                message_data = msg_dict['data']
                finalthing = np.array(message_data)
                tag = msg_dict['type']
            elif isinstance(msg, str):
                msg_dict, category = self._msg_unpacker(msg)
                tag = 'stim'
                
            # logger.info('Receiving microscope image--')
        except Exception as e:
            logger.info('error: {}'.format(e))

        # try receiving pandastim message:
        # try:
        #     msg = self.socket.recv_multipart()
        #     msg_dict, category = self._msg_unpacker(msg)
        #     tag = 'stim'
        #     # logger.info('Receiving stimuli information--')
        # except:
        #     pass
        # logger.info('RECIEVING IMAGES ---------')
        # logger.info('image message received (raw): {}'.format(msg))
        # logger.info('msg type: {}'.format(type(msg)))
        # try:
        #     #NOTE: brucker_2pcontrol sends msg as dict, so no need to use msg_unpacker for this
            
        #     msg_dict = self._msg_unpacker(msg
        #     # logger.info('inside try block - msg_dict')
        #     tag = msg_dict['type'] 
        # except Exception as e:
        #     msg_dict = msg
        #     logger.error('Weird format message {}'.format(e))
        
        # logger.info('tag: {}'.format(tag))
        
        # trying to visualize data
         #np.array((np.array(message_data) - 0) / 1 * 255, np.uint8)
        # logger.info('finalthing shape {}'.format(finalthing.shape))
        # logger.info('finalthing: {}'.format(finalthing))
        # plt.imshow("Microscope Image", finalthing)
        # plt.show()

        if 'stim' in tag: 
            if not self.stimF:
                logger.info('Receiving stimulus information')
                self.stimF = True
            t0 = time.time()
            self.fullStimmsg.append(msg)
            self._collect_stimulus(msg_dict, category)
            self.total_times_pstim.append(time.time() - t0)
            self.timestamp_pstim.append([dt.now(), self.frame_num])

        # elif 'frame' in tag: 
        else:
            t0 = time.time()
            if self.track %2 == 0:
                self._collect_frame(finalthing)
                self.frame_num += 1
            self.total_times_frame.append(time.time() - t0)
            self.timestamp_frame.append([dt.now(), self.frame_num])
            self.track += 1

        # elif str(tag) in 'tail':
        #     if not self.tailF:
        #         logger.info('Receiving tail information')
        #         self.tailF = True
        #     self._collect_tail(msg_dict)

        # elif 'scan' in tag:track
        #     if 'scanner2' in msg_dict['source']:
        #         logger.info('Photostim happened at frame {}'.format(self.frame_num))
        #         self.photostims.append(self.frame_num)

        # else:
        #     logger.info('Had an error in tag: {}'.format(tag))
        #     logger.info('{}'.format(msg))


    def _collect_frame(self, array):
        # array = np.array(json.loads(msg_dict['data']))
        if not self.frameF:
            logger.info('Receiving frame information')
            self.frameF = True
            logger.info('Image frame(s) size is {}'.format(array.shape))
            if array.shape[0] == 2:
                logger.info('Acquiring also in the red channel')
        self.saveArray.append(array)
        if array.shape[0] == 2:
            self.saveArrayRedChan.append(array[1])
        
        if not self.align_flag:
            array = None
        obj_id = self.client.put(array)
        self.q_out.put([{str(self.frame_num): obj_id}])

        # sendtime =  array['timestamp'] 

        self.frametimes.append([self.frame_num, time.time()])
        # self.framesendtimes.append([sendtime])
        # logger.info('sent a frame on')
        if len(self.saveArray) >= 1000:
            self.imgs = np.array(self.saveArray)
            f = h5py.File(self.output_folder+'/sample_stream'+str(self.save_ind)+'.h5', 'w', libver='earliest')
            f.create_dataset("default", data=self.imgs)
            f.close()
            self.save_ind += 1
            del self.saveArray
            self.saveArray = []
            logger.info('after saving internal')
        
        

    def _collect_stimulus(self, msg_dict, category):
        # sendtime = msg_dict['time']

        # category = str(msg_dict['raw_msg']) #'motionOn' 
        if 'alignment' in category:
            ## Currently not using
            s = msg_dict[5]
            status = str(s.decode('utf8').encode('ascii', errors='ignore'))
            if 'start' in status:
                self.align_flag = False
                logger.info('Starting alignment...')
            elif 'completed' in status:
                self.align_flag = True
                print(msg_dict)
                logger.info('Alignment done, continuing')
        elif 'move' in category:
            pass 
            # print(msg)  
        elif 'motionOn' in category:
            self.stim_count += 1
            
            ## visual stim with Matt
            # angle2 = None
            # angle, angle2 = make_tuple(msg_dict['angle'])
            # if angle>=360:
            #     angle-=360
            # stim = self._realign_angle(angle)
            # self.links['stim_queue'].put({self.frame_num:[stim, float(angle), float(angle2)]})
            # self.stimmed.append([self.frame_num, stim, angle, angle2, time.time()])
            # logger.info('Stimulus: {}, angle: {},{}, frame {}'.format(stim, angle, angle2, self.frame_num))
            if msg_dict['texture']['texture_name'] in ('sin_gray', 'sin_rgb', 'grating_gray', 'grating_rgb'):
                angle = float(msg_dict['stimulus']['angle'])
                vel = float(msg_dict['stimulus']['velocity'])
                self.links['stim_queue'].put({self.frame_num:[angle, vel]})
                self.stimmed.append([self.frame_num, angle, vel])
                logger.info('Stimulus: Moving gratings angle {} with velocity {} at frame {}'.format(angle, vel, self.frame_num))

            ## spots stim with Karina
            if msg_dict['texture']['texture_name'] == 'gray_circle':
                size = float(msg_dict['circle_radius'])
                vel = float(msg_dict['velocity'])
                self.links['stim_queue'].put({self.frame_num:[size, vel]})
                self.stimmed.append([self.frame_num, size, vel])
                logger.info('Stimulus: Circle radius {} with velocity {} at frame {}'.format(size, vel, self.frame_num))

            logger.info('Number of stimuli: {}'.format(self.stim_count))
            # self.stimsendtimes.append([sendtime])

    def _collect_tail(self, msg_dict):
        sendtime = msg_dict['timestamp']
        tails = np.array(msg_dict['tail_points']) 
        self.tails.append(tails) 
        self.tailsendtimes.append([sendtime])

    def _msg_unpacker(self, msg):
        # logger.info('keys: {}'.format(msg[::2]))
        # logger.info('vals: {}'.format(msg[1::2]))
        # keys = msg[::2]
        # vals = msg[1::2]
        
        # msg_dict = {}
        # for k, v in zip(keys, vals):
        #     msg_dict[k.decode()] = v.decode()
        
        # logger.info('msg_dict inside msg_unpacker: {}'.format(msg_dict))

        msg_unpacked = msg #pickle.loads(msg[0])
        # logger.info('unpacked message: {}'.format(msg_unpacked))

        category = None
        if 'motionOn' in msg_unpacked:
            category = 'motionOn'
        elif 'queueAddition' in msg_unpacked:
            category = 'queueAddition'
        elif 'None' in msg_unpacked:
            category = 'noStimChange'
        else:
            category = 'stimChange'
        # logger.info('CATEGORY: {}'.format(category))

        if category == 'noStimChange':
            msg_dict = {}
            logger.info('No stim change')
        else:
            start_idx = msg_unpacked.find("{")
            end_idx = msg_unpacked.find("}}")+2
            msg_str= msg_unpacked[start_idx:end_idx]
            msg_str = re.sub(r"np\.float64\(([^)]+)\)", r"\1", msg_str)

            # logger.info('formatted message type: {}'.format(msg_str))
        #     try:
            msg_dict = ast.literal_eval(msg_str)
        #     except Esxception as e:
        #         logger.error('ERROR: {}'.format(e))

        # logger.info('msg_dict: {}'.format(msg_dict))
        return msg_dict, category

    def _realign_angle(self, angle):
        if 23 > angle >=0:
            stim = 9
        elif 360 > angle >= 338:
            stim = 9
        elif 113 > angle >= 68:
            stim = 3
        elif 203 > angle >= 158:
            stim = 13
        elif 293 > angle >= 248:
            stim = 4
        elif 68 > angle >= 23:
            stim = 10
        elif 158 > angle >= 113:
            stim = 12
        elif 248 > angle >= 203:
            stim = 14
        elif 338 > angle >= 293:
            stim = 16
        else:
            logger.error('Stimulus angle unrecognized')
            stim = 0
        return stim
