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
import struct
from datetime import datetime as dt
import cv2

import logging; logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

from experiments.savier.gen_stim import StimulusSpace

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

        # Stimulus Space information (loading from stimulus class)
        self.stimuli_space = StimulusSpace()
        self.stim_space = self.stimuli_space.stim_space

    def setup(self):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.SUB)
        for port in self.ports:
            self.socket.connect("tcp://"+str(self.ip)+":"+str(port))
            logger.info('Connected to '+str(self.ip)+':'+str(port))
        self.socket.connect("tcp://localhost:6010")
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
        # self.framesendtimes = []
        # self.stimsendtimes = []
        # self.tailsendtimes = []
        self.tails = []
        self.photostims = []
        # self.receive_time = []
        # self.pickle_load_time = []
        # self.unpacking_time = []
        self.acq_stim_queue_ts = []
        self.improv_recv_img_ts = []
        self.improv_recv_pstim_ts = []

        self.tailF = False
        self.stimF = False
        self.frameF = False
        self.align_flag = True
        self.counter_img_number = 0  # TODO: delete after use

        if not os.path.exists(self.init_filename):

            ## Save initial set of frames to output/initialization.h5
            self.kill_flag = False
            while self.frame_num < self.initial_frame_num:
                self.runStep()

            self.imgs = np.array(self.saveArray)
            f = h5py.File(self.init_filename, 'w', libver='earliest')
            f.create_dataset("default", data=self.imgs)
            f.close()

        self.frame_num = 0
        self.track = 0

        self.kill_flag = True


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

        np.save('output/stimmed.npy', np.array(self.stimmed))
        # np.savetxt('output/photostimmed_msgs.txt', np.array(self.photostims))
        np.save('output/tails.npy', np.array(self.tails))
        np.savetxt('output/timing/frametimes.txt', np.array(self.frametimes))
        # np.savetxt('output/timing/framesendtimes.txt', np.array(self.framesendtimes), fmt="%s")
        # np.savetxt('output/timing/stimsendtimes.txt', np.array(self.stimsendtimes), fmt="%s")
        # np.savetxt('output/timing/tailsendtimes.txt', np.array(self.tailsendtimes), fmt="%s")
        # np.savetxt('output/timing/acquire_frame_time.txt', self.total_times_frame, fmt="%s")
        # np.savetxt('output/timing/acquire_pstim_time.txt', self.total_times_pstim, fmt="%s")
        np.savetxt('output/timing/acquire_frame_timestamp.txt', self.timestamp_frame, fmt="%s")
        np.savetxt('output/timing/acquire_pstim_timestamp.txt', self.timestamp_pstim, fmt="%s")
        # np.savetxt('output/timing/acquire_receive_time.txt', self.receive_time, fmt="%s")
        # np.savetxt('output/timing/acquire_pickle_load_time.txt', self.pickle_load_time, fmt="%s")
        # np.savetxt('output/timing/acquire_unpacking_time.txt', self.unpacking_time, fmt="%s")
        np.savetxt('output/timing/acquire_stim_queue_time.txt', self.acq_stim_queue_ts)
        np.savetxt("output/timing/improv_recv_img_ts.txt", self.improv_recv_img_ts) 
        np.savetxt("output/timing/improv_recv_pstim_ts.txt", self.improv_recv_pstim_ts)
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
            logger.info('error: {}'.format(e))

    def get_message(self, timeout=0.001):
        #  try receiving microscope message: 
        try:
            # BUG: 031925, recv_pyobj may not work
            # msg = self.socket.recv_pyobj(flags=0)
            msg_obj = self.socket.recv()
            
        except Exception as e:
            logger.info('error from receiving: {}'.format(e))
   
        try:
            # pickle_load_time = time.time()
            msg = pickle.loads(msg_obj)
            # self.pickle_load_time.append(time.time() - pickle_load_time)
            # unpacker_time = time.time()
            if isinstance(msg, dict):
                self.improv_recv_img_ts.append([msg['counter'], self.frame_num, time.time(), msg['ts']])  # only get the micrscopic msg
                # logger.info("dictionary raw msg: {}".format(msg))
                msg_dict = msg
                message_data = msg_dict['data']
                finalthing = np.array(message_data)
                tag = msg_dict['type']
                # self.unpacking_time.append(time.time() - unpacker_time)
            elif isinstance(msg, str):
                # logger.info('pandastim raw msg: {}'.format(msg))
                msg_dict, category = self._msg_unpacker(msg)
                tag = 'stim'
                # logger.info("the tag is (pandastim) {}".format(tag))
                # self.unpacking_time.append(time.time() - unpacker_time)
            else:
                logger.info("hey this is from the inside of pyobj we don't know what the type is")
        except pickle.UnpicklingError:
            pass
        except Exception as e:
            # logger.info('error: {}'.format(e))
            logger.info('error from pickle load: {} - {}'.format({type(e).__name__}, e))

        
        if 'stim' in tag: 
            if not self.stimF:
                logger.info('Receiving stimulus information')
                self.stimF = True
            # t0 = time.time()
            self.fullStimmsg.append(msg)
            self._collect_stimulus(msg_dict, category)
            # self.total_times_pstim.append(time.time() - t0)
            self.timestamp_pstim.append([dt.now(), self.frame_num])

        # elif 'frame' in tag: 
        else:
            # t0 = time.time()
            # if self.track %2 == 0:
            self._collect_frame(finalthing)
            self.frame_num += 1
            # self.total_times_frame.append(time.time() - t0)
            self.timestamp_frame.append([dt.now(), self.frame_num])  #should we use dt.now() or time.time() here?
            # self.track += 1


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
            self.stim_set(msg_dict)

            logger.info('Number of stimuli: {}'.format(self.stim_count))

    def _collect_tail(self, msg_dict):
        sendtime = msg_dict['timestamp']
        tails = np.array(msg_dict['tail_points']) 
        self.tails.append(tails) 
        # self.tailsendtimes.append([sendtime])

    def _msg_unpacker(self, msg):

        msg_unpacked = msg 

        category = None
        if 'motionOn' in msg_unpacked or 'stimChange: {' in msg_unpacked: # since flashing spots has speed of 0, the tag is not motionOn but instead stimChange (which causes issues later on)
            category = 'motionOn'
        elif 'queueAddition' in msg_unpacked:
            category = 'queueAddition'
        elif 'None' in msg_unpacked:
            category = 'noStimChange'
        else:
            category = 'stimChange'

        if category == 'noStimChange':
            msg_dict = {}
            logger.info('No stim change')
        else:
            try:
                if category == "motionOn":
                    _, ts_str, content = msg_unpacked.split('|', 2)
                    dt_obj = dt.strptime(ts_str.strip(), '%Y-%m-%d %H:%M:%S.%f')
                    sent_time = dt_obj.timestamp()
                    self.improv_recv_pstim_ts.append([sent_time, time.time()])
            except Exception as e:
                logger.info(f"whaat acquirer zmq error: {e}")
            start_idx = msg_unpacked.find("{")
            end_idx = msg_unpacked.find("}}")+2
            msg_str= msg_unpacked[start_idx:end_idx]
            msg_str = re.sub(r"np\.float64\(([^)]+)\)", r"\1", msg_str)

            msg_dict = ast.literal_eval(msg_str)

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


    def stim_set(self, msg_dict):

        if msg_dict['texture']['texture_name'] == 'gray_ellipse':
            angle = int(msg_dict['stimulus']['angle'])
            speed = float(msg_dict['stimulus']['velocity'])
            size = int(msg_dict['texture']['length'])
            freq = int(msg_dict['texture']['frequency'])
            center_x = int(msg_dict['texture']['center_x'])
            center_y = int(msg_dict['texture']['center_y'])
            contrast = int(msg_dict['texture']['fg_intensity'])
            shape = 0
                        
            # logger.info('Is speed = float(0): {}'.format(speed == float(0)))
            if speed == float(0):
                logger.info('Stimulus: Flashing spot at ({},{}) at frame {}'.format(center_x, center_y, self.frame_num))
            else:
                # if angle in [45, 135, 225, 315]:
                #     # Adjust angle to match stimulus space, remapping to the "center" of the stimulus screen
                #     center_x = 850
                #     center_y = 1000
                logger.info('Stimulus: {} Moving dots with size {} at angle {} and speed {} with contrast {} at frame {}'.format(freq, size, angle, speed, contrast, self.frame_num))


        elif msg_dict['texture']['texture_name'] == 'grating_gray':
            try:
                angle = int(msg_dict['stimulus']['angle'])
                speed = float(msg_dict['stimulus']['velocity'])
                size = -99 
                freq = int(msg_dict['texture']['frequency'])
                center_x = -99 
                center_y = -99 
                contrast = int(msg_dict['texture']['dark_value'])
                shape = 1
            except Exception as e:
                logger.info('acquirer receiving msg error: {}'.format(e))
            logger.info('Stimulus: Sin Drift Gratings at angle {} and speed {} at frame {}'.format(angle, speed, self.frame_num))


        stim_set_tag = msg_dict['stimulus']['note']
        indices = self.stimuli_space.param_to_ridx([angle, speed, size, freq, center_x, center_y, contrast, shape], tag=stim_set_tag)
        # logger.info(f"{[angle, speed, size, freq, center_x, center_y, contrast, shape]}")
        # self.links['stim_queue'].put({self.frame_num:indices})
        self.links['stim_queue'].put({"frame": self.frame_num,"indices": indices,"tag": stim_set_tag, "stim_count": self.stim_count})
        self.acq_stim_queue_ts.append([self.frame_num, time.time()])
        self.stimmed.append([self.frame_num, angle, speed, size, freq, center_x, center_y, contrast, shape])
        