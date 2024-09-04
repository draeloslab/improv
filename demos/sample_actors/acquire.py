import time
import os
import h5py
import random
import numpy as np
from skimage.io import imread
import pathlib
from pathlib import Path
# from ast import literal_eval as make_tuple'
import ast
import zmq
import pickle
import re

from improv.actor import Actor

import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Classes: File Acquirer, Stim, Behavior, Tiff


class FileAcquirer(Actor):
    """Class to import data from files and output
    frames in a buffer, or discrete.
    """

    def __init__(self, *args, ip=None, ports=None, output=None, init_filename=None, filename=None, init_frame=60, framerate=30, **kwargs):
        super().__init__(*args, **kwargs)
        self.frame_num = 0
        self.data = None
        self.done = False
        self.flag = False
        self.initial_frame_num = init_frame
        self.init_filename = init_filename
        self.filename = filename
        self.framerate = 1 / framerate
        self.ip = ip
        self.ports = ports
        self.stim_count = 0

        self.output_folder = str(output)
        pathlib.Path(output).mkdir(exist_ok=True)
        pathlib.Path(output+'timing/').mkdir(exist_ok=True)
        
    def setup(self):
        """Get file names from config or user input
         Also get specified framerate, or default is 10 Hz
        Open file stream
        #TODO: implement more than h5 files
        """
        if os.path.exists(self.filename):
            n, ext = os.path.splitext(self.filename)[:2]
            if ext == ".h5" or ext == ".hdf5":
                with h5py.File(self.filename, "r") as file:
                    keys = list(file.keys())
                    data = file[keys[0]][()]
                    self.data = data[self.initial_frame_num:]

            f = h5py.File(self.init_filename, 'w', libver='earliest')
            f.create_dataset("default", data=data[:self.initial_frame_num])

        else:
            raise FileNotFoundError
        
        if os.path.exists(self.init_filename):
            pass
        else:
            raise FileNotFoundError

        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.SUB)
        # TODO: need to include an additional IP address for the microscope computer (so it can listen to both the microscope and pandastim)
        for port in self.ports:
            self.socket.connect("tcp://" + str(self.ip) + ":" + str(port))
            print('Connected to ' + str(self.ip) + ':' + str(port))
        self.socket.setsockopt(zmq.SUBSCRIBE, b'')

        self.total_times = []
        self.timestamp = []
        self.fullStimmsg = []
        self.stimmed = []
        self.stimF = []

    def stop(self):
        print("Done running Acquire, avg time per frame: ", np.mean(self.total_times))
        print("Acquire got through ", self.frame_num, " frames")
        # if not os._exists("output"):
        #     try:
        #         os.makedirs("output")
        #     except:
        #         pass
        # if not os._exists("output/timing"):frame:
        #         pass
        np.savetxt("output/timing/acquire_frame_time.txt", np.array(self.total_times))
        np.savetxt("output/timing/acquire_timestamp.txt", np.array(self.timestamp))

    def runStep(self):
        """While frames exist in location specified during setup,
        grab frame, save, put in store
        """
        t = time.time()

        if self.done:
            pass
        elif self.frame_num < len(self.data) * 5:
            frame = self.getFrame(self.frame_num % len(self.data))
            ## simulate frame-dropping
            # if self.frame_num > 1500 and self.frame_num < 1800:
            #     frame = None
            t = time.time()
            # id = self.client.put(frame, "acq_raw" + str(self.frame_num))
            id = self.client.put(frame)           
            t1 = time.time()
            self.timestamp.append([time.time(), self.frame_num])
            try:
                self.q_out.put([{str(self.frame_num): id}])
                self.frame_num += 1
                # also log to disk #TODO: spawn separate process here?
            except Exception as e:
                logger.error("Acquirer general exception: {}".format(e))

            time.sleep(self.framerate)  # pretend framerate
            self.total_times.append(time.time() - t)

        else:  # simulating a done signal from the source (eg, camera)
            logger.error("Done with all available frames: {0}".format(self.frame_num))
            self.data = None
            self.q_comm.put(None)
            self.done = True  # stay awake in case we get e.g. a shutdown signal
            # if self.saving:
            #    self.f.close()

        self.get_message()
    
    def get_message(self, timeout=0.001):
        msg = self.socket.recv_multipart()
        # logger.info('Raw pandastim message: {}'.format(msg))
        try:
            msg_dict, category = self._msg_unpacker(msg)
            # tag = msg_dict['type']
        except Exception as e: 
            logger.error('Weird format message {}'.format(e))
        
        if 'stim' in str(msg_dict):
            tag = 'pandastim'
        # if 'stim' in tag:
            if not self.stimF:
                logger.info('Receiving stimulus information')
                self.stimF = True
            self.fullStimmsg.append(msg)
            self._collect_stimulus(msg_dict, category)

    def getFrame(self, num):
        """Here just return frame from loaded data"""
        return self.data[num, :, :]

    def saveFrame(self, frame):
        """Save each frame via h5 dset"""
        self.dset[self.frame_num - 1] = frame
        self.f.flush()

    def _collect_stimulus(self, msg_dict, category):
        # sendtime = msg_dict['time']

        # category = str(msg_dict) #'motionOn' 
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

            if msg_dict['stimulus']['stim_name'] == 'moving_gray':
                angle = float(msg_dict['stimulus']['angle'])
                vel = float(msg_dict['stimulus']['velocity'])
                self.links['stim_queue'].put({self.frame_num:[angle, vel]})
                self.stimmed.append([self.frame_num, angle, vel])
                logger.info('Stimulus: Moving gratings angle {} with velocity {} at frame {}'.format(angle, vel, self.frame_num))

            elif msg_dict['stimulus']['stim_name'] == 'circle_radius':
            ## spots stim with Karina
                size = float(msg_dict['texture']['circle_radius'])
                vel = float(msg_dict['stimulus']['velocity'])
                self.links['stim_queue'].put({self.frame_num:[size, vel]})
                self.stimmed.append([self.frame_num, size, vel])
                logger.info('Stimulus: Circle radius {} with velocity {} at frame {}'.format(size, vel, self.frame_num))

            logger.info('Number of stimuli: {}'.format(self.stim_count))
            # self.stimsendtimes.append([sendtime])

    def _msg_unpacker(self, msg):

        # if len(msg) == 1:
        #     msg = msg[0]
        
        # # this is a lot of hardcoding
        msg_unpacked = pickle.loads(msg[0])
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


class StimAcquirer(Actor):
    """Class to load visual stimuli data from file
    and stream into the pipeline
    """

    def __init__(self, *args, param_file=None, filename=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.param_file = param_file
        self.filename = filename

    def setup(self):
        self.n = 0
        self.sID = 0
        if os.path.exists(self.filename):
            print("Looking for ", self.filename)
            n, ext = os.path.splitext(self.filename)[:2]
            if ext == ".txt":
                # self.stim = np.loadtxt(self.filename)
                self.stim = []
                f = np.loadtxt(self.filename)
                for _, frame in enumerate(f):
                    stiminfo = frame[0:2]
                    self.stim.append(stiminfo)
            else:
                logger.error("Cannot load file, possible bad extension")
                raise Exception

        else:
            raise FileNotFoundError

    def runStep(self):
        """Check for input from behavioral control"""
        if self.n < len(self.stim):
            # s = self.stim[self.sID]
            # self.sID+=1
            self.q_out.put({self.n: self.stim[self.n]})
        time.sleep(0.5)  # simulate a particular stimulus rate
        self.n += 1


class BehaviorAcquirer(Actor):
    """Actor that acquires information of behavioral stimulus
    during the experiment

    Current assumption is that stimulus is off or on, on has many types,
    and any change in stimulus is _un_associated with a frame number.
    TODO: needs to be associated with time, then frame number
    """

    def __init__(self, *args, param_file=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.param_file = param_file

    def setup(self):
        """Pre-define set of input stimuli"""
        self.n = 0  # our fake frame number here
        # TODO: Consider global frame_number in store...or signal from Nexus

        # TODO: Convert this to Config and load from there
        if self.param_file is not None:
            try:
                params_dict = None  # self._load_params_from_file(param_file)
            except Exception as e:
                logger.exception("File cannot be loaded. {0}".format(e))
        else:
            self.behaviors = [0, 1, 2, 3, 4, 5, 6, 7]  # 8 sets of input stimuli

    def runStep(self):
        """Check for input from behavioral control"""
        # Faking it for now.
        if self.n % 50 == 0:
            self.curr_stim = random.choice(self.behaviors)
            self.onoff = random.choice([0, 10])
            self.q_out.put({self.n: [self.curr_stim, self.onoff]})
            logger.info("Changed stimulus! {}".format(self.curr_stim))
        time.sleep(1)  # 0.068)
        self.n += 1


class FileStim(Actor):
    """Actor that acquires information of behavioral stimulus
    during the experiment from a file
    """

    def __init__(self, *args, File=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.file = File

    def setup(self):
        """Pre-define set of input stimuli"""
        self.n = 0  # our fake frame number here
        # TODO: Consider global frame_number in store...or signal from Nexus

        self.data = np.loadtxt(self.file)

    def runStep(self):
        """Check for input from behavioral control"""
        # Faking it for now.
        if self.n % 50 == 0 and self.n < self.data.shape[1] * 50:
            self.curr_stim = self.data[0][int(self.n / 50)]
            self.onoff = self.data[1][int(self.n / 50)]
            self.q_out.put({self.n: [self.curr_stim, self.onoff]})
            logger.info("Changed stimulus! {}".format(self.curr_stim))
        time.sleep(0.068)
        self.n += 1


class TiffAcquirer(Actor):
    """Loops through a TIF file."""

    def __init__(self, *args, filename=None, framerate=30, **kwargs):
        super().__init__(*args, **kwargs)
        self.filename = filename

        if not os.path.exists(filename):
            raise ValueError("TIFF file {} does not exist.".format(filename))
        self.imgs = None  # np.array(0)

        self.n_frame = 0
        self.fps = framerate

        self.t_per_frame = list()

    def setup(self):
        self.imgs = imread(self.filename)
        print(self.imgs.shape)

    def runStep(self):
        t0 = time.time()
        # id_store = self.client.put(
        #     self.imgs[self.n_frame], "acq_raw" + str(self.n_frame)
        # )
        id_store = self.client.put(self.imgs[self.n_frame])
        self.q_out.put([[id_store, str(self.n_frame)]])
        self.n_frame += 1

        time.sleep(1 / self.fps)

        self.t_per_frame.append(time.time() - t0)
