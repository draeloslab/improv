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

# TODO: write actor that sends files through ZMQ to acquire_zmq.py

class FileAcquirer(Actor):

    def __init__(self, *args, ip=None, port=None, output=None, init_filename=None, filename=None, init_frame=60, framerate=15, **kwargs):
        super().__init__(*args, **kwargs)
        print("init")
        self.ip = ip
        self.done=False
        self.flag= False
        self.port = port
        self.frame_num = 0
        self.stim_count = 0
        self.initial_frame_num = init_frame     # Number of frames for initialization
        self.init_filename = init_filename
        self.filename = filename
        self.framerate = 1 / framerate 
        
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

            # f = h5py.File(self.init_filename, 'w', libver='earliest')
            # f.create_dataset("default", data=data[:self.initial_frame_num])

        else:
            raise FileNotFoundError(f"{self.filename}")
        
        if os.path.exists(self.init_filename):
            pass
        else:
            raise FileNotFoundError(f"{self.init_filename}")

        context = zmq.Context()
        self._socket = context.socket(zmq.PUB)
        send_IP =  self.ip
        send_port = self.port
        self._socket.bind('tcp://' + str(send_IP)+":"+str(send_port))
        # self._socket.setsockopt(zmq.SUBSCRIBE, b'')


    def stop(self):
        logger.info('Sending files through ZMQ stopping procedure --')
        

        # logger.info('Acquisition complete, avg time per frame: {}'.format(np.mean(self.total_times)))
        # logger.info('Acquire got through {} frames'.format(self.frame_num))

    def runStep(self):
        """While frames exist in location specified during setup,
        grab frame, save, send to acquire_zmq.py
        """
        t = time.time()

        if self.done:
            pass
        elif self.frame_num < len(self.data) * 5:
            frame = self.getFrame(self.frame_num % len(self.data))
            t = time.time()
            # try:
                # self.q_out.put([{str(self.frame_num): id}])
                
            self._socket.send_pyobj(dict({
                "type": "img_data",
                "data": frame,
                "timestamp": str(dt.now())
            }))
            # self.timestamp.append([time.time(), self.frame_num])
            self.frame_num += 1
            # except Exception as e:
            #     logger.error("Image message not sent {}".format(e))

            time.sleep(self.framerate)  # pretend framerate
            # self.total_times.append(time.time() - t)

        else:  # simulating a done signal from the source (eg, camera)
            logger.info('Done with available frames, starting again')
            # logger.error("Done with all available frames: {0}".format(self.frame_num))
            self.frame_num = 0
            # self.data = None
            # # self.q_comm.put(None)
            # self.done = True  # stay awake in case we get e.g. a shutdown signals

    def getFrame(self, num):
        """Here just return frame from loaded data"""
        return self.data[num, :, :]