import os
import time
import cv2
import threading
import yaml
import numpy as np
from pathlib import Path
import subprocess
from improv.actor import ManagedActor, Actor, Signal
from .offline_converter_front_end import OfflineConversionWidget
from PyQt5 import QtWidgets

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Create a file handler
log_file = "offline_conversion.log"
file_handler = logging.FileHandler(log_file)
file_handler.setLevel(logging.INFO)

# Create a formatter and set it for the handler
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)

# Add the handler to the logger
logger.addHandler(file_handler)

class Visual(Actor):
    def setup(self, visual):
        self.visual = visual
        self.visual.setup()

    def run(self):
        logger.info("Loading FrontEnd")        
        self.app = QtWidgets.QApplication([])
        self.viewer = OfflineConversionWidget(self.visual, self.q_comm, self.q_sig)
        self.viewer.show()
        self.q_comm.put([Signal.ready()])
        self.visual.q_comm.put([Signal.ready()])
        self.app.exec_()
        logger.info("GUI ready")

class ConversionScreen(ManagedActor):
    def setup(self):
        # store init
        self._getStoreInterface()

        self.start_program = False

        # load the configuration file
        source_folder = Path(__file__).resolve().parent.parent
        home_dir = os.path.expanduser('~')

        with open(f'{source_folder}/config/camera_config.yaml', 'r') as file:
            config = yaml.safe_load(file)

        cameras_config = config['active_cameras']
        camera_params = config['camera_params']

        self.num_cameras = len(cameras_config)

        self.num_buffers_rec = [0 for _ in range(self.num_cameras)] # num of buffers recorded by each camera
        self.num_buffers_progress = [0 for _ in range(self.num_cameras)] # num of buffers converted for each camera
        self.buffer_conv_completed = [False for _ in range(self.num_cameras)] # flag to indicate if the buffer conversion is completed

        # load the video configuration params
        with open(f'{source_folder}/config/video_config.yaml', 'r') as file:
            video_config = yaml.safe_load(file)

        raw_chunks_path = video_config['raw_chunks_path']

        self.default_video_folder = f"{home_dir}/{raw_chunks_path}"

        logger.info(f"Video GUI setup completed")

    def start_buffer_conversion(self, buffer_path):
        """Function to start the buffer data conversion for each camera."""

        msg = {'type': 'buffer_folder', 'value': buffer_path}
        self.links[f"msg_out"].put(msg)

        time.sleep(0.5)

        msg = {'type': 'video_conversion', 'value': True}
        self.links[f"msg_out"].put(msg)
        logger.info("buffer video_conversion message sent")

    def get_default_video_path(self):
        """Function to get the default folder path for the buffer data conversion."""
        return self.default_video_folder

    def get_number_buffer_conversion(self):
        """Function to get the number of buffer files that need to be converted for each camera."""
        
        msg_received = [False for _ in range(self.num_cameras)]

        # Continue looping until all cameras have received their num_buffer_files
        while not all(msg for msg in msg_received):
            for camera_id in range(self.num_cameras):
                # Only attempt to get messages for cameras that haven't received num_buffer_files yet
                if self.num_buffers_rec[camera_id] == 0:
                    try:
                        msg = self.links[f"camera{camera_id}_msg_in"].get(timeout=0.25)
                    except:
                        msg = None
                    
                    if msg is not None:
                        if msg['type'] == 'num_buffer_files':
                            msg_received[camera_id] = True
                            self.num_buffers_rec[camera_id] = msg['value']
                        elif msg['type'] == 'buffer_conv_progress':
                            self.num_buffers_progress[camera_id] = msg['value']
                        elif msg['type'] == 'buffer_conv_done':
                            self.buffer_conv_completed[camera_id] = True

            time.sleep(0.5)

        return self.num_buffers_rec

    def check_buffer_conversion_progress(self):
        """Function to check the progress of the camera buffer data conversion."""
        
        # Continue looping until all cameras have received their num_buffer_files
        for camera_id in range(self.num_cameras):
            try:
                msg = self.links[f"camera{camera_id}_msg_in"].get(timeout=0.5)
            except:
                msg = None
            
            if msg is not None:
                if msg['type'] == 'buffer_conv_progress':
                    self.num_buffers_progress[camera_id] = msg['value']
                elif msg['type'] == 'buffer_conv_done':
                    self.buffer_conv_completed[camera_id] = True

        return self.num_buffers_progress

    def runStep(self): 
        pass

    def stop(self):
        pass