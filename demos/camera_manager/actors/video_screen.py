import time
import cv2
import threading
import yaml
import numpy as np
from pathlib import Path
import subprocess
from improv.actor import ManagedActor, Actor, Signal
from .camera_front_end import CameraStreamWidget
from PyQt5 import QtWidgets

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Create a file handler
log_file = "video_screen.log"
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
        logger.info("Running setup for " + self.name)

    def run(self):
        logger.info("Loading FrontEnd")        
        self.app = QtWidgets.QApplication([])
        self.viewer = CameraStreamWidget(self.visual, self.q_comm, self.q_sig)
        self.viewer.show()
        self.q_comm.put([Signal.ready()])
        self.visual.q_comm.put([Signal.ready()])
        self.app.exec_()
        logger.info("GUI ready")

class VideoScreen(ManagedActor):    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.num_cameras = kwargs['num_active_cameras']

    def setup(self):
        # store init
        self._getStoreInterface()

        self.start_program = False

        # load the configuration file
        source_folder = Path(__file__).resolve().parent.parent

        with open(f'{source_folder}/config/camera_config.yaml', 'r') as file:
            config = yaml.safe_load(file)

        camera_params = config['camera_params']

        self.frame_w = camera_params['resolution']['width'] # frame width
        self.frame_h = camera_params['resolution']['height'] # frame height

        self.num_buffers_rec = [0 for _ in range(self.num_cameras)] # num of buffers recorded by each camera
        self.num_buffers_progress = [0 for _ in range(self.num_cameras)] # num of buffers converted for each camera
        self.buffer_conv_completed = [False for _ in range(self.num_cameras)] # flag to indicate if the buffer conversion is completed

        logger.info(f"Video GUI setup completed")

    def get_last_frame(self, camera_id):
        frame_id = None

        # clear the queue
        while not self.links[f"camera{camera_id}_in"].empty():
            self.links[f"camera{camera_id}_in"].get_nowait()

        try:
            frame_id = self.links[f"camera{camera_id}_in"].get(timeout=0.1)

            if frame_id is not None:
                frame_enc = self.client.get(frame_id)

                # uncompressing the frame
                frame = cv2.imdecode(frame_enc, cv2.IMREAD_COLOR)
            else:
                frame = np.zeros((self.frame_h, self.frame_w, 3), dtype=np.uint8)
                frame[:,:] = [217, 30, 24]
        except Exception as e:
            frame = np.zeros((self.frame_h, self.frame_w, 3), dtype=np.uint8)

        return frame

    def start_buffer_conversion(self):
        """Function to start the buffer data conversion for each camera."""
        msg = {'type': 'video_conversion', 'value': True}
        self.links[f"msg_out"].put(msg)

        logger.info("buffer video_conversion message sent")

    def get_number_buffer_conversion(self):
        """Function to get the number of buffer files that need to be converted for each camera."""
        
        # Continue looping until all cameras have received their num_buffer_files
        while not all(count > 0 for count in self.num_buffers_rec):
            for camera_id in range(self.num_cameras):
                # Only attempt to get messages for cameras that haven't received num_buffer_files yet
                if self.num_buffers_rec[camera_id] == 0:
                    try:
                        msg = self.links[f"camera{camera_id}_msg_in"].get(timeout=0.25)
                    except:
                        msg = None
                    
                    if msg is not None:
                        if msg['type'] == 'num_buffer_files':
                            self.num_buffers_rec[camera_id] = msg['value']
                        elif msg['type'] == 'buffer_conv_progress':
                            self.num_buffers_progress[camera_id] = msg['value']
                        elif msg['type'] == 'buffer_conv_done':
                            self.buffer_conv_completed[camera_id] = True

            time.sleep(0.25)

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