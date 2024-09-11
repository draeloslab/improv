import time
import threading
import yaml
import numpy as np
from pathlib import Path
import subprocess
from improv.actor import ManagedActor, Actor, Signal
from .front_end import CameraStreamWidget
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
    def setup(self):
        # store init
        self._getStoreInterface()

        self.stop_program = False
        self.start_program = False

        # load the configuration file
        source_folder = Path(__file__).resolve().parent.parent

        with open(f'{source_folder}/config/camera_config.yaml', 'r') as file:
            config = yaml.safe_load(file)

        cameras_config = config['active_cameras']
        camera_params = config['camera_params']

        self.frame_w = camera_params['resolution']['width'] # frame width
        self.frame_h = camera_params['resolution']['height'] # frame height
        
        self.num_cameras = len(cameras_config)
        self.camera_names = []
        self.camera_ids = []
        
        for camera in cameras_config:
            self.camera_names.append(camera['camera']['name'])
            self.camera_ids.append(camera['camera']['serial_id'])

        self.frame_rate_update = 60 # Update rate for the video stream
        self.frame_i = self.frame_rate_update

        logger.info(f"Video GUI setup completed")

    def getLastFrame(self, camera_id):
        frame_id = None

        # clear the queue
        while not self.links[f"camera{camera_id}_in"].empty():
            self.links[f"camera{camera_id}_in"].get_nowait()
            self.q_in.get_nowait()

        try:
            frame_id = self.links[f"camera{camera_id}_in"].get(timeout=0.1)
            pred_id = self.q_in.get()

            if frame_id is not None and pred_id is not None:
                frame = self.client.get(frame_id)
                predictionData = self.client.get(pred_id)
            else:
                frame = np.zeros((self.frame_h, self.frame_w, 3), dtype=np.uint8)
        except Exception as e:
            return np.zeros((self.frame_h, self.frame_w, 3), dtype=np.uint8)

        return frame, predictionData['prediction']

    def runStep(self): 
        pass

    def stop(self):
        self.stop_program = True
        logger.info(f"Video GUI stopped")