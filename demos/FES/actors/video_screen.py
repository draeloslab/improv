import time
import threading
import yaml
import numpy as np
from pathlib import Path
import subprocess
from improv.actor import ManagedActor, Actor, Signal
from .front_end import CameraStreamWidget
from PyQt5 import QtWidgets
import queue
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
        self.frame_latencies =[]
        self.pred_latencies = []
        self.frame_count = 0

        timestamp = time.strftime("%Y%m%d-%H%M")
        self.out_folder = Path(f"/home/chesteklab/predictions/{timestamp}")
        self.out_folder.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output folder set to {self.out_folder}")

        logger.info(f"Video GUI setup completed")

    def getLastFrame(self, camera_id):
        frame_id = None

        # Clear the frame queue for the specific camera
        while not self.links[f"camera{camera_id}_in"].empty():
            self.links[f"camera{camera_id}_in"].get_nowait()

        try:
            frame_id = self.links[f"camera{camera_id}_in"].get(timeout=0.1)
            frame_start = time.time()
            frame = self.client.get(frame_id) if frame_id is not None else np.zeros((self.frame_h, self.frame_w, 3), dtype=np.uint8)
            if self.frame_count % 100 == 0:
                logger.info(f'Avg frame latency: {1/np.mean(self.frame_latencies)}')
            self.frame_latencies.append(time.time() - frame_start)
        except Exception as e:
            logger.error(f"Error getting frame for camera {camera_id}: {e}")
            frame = np.zeros((self.frame_h, self.frame_w, 3), dtype=np.uint8)

        # Increment frame counter
        self.frame_count += 1

        # Only get predictions for camera 0
        predictions = None
        if camera_id == 2:
            try:
                # Use get with timeout to prevent blocking
                pred_id = self.q_in.get(timeout=0.01)
                pred_start = time.time()
                # logger.info(f"Pred Key received for camera 0: {pred_id}")

                if pred_id is not None:
                    try:
                        predictions = self.client.get(pred_id)
                        if self.frame_count % 100 == 0:
                            logger.info(f'Avg pred latency: {1/np.mean(self.pred_latencies)}')
                        self.pred_latencies.append(time.time() - pred_start)
                        # logger.info(f"Got prediction for camera 0")
                    except Exception as e:
                        logger.error(f"Could not get prediction data for camera 0: {e}")
                        predictions = None
            except queue.Empty:
                # No prediction data available
                predictions = None
            except Exception as e:
                logger.error(f"Error getting prediction for camera 0: {e}")
                predictions = None

        return frame, predictions

    def runStep(self): 
        pass

    def stop(self):
        self.stop_program = True
        np.save(self.out_folder / "vizframelatencies.npy", self.frame_latencies)
        np.save(self.out_folder / "vizpredictionslatencies.npy", self.pred_latencies)
        logger.info(f"Video GUI stopped")