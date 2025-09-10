import time
import threading
import yaml
import numpy as np
import cv2
import queue
import logging
import traceback
import subprocess
from pathlib import Path
from improv.actor import ManagedActor, Actor, Signal
from .front_end import CameraStreamWidget
from PyQt5 import QtWidgets
import os
from collections import deque

# Set Qt backend before any imports
os.environ['QT_API'] = 'pyqt5'
os.environ['MPLBACKEND'] = 'Qt5Agg'

# Force matplotlib to use Qt5Agg backend before any Qt imports
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg

from PyQt5.QtWidgets import QApplication, QLabel, QWidget, QGridLayout
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QBrush, QFont

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Create a file handler
log_file = "video_screen.log"
file_handler = logging.FileHandler(log_file)
file_handler.setLevel(logging.DEBUG)

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
        self.num_buffers_rec = [0 for _ in range(self.num_cameras)] # num of buffers recorded by each camera
        self.num_buffers_progress = [0 for _ in range(self.num_cameras)] # num of buffers converted for each camera
        self.buffer_conv_completed = [False for _ in range(self.num_cameras)] # flag to indicate if the buffer conversion is completed
        
        self.num_cameras = len(cameras_config)
        self.camera_names = []
        self.camera_ids = []
        
        for camera in cameras_config:
            self.camera_names.append(camera['camera']['name'])
            self.camera_ids.append(camera['camera']['serial_id'])

        self.frame_rate_update = 60 # Update rate for the video stream
        self.frame_i = self.frame_rate_update
        self.frame_latencies =[]
        # self.pred_latencies = []
        self.frame_count = 0

        timestamp = time.strftime("%Y%m%d-%H%M")
        self.out_folder = Path(f"/home/chesteklab/predictions/{timestamp}")
        self.out_folder.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output folder set to {self.out_folder}")

        logger.info(f"Video GUI setup completed")


    def getLastFrame(self, camera_id):
        frame_id = None
        predictions = None  # Initialize predictions with a default value
        angle = None

        # Clear the frame queue for the specific camera
        # while not self.links[f"preds{camera_id}_in"].empty():
            #self.links[f"preds{camera_id}_in"].get_nowait()
        frame_start = time.perf_counter()    
        try:
            frame_id = self.links[f"images{camera_id}_in"].get(timeout=0.01)
            # frame_start = time.perf_counter()
            if frame_id is not None:
                # frame = self.client.get(frame_id)
                frame_enc = self.client.get(frame_id)

                # uncompressing the frame
                frame = cv2.imdecode(frame_enc, cv2.IMREAD_COLOR)
                self.frame_latencies.append(time.perf_counter())

            else:
                frame = np.zeros((self.frame_h, self.frame_w, 3), dtype=np.uint8)
                logger.debug('Was unable to grab frame!')
            # self.frame_latencies.append(time.perf_counter() - frame_start)
        except queue.Empty:
            frame = np.zeros((self.frame_h, self.frame_w, 3), dtype=np.uint8)
            # logger.debug(f'No frame available for camera {camera_id}')
        except KeyError:
            frame = np.zeros((self.frame_h, self.frame_w, 3), dtype=np.uint8)
            # logger.debug(f'No frame available for camera {camera_id}')
        except Exception:
            logger.error(f"Error getting frame for camera {camera_id}: {e}")
            logger.info(len(self.frame_latencies))
            logger.info(len(self.pred_latencies))
            logger.info(f"error on {camera_id} [frame: {frame_id}]")
            frame = np.zeros((self.frame_h, self.frame_w, 3), dtype=np.uint8)
            logger.debug(f'Exception getting frame for camera {camera_id}: {traceback.format_exc()}')
        # self.frame_latencies.append(time.perf_counter())

        try:
            element = self.links[f"preds{camera_id}_in"].get(timeout=0.01)
            # pred_start = time.perf_counter()

            # element = self.links[f"preds{camera_id}_in"].get(timeout=0.1)

            # frame_id = element[0]
            predictions = element[0]
            angle = element[1]
            logger.debug(f'Angle received: {angle}')

            # self.pred_latencies.append(time.perf_counter() - pred_start)
        except queue.Empty:
            # logger.debug(f'No prediction available for camera {camera_id}')
            pass
        except KeyError:
            pass
        except Exception:
            logger.error(f"Error getting frame for camera {camera_id}: {e}")
            logger.info(len(self.frame_latencies))
            logger.info(len(self.pred_latencies))
            logger.info(f"error on {camera_id} [frame: {frame_id}]")
            frame = np.zeros((self.frame_h, self.frame_w, 3), dtype=np.uint8)
            logger.debug(f'Unexpected error getting prediction for camera {camera_id}: {traceback.format_exc()}')
            # pass

        return frame,predictions,angle
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

        # # Increment frame counter
        # self.frame_count += 1

        # # Only get predictions for camera 0
        # predictions = None

        # try:
        #     # Use get with timeout to prevent blocking
        #     pred_id = self.q_in.get(timeout=0.01)
        #     pred_start = time.time()
        #     # logger.info(f"Pred Key received for camera 0: {pred_id}")

        #     if pred_id is not None:
        #         try:
        #             dlcGrab = self.client.get(pred_id)
        #             predictions = dlcGrab[0]
        #             dlcframe = dlcGrab[1]
        #             logger.info(f'Got predicitons:{predictions} and frame: {dlcframe}')
        #             if self.frame_count % 100 == 0:
        #                 logger.info(f'Avg pred latency: {1/np.mean(self.pred_latencies)}')
        #             self.pred_latencies.append(time.time() - pred_start)
        #             # logger.info(f'Frame length: {len(self.frame_latencies)}')
        #             # logger.info(f'Pred Length: {len(self.pred_latencies)}')
        #             # logger.info(f"Got prediction for camera 0")
        #         except Exception as e:
        #             logger.error(f"Could not get prediction data for camera 0: {e}")
        #             predictions = None
        # except queue.Empty:
        #     # No prediction data available
        #     predictions = None
        # except Exception as e:
        #     logger.error(f"Error getting prediction for camera 0: {e}")
        #     predictions = None

        # return frame, predictions

    def runStep(self): 
        self.start_program = True
        pass

    def stop(self):
        logger.info(f"{self.name}: Stopping Video GUI")
        self.stop_program = True
        # logger.info(f'End Frame length: {len(self.frame_latencies)}')
        # logger.info(f'end Pred Length: {len(self.pred_latencies)}')

        # try:
        #     np.save(self.out_folder / "vizframelatencies.npy", self.frame_latencies)
        #     np.save(self.out_folder / "vizpredictionslatencies.npy", self.pred_latencies)
        #     logger.info(f"{self.name}: Video GUI stopped")
        # except Exception:
        #     logger.info(f'Could not save latencies: {traceback.format_exc()}')