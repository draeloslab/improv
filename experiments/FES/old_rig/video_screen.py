"""improv actor pair behind the 2D GUI (old_rig/front_end5.CameraStreamWidget)."""
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
from actors import cpu_affinity
from .front_end5 import CameraStreamWidget
from PyQt5 import QtWidgets
import os
from collections import deque

# Set Qt backend before any imports
os.environ['QT_API'] = 'pyqt5'

from PyQt5.QtWidgets import QApplication, QLabel, QWidget, QGridLayout
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QBrush, QFont

from actors.run_paths import get_logger, run_folder

logger = get_logger(__name__, "video_screen.log", level=logging.DEBUG, handler_level=logging.INFO)

class Visual(Actor):
    """improv's GUI actor for the 2D graphs: owns the Qt application and the CameraStreamWidget window."""

    def setup(self, visual):
        """visual: the VideoScreen actor the window reads frames and predictions from."""
        self.visual = visual
        self.visual.setup()
        logger.info("Running setup for " + self.name)

    def run(self):
        """Start Qt, show the window, tell improv the GUI is ready, then run the Qt event loop."""
        logger.info("Loading FrontEnd")        
        self.app = QtWidgets.QApplication([])
        self.viewer = CameraStreamWidget(self.visual, self.q_comm, self.q_sig)
        self.viewer.show()
        self.q_comm.put([Signal.ready()])
        self.visual.q_comm.put([Signal.ready()])
        self.app.exec_()
        logger.info("GUI ready")

class VideoScreen(ManagedActor):
    """Feeds the 2D GUI: newest frame per camera (images{i}_in), newest [prediction, angle] per camera (preds{i}_in).
    Also drives the offline buffer -> mp4 conversion from the GUI button."""

    def __init__(self, *args, **kwargs):
        """Graph kwargs: num_active_cameras."""
        super().__init__(*args, **kwargs)
        
        self.num_cameras = kwargs['num_active_cameras']

    def setup(self):
        """Pin to the E-cores and set up the frame / prediction fetch logs."""
        # The GUI redraws on its own timer (~0.1 ms per frame), so it does not need a P-core.
        cpu_affinity.pin_actor(cpu_affinity.BACKGROUND, label="VideoScreen")

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

        # Frames actually flowing through the store are at stream_resolution
        # (TIS downscales in GStreamer before the store) -- fall back to the
        # native `resolution` if stream_resolution isn't configured.
        stream_res = camera_params.get('stream_resolution', camera_params['resolution'])
        self.frame_w = stream_res['width'] # frame width
        self.frame_h = stream_res['height'] # frame height
        # self.num_cameras is the graph's num_active_cameras (cameras wired in this graph), not
        # len(cameras_config) (every camera camera_config.yaml knows about).
        self.num_buffers_rec = [0 for _ in range(self.num_cameras)] # num of buffers recorded by each camera
        self.num_buffers_progress = [0 for _ in range(self.num_cameras)] # num of buffers converted for each camera
        self.buffer_conv_completed = [False for _ in range(self.num_cameras)] # flag to indicate if the buffer conversion is completed

        self.camera_names = []
        self.camera_ids = []

        for camera in cameras_config:
            self.camera_names.append(camera['camera']['name'])
            self.camera_ids.append(camera['camera']['serial_id'])

        fps_str = str(camera_params.get('fps', 60))
        if '/' in fps_str:
            num, den = fps_str.split('/')
            self.frame_rate_update = int(int(num) / int(den))
        else:
            self.frame_rate_update = int(fps_str)
        self.frame_i = self.frame_rate_update
        self.frame_latencies =[]
        self.videoStarts = []
        self.pred_latencies = []
        self.frame_count = 0
        self.frames_skipped = {}   # camera_id -> frames dropped because the GUI was behind

        self.out_folder = run_folder()
        logger.info(f"Output folder set to {self.out_folder}")

        logger.info(f"Video GUI setup completed")


    def getLastFrame(self, camera_id):
        """(frame, predictions, angle) for one camera; any of them None if nothing new arrived."""
        frame_id = None
        predictions = None  # Initialize predictions with a default value
        angle = None
        frame = None
        # Real physical camera identity, from the prediction message's 5th
        # element (see processor.py). Falls back to the slot index (camera_id)
        # if a message doesn't carry it -- slot index is only a stand-in, since
        # it's just wiring order in the yaml and can differ from camera_num
        # (e.g. Processor1 can be camera_num=3 while wired to preds1_in).
        camera_num = camera_id

        self.videoStarts.append(time.time())
        frame_start = time.perf_counter()    
        try:
            # Show the newest frame and drop the backlog, so the display never falls behind.
            link = self.links[f"images{camera_id}_in"]
            msg = link.get(timeout=0.003)
            while True:
                try:
                    msg = link.get_nowait()
                    self.frames_skipped[camera_id] = self.frames_skipped.get(camera_id, 0) + 1
                except queue.Empty:
                    break
            frame_id = msg[0]
            if frame_id is not None:
                frame = self.client.get(frame_id)
                try:
                    if not (isinstance(frame, np.ndarray) and len(frame.shape) == 3):
                        frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)
                    self.frame_latencies.append(time.perf_counter()- frame_start)
                except (cv2.error, TypeError):
                    logger.error(f'Suspect none in frame {frame}; camera id is {camera_id}')

            else:
                logger.debug('Was unable to grab frame!')
        except (queue.Empty, KeyError):
            pass
        except Exception as e:
            logger.error(f"Error getting frame for camera {camera_id}: {e}")
            logger.info(len(self.frame_latencies))
            logger.info(len(self.pred_latencies))
            logger.info(f"error on {camera_id} [frame: {frame_id}]")
            frame = np.zeros((self.frame_h, self.frame_w, 3), dtype=np.uint8)
            logger.info(f'Exception getting frame for camera {camera_id}: {traceback.format_exc()}')

        try:
            pred_start = time.perf_counter()
            link = self.links[f"preds{camera_id}_in"]
            element = link.get(timeout=0.003)
            while True:                      # latest prediction only, same reason as the frames above
                try:
                    element = link.get_nowait()
                except queue.Empty:
                    break

            # Support [pred, angle], [pred, angle, camera_start, frame_num], and
            # [pred, angle, camera_start, frame_num, camera_num] formats.
            predictions = element[0]
            angle = element[1]
            if len(element) >= 5:
                camera_num = element[4]

            self.pred_latencies.append(time.perf_counter() - pred_start)
        except (queue.Empty, KeyError):
            pass
        except Exception as e:
            logger.error(f"Error getting frame for camera {camera_id}: {e}")
            logger.info(len(self.frame_latencies))
            logger.info(len(self.pred_latencies))
            logger.info(f"error on {camera_id} [frame: {frame_id}]")
            frame = np.zeros((self.frame_h, self.frame_w, 3), dtype=np.uint8)
            logger.info(f'Unexpected error getting prediction for camera {camera_id}: {traceback.format_exc()}')

        return frame, predictions, angle, camera_num

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
                    except (queue.Empty, KeyError):
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
        for camera_id in range(self.num_cameras):
            try:
                msg = self.links[f"camera{camera_id}_msg_in"].get(timeout=0.5)
            except (queue.Empty, KeyError):
                msg = None
            
            if msg is not None:
                if msg['type'] == 'buffer_conv_progress':
                    self.num_buffers_progress[camera_id] = msg['value']
                elif msg['type'] == 'buffer_conv_done':
                    self.buffer_conv_completed[camera_id] = True

        return self.num_buffers_progress

    def runStep(self):
        """Nothing per step: the GUI pulls on its own timer."""
        self.start_program = True

    def stopMe(self):
        """Save the GUI's fetch latencies."""
        logger.info(f"{self.name}: Stopping Video GUI")
        self.stop_program = True

        try:
            np.save(self.out_folder / "vizframelatencies.npy", self.frame_latencies)
            np.save(self.out_folder / "vizpredictionslatencies.npy", self.pred_latencies)
            np.save(self.out_folder / "vizStarts.npy", self.videoStarts)
            logger.info(f"{self.name}: Video GUI stopped")
        except Exception:
            logger.info(f'Could not save latencies: {traceback.format_exc()}')