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
from . import cpu_affinity
from .front_end5 import CameraStreamWidget
from PyQt5 import QtWidgets
import os
from collections import deque

# Set Qt backend before any imports
os.environ['QT_API'] = 'pyqt5'

from PyQt5.QtWidgets import QApplication, QLabel, QWidget, QGridLayout
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QBrush, QFont

from .run_paths import get_logger, run_folder

logger = get_logger(__name__, "video_screen.log", level=logging.DEBUG, handler_level=logging.INFO)

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
        # The GUI redraws on its own timer and was measured at ~106 us per
        # frame, so it does not belong on a P-core -- it only needs to keep up
        # with the eye, not with the FES loop.
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
        # self.num_cameras comes from the num_active_cameras kwarg (set per-yaml,
        # e.g. 1 for a single-camera run, 3 for latency_benchmarking.yaml's
        # current 3-camera wiring) -- NOT from len(cameras_config), which is
        # every camera camera_config.yaml *knows about* (hardware inventory),
        # not how many are wired up in this particular experiment. Getting
        # this wrong means the GUI tries to open images{N}_in/preds{N}_in
        # links that were never connected, for every camera in the building.
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

        self.out_folder = run_folder()
        logger.info(f"Output folder set to {self.out_folder}")

        logger.info(f"Video GUI setup completed")


    def getLastFrame(self, camera_id):
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

        # Clear the frame queue for the specific camera
        # while not self.links[f"preds{camera_id}_in"].empty():
            #self.links[f"preds{camera_id}_in"].get_nowait()
        self.videoStarts.append(time.time())
        frame_start = time.perf_counter()    
        try:
            msg = self.links[f"images{camera_id}_in"].get(timeout=0.01)
            frame_id = msg[0]
            camera_start = msg[1]
            # frame_start = time.perf_counter()
            if frame_id is not None:
                frame = self.client.get(frame_id)
                try:
                    if isinstance(frame, np.ndarray) and len(frame.shape) == 3:
                        pass
                    else:
                    # uncompressing the frame
                        frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)
                    self.frame_latencies.append(time.perf_counter()- frame_start)
                except:
                    logger.error(f'Suspect none in frame {frame}; camera id is {camera_id}')

            else:
                # frame = np.zeros((self.frame_h, self.frame_w, 3), dtype=np.uint8)
                logger.debug('Was unable to grab frame!')
            # self.frame_latencies.append(time.perf_counter() - frame_start)
        except queue.Empty:
            pass
            #frame = np.zeros((self.frame_h, self.frame_w, 3), dtype=np.uint8)
            # logger.debug(f'No frame available for camera {camera_id}')
        except KeyError:
            # frame = np.zeros((self.frame_h, self.frame_w, 3), dtype=np.uint8)
            # logger.debug(f'No frame available for camera {camera_id}')
            pass
        except Exception as e:
            logger.error(f"Error getting frame for camera {camera_id}: {e}")
            logger.info(len(self.frame_latencies))
            logger.info(len(self.pred_latencies))
            logger.info(f"error on {camera_id} [frame: {frame_id}]")
            frame = np.zeros((self.frame_h, self.frame_w, 3), dtype=np.uint8)
            logger.info(f'Exception getting frame for camera {camera_id}: {traceback.format_exc()}')
        # self.frame_latencies.append(time.perf_counter())

        try:
            pred_start = time.perf_counter()
            element = self.links[f"preds{camera_id}_in"].get(timeout=0.01)

            # Support [pred, angle], [pred, angle, camera_start, frame_num], and
            # [pred, angle, camera_start, frame_num, camera_num] formats.
            predictions = element[0]
            angle = element[1]
            if len(element) >= 5:
                camera_num = element[4]
            # logger.debug(f'Angle received: {angle}')

            self.pred_latencies.append(time.perf_counter() - pred_start)
        except queue.Empty:
            # logger.info(f'No prediction available for camera {camera_id}. Empty Queue.')
            pass
        except KeyError:
            # logger.info(f'No prediction available for camera {camera_id}. Key Error.')
            pass
        except Exception as e:
            logger.error(f"Error getting frame for camera {camera_id}: {e}")
            logger.info(len(self.frame_latencies))
            logger.info(len(self.pred_latencies))
            logger.info(f"error on {camera_id} [frame: {frame_id}]")
            frame = np.zeros((self.frame_h, self.frame_w, 3), dtype=np.uint8)
            logger.info(f'Unexpected error getting prediction for camera {camera_id}: {traceback.format_exc()}')
            # pass

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
        #             logger
            # np.save(self.out_folder / "vizframelatencies.npy", self.frame_latencies)
            # np.save(self.out_folder / "vizpredictionslatencies.npy", self.pred_latencies)
            # np.save(self.out_folder / "vizStarts.npy", self.videoStarts).info(f'Got predicitons:{predictions} and frame: {dlcframe}')
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

    def stopMe(self):
        logger.info(f"{self.name}: Stopping Video GUI")
        self.stop_program = True
        # logger.info(f'End Frame length: {len(self.frame_latencies)}')
        # logger.info(f'end Pred Length: {len(self.pred_latencies)}')

        try:
            np.save(self.out_folder / "vizframelatencies.npy", self.frame_latencies)
            np.save(self.out_folder / "vizpredictionslatencies.npy", self.pred_latencies)
            np.save(self.out_folder / "vizStarts.npy", self.videoStarts)
            logger.info(f"{self.name}: Video GUI stopped")
        except Exception:
            logger.info(f'Could not save latencies: {traceback.format_exc()}')