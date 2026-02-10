import os
# Limit numpy/BLAS threading to avoid contention with PyTorch in multiprocessing
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import numpy as np
import logging
import yaml
import time
import traceback
import cv2
import copy
import torch
# from dlclive import DLCLive
from pathlib import Path
from improv.actor import Actor
from collections import deque
# from .dlcProcessor import IndexAngles
# from deeplabcut.pose_estimation_pytorch import Task
# from deeplabcut.pose_estimation_pytorch.apis.analyze_videos import video_inference
from deeplabcut.pose_estimation_pytorch.config import read_config_as_dict
from deeplabcut.pose_estimation_pytorch.apis.utils import get_inference_runners
from .kalmanfilter import KalmanFilterPredictor
from improv.store import ObjectNotFoundError

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Create a file handler
log_file = "processor.log"
file_handler = logging.FileHandler(log_file)
file_handler.setLevel(logging.INFO)

# Create a formatter and set it for the handler
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)

# Add the handler to the logger
logger.addHandler(file_handler)


class Processor(Actor):
    """ Applying DLC inference to each video frame
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.pred_active = kwargs['pred_active']
        self.camera_num = kwargs.get('camera_num', 0)  # Default to camera 0 if not specified

    def setup(self):
        """Initializes all class variables."""
        self._getStoreInterface()

        if self.pred_active:
            logger.info("Beginning setup for Processor")

            # Limit PyTorch threading to avoid contention with numpy in multiprocessing
            torch.set_num_threads(1)

            # load the configuration file
            source_folder = Path(__file__).resolve().parent.parent

            with open(f'{source_folder}/config.yaml', 'r') as file:
                config = yaml.safe_load(file)

            self.resize = config['resize']

            # Select model path and snapshot based on camera number
            model_path_key = f'model_path_{self.camera_num}'
            model_snapshot_key = f'model_snapshot_{self.camera_num}'
            
            train_dir = Path(config[model_path_key])
            pytorch_config_path = train_dir / "pytorch_config.yaml"
            snapshot_path = train_dir / config[model_snapshot_key]

            # for top-down models, otherwise None
            detector_snapshot_path = None

            # video and inference parameters
            max_num_animals = 1
            batch_size = 1
            detector_batch_size = 8

            # read model configuration
            model_cfg = read_config_as_dict(pytorch_config_path)
            
            logger.info(f"Camera {self.camera_num}: Using model from {train_dir}")
            logger.info(f"Camera {self.camera_num}: Using snapshot {config[model_snapshot_key]}")

            self.pose_runner, detector_runner = get_inference_runners(
                model_config=model_cfg,
                snapshot_path=snapshot_path,
                max_individuals=max_num_animals,
                batch_size=batch_size,
                detector_batch_size=detector_batch_size,
                detector_path=detector_snapshot_path,
            )
        

            # Initializing Kalman Filter with smoother parameters
            self.kalman_filter = KalmanFilterPredictor(
                adapt=False,
                forward=0.002,
                fps=30,  
                nderiv=2,
                priors=[1, 1],
                initial_var=10,    
                process_var=1,     
                dlc_var=10,        
                lik_thresh=0.2     
            )
            logger.info(f'Kalman filter initialized for camera {self.camera_num}')

            self.resize = config['resize']
            self.name = "Processor"
            self.frame = None
            # self.dlc_live = DLCLive(self.model_path, resize=self.resize, dynamic=(True, 0.9, 30))
            # frame = np.random.rand(1080, 1920, 3)
            # self.dlc_live.init_inference(frame)  # putting in a random frame to initialize the model
            self.predictions = []
            self.latencies = []
            self.latenciesFull = []
            self.start_time = []
            # self.dlc_latencies = []
            # self.grab_latencies = []
            # self.put_latencies = []    
            self.dlc_latencies = []        
            self.time_start = time.perf_counter()
            self.frame_num = 0
            self.frame_sentTime = 0
            self.frames_log = 200 # num frames after which to log
            self.angle_queue = deque(maxlen=10)  # to store last 5 angles for smoothing
            self.recent_predictions = [None for _ in range(5)]
            self.alpha = config['alpha']
            self.interp_thresh = config['threshold']
            self.prev_angle = None
            self.smoothed_prediction = None


            date = time.strftime("%Y%m%d")
            timestamp = time.strftime("%Y%m%d-%H%M")
            string = config['output_path']
            self.out_folder = Path(f"{string}/{date}/{timestamp}")
            self.out_folder.mkdir(parents=True, exist_ok=True)
            logger.info(f"Output folder set to {self.out_folder}")
            logger.info(f"Using alpha: {self.alpha} and interp_thresh: {self.interp_thresh} and resize: {self.resize}")
            logger.info("Completed setup for Processor")

    def stop(self):
        """Stop function for saving results and cleaning up."""
        if self.pred_active:
            self.done = True
            # np.save(self.out_folder / "latencies.npy", self.latencies)
            np.save(self.out_folder / f"predictions_cam{self.camera_num}.npy", self.predictions)
            np.save(self.out_folder / f"latencies_cam{self.camera_num}.npy", self.latencies)
            np.save(self.out_folder / f"startLatencies_cam{self.camera_num}.npy", self.start_time)
            np.save(self.out_folder / f"dlcLatencies_cam{self.camera_num}.npy", self.dlc_latencies)
            np.save(self.out_folder / f"latenciesFull_cam{self.camera_num}.npy", self.latenciesFull)
            # np.save(self.out_folder / "startLatencies.npy", self.start_time)
            # np.save(self.out_folder / "dlcLatencies.npy", self.dlc_latencies)
            # np.save(self.out_folder / "latenciesFull.npy", self.latenciesFull)

            # np.save(self.out_folder / "grabLatencies.npy", self.grab_latencies)
            # np.save(self.out_folder / "putLatencies.npy", self.put_latencies)
            logger.info("Predictions and latencies saved")
        logger.info(f"Processor {self.name} stopped")

    def runStep(self):
        frame_id = None
        self.prediction = None
        angle = None
        smoothed_prediction = None
        smoothed_angle = None  # Initialize smoothed_angle
        # start_time = time.perf_counter()
        if self.pred_active:
            self.start_time.append(time.time())
            
            try:
                frame_id = self.q_in.get(timeout=0.01)
                self.start_perf = time.perf_counter()
                # start_time = time.perf_counter()

                # logger.info(f"Frame Id received: {frame_id}")
            except Exception as e:
                pass
                # logger.error(f"Could not get message!  {e}")
                # Log latency even on error
                
            if frame_id is not None:
                self.done = False

                try:
                    # dlc_start = time.perf_counter()
                    frame = self.client.get(frame_id)
                    if isinstance(frame, np.ndarray) and len(frame.shape) == 3:
                        pass
                    else:
                        # uncompressing the frame
                        frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)

                    self.frame_num += 1

                    # Perform inference
                    dlc_start = time.perf_counter()
                    # kalman_time = time.time()
                    # self.prediction = self.dlc_live.get_pose(frame)
                    frame = cv2.resize(frame, (int(frame.shape[1] * self.resize), int(frame.shape[0] * self.resize)))
                    # if self.camera_num == 2:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                    
                    # Quick check: log color channel info on first frame
                    if self.frame_num % 100 == 0:
                        avg_channels = np.mean(frame, axis=(0, 1))
                        logger.debug(f"Frame shape: {frame.shape}, Average channel values: {avg_channels}")
                        # Check if likely BGR (OpenCV) or RGB format
                        # In most natural images, blue channel has lower values than red
                        if avg_channels[0] < avg_channels[2]:
                            logger.info(f"Frame from camera {self.camera_num} appears to be BGR format (channel 0 < channel 2)")
                        else:
                            logger.info(f"Frame from camera {self.camera_num} appears to be RGB format (channel 0 >= channel 2)")

                    # Convert BGR to RGB for the PyTorch model (trained with RGB images)
                    # frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    try:
                        raw_prediction = self.pose_runner.inference([frame])  # this needs to be switched back to just frame for camera input
                    except Exception as e:
                        logger.error(f"DLC inference error: {e}")
                        logger.error(traceback.format_exc())
                        # Use previous prediction as fallback if available
                        if self.smoothed_prediction is not None:
                            smoothed_prediction = self.smoothed_prediction
                            smoothed_angle = self.prev_angle
                        self.latencies.append(time.perf_counter() - self.start_perf)
                        self.q_out.put([smoothed_prediction, smoothed_angle])
                        logger.info(f"Processor camera {self.camera_num}: Sent fallback prediction and angle {smoothed_prediction}, {smoothed_angle}")
                        return
                    self.dlc_latencies.append(time.perf_counter() - dlc_start)
                    # Extract the bodyparts array from the prediction dictionary
                    # The format is [{'bodyparts': array([[[x, y, likelihood], ...]])}]
                    self.prediction = raw_prediction[0]['bodyparts'][0]  # Get the first (and only) frame's bodyparts
                    # logger.info(f" Got prediction from model for camera {self.camera_num}")
                    # logger.info(f"Shape of prediction: {self.prediction.shape}")
                    # logger.info(f"Prediction data type: {type(self.prediction)}")
                    # Select bodyparts based on camera number
                    if self.camera_num == 0:
                        # Use first 4 bodyparts for camera 0
                        self.prediction = self.prediction[:4]
                    elif self.camera_num == 2:
                        # Use only the last bodypart for camera 2
                        # self.prediction = self.prediction[-1:]
                        self.prediction = self.prediction[:1]
                    
                    # logger.info(f"Raw prediction: {self.prediction}")
                    # logger.info(f' Shape of prediction: {self.prediction.shape}')
                    # logger.info(f"Type of prediction: {type(self.prediction)}")
                    # logger.info(f' Prediction {self.prediction}')
                    # logger.info(f"Time {kalman_time}")
                    try:
                        # assert True==False
                        smoothed_prediction = self.kalman_filter.process(self.prediction)
                    except Exception as e:
                        logger.error(f"Kalman filter processing error: {e}")
                        logger.error(traceback.format_exc())
                        smoothed_prediction = self.prediction  # fallback to raw prediction on error
                    # logger.info(f"Smoothed prediction: {smoothed_prediction.shape}")
                    # logger.info(f"Smoothed prediction: {smoothed_prediction}")
                    
                    # Save prediction for analysis
                    self.predictions.append(smoothed_prediction)
                    self.smoothed_prediction = smoothed_prediction  # cache for fallback on inference errors

                    # Only calculate angle if we have at least 3 bodyparts
                    if len(smoothed_prediction) >= 3:
                        angle = self.calculateAngle(smoothed_prediction)
                        
                        #Angle Smoothing
                        self.angle_queue.append(angle)
                        smoothed_angle = np.mean(self.angle_queue) if len(self.angle_queue) > 0 else angle

                        # Apply sudden jump detection on the smoothed angle
                        if self.prev_angle is not None and np.abs(smoothed_angle - self.prev_angle) > 50:
                            smoothed_angle = self.prev_angle  # ignore sudden large jumps
                        self.prev_angle = smoothed_angle
                    else:
                        self.angle_queue.append(smoothed_prediction[0][0])  # Just treat the x value as angle for queue
                        smoothed_angle = np.mean(self.angle_queue) if len(self.angle_queue) > 0 else angle

                        # Apply sudden jump detection on the smoothed angle
                        if self.prev_angle is not None and np.abs(smoothed_angle - self.prev_angle) > 50:
                            smoothed_angle = self.prev_angle  # ignore sudden large jumps
                        self.prev_angle = smoothed_angle
                        # angle = None
                        # smoothed_angle = None
                        # logger.warning(f"Not enough bodyparts for angle calculation. Got {len(smoothed_prediction)}, need 3.")

                    dlc_end = time.perf_counter()

                    if self.frame_num % self.frames_log == 0:
                        total_time = dlc_end - self.time_start                    
                        logger.info(f"Frame number: {self.frame_num}")
                        logger.info(f"Overall Average FPS: {round(self.frames_log / total_time,2)}")
                        self.time_start = time.perf_counter() # reset the timer

                except ObjectNotFoundError:
                    logger.error("Processor: Frame unavailable from store, dropping")
                    # Log latency even on error
                    # if self.pred_active:
                    #     self.latencies.append(time.perf_counter())
                    # return
                except Exception as e:
                    logger.error(f"Processing error: {e}")
                    logger.error(traceback.format_exc())
                    # # Log latency even on error
                    # if self.pred_active:
                    #     self.latencies.append(time.perf_counter())
                    # return
                self.latencies.append(time.perf_counter() - self.start_perf)
                
                
                try:
                    self.q_out.put([smoothed_prediction, smoothed_angle])
                    # logger.info(f"Processor camera {self.camera_num}: Sent prediction and angle {smoothed_prediction}, {smoothed_angle}")

                except Exception as e:
                    logger.error(f"Processor Exception: {e}")
                    logger.error(traceback.format_exc())
                    # Log latency even on error
                    # if self.pred_active:
                    #     self.latencies.append(time.perf_counter())
                    # return
            
                # Log latency for successful processing
                self.latenciesFull.append(time.perf_counter() - self.start_perf)

            # self.latencies.append(time.perf_counter())
        else:
            pass

    def calculateAngle(self,prediction):
        # Check if we have at least 3 points
        if len(prediction) < 3:
            logger.error(f"Cannot calculate angle: need 3 points, got {len(prediction)}")
            return None

        p2, p3, p4 = prediction[1, :2], prediction[2, :2], prediction[3, :2]
        #  DIP=0, PIP=1, MCP=2, Wrist=3, currently getting angle at MCP
        # Define vectors from point 3 to points 2 and 4
        v3_to_2 = p2 - p3
        v3_to_4 = p4 - p3

        # Calculate dot product and determinant
        dot_product = np.dot(v3_to_2, v3_to_4)
        determinant = v3_to_2[0] * v3_to_4[1] - v3_to_2[1] * v3_to_4[0]

        # Calculate angle in degrees at point 3
        angle = np.degrees(np.arctan2(determinant, dot_product)) % 360
        return angle