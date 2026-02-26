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
                adapt=True,
                forward=0.002,
                fps=30,  
                nderiv=2,
                priors=[1, 1],  #[1e5, 1e5]
                initial_var=10,    
                process_var=1,     
                dlc_var=10,        
                lik_thresh=0.6     
            )
            logger.info(f'Kalman filter initialized for camera {self.camera_num}')

            self.resize = config['resize']
            self.name = "Processor"
            self.frame = None
            self.predictions = []
            self.raw_predictions = []
            self.frame_num = 0
            self.frame_sentTime = 0
            self.frames_log = 200 # num frames after which to log
            self.angle_queue = deque(maxlen=10)  # to store last 5 angles for smoothing
            self.recent_predictions = [None for _ in range(5)]
            self.alpha = config['alpha']
            self.interp_thresh = config['threshold']
            self.prev_angle = None
            self.smoothed_prediction = None

            # --- Timing logs ---
            # Wall-clock timestamp when each frame starts processing (time.time())
            self.timestamps = []
            # Frame numbers received from generator, for cross-actor correlation
            self.frame_nums_received = []
            # Generator timestamps received, to compute queue wait time
            self.generator_timestamps = []
            # Per-step breakdowns (perf_counter durations in seconds)
            self.queue_wait_latencies = []   # time spent waiting on q_in.get
            self.store_get_latencies = []    # client.get
            self.resize_latencies = []       # cv2.resize
            self.dlc_latencies = []          # DLC inference only
            self.kalman_latencies = []       # Kalman filter
            self.postprocess_latencies = []  # angle calc + smoothing
            self.queue_put_latencies = []    # q_out.put
            # Total latencies (perf_counter)
            self.total_latencies = []        # queue get → q_out put (full runStep work)
            # Angles sent out, for synchronization with sender
            self.angles_sent = []
            # Queue drain tracking — how many frames were skipped each step
            self.frames_dropped = []         # number of stale frames skipped per runStep

            self.time_start = time.perf_counter()


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
            cam = self.camera_num

            # Predictions and raw data
            np.save(self.out_folder / f"predictions_cam{cam}.npy", self.predictions)
            np.save(self.out_folder / f"raw_predictions_cam{cam}.npy", self.raw_predictions)
            np.save(self.out_folder / f"angles_sent_cam{cam}.npy", self.angles_sent)

            # Timestamps for cross-actor correlation
            np.save(self.out_folder / f"proc_timestamps_cam{cam}.npy", self.timestamps)
            np.save(self.out_folder / f"proc_frame_nums_cam{cam}.npy", self.frame_nums_received)
            np.save(self.out_folder / f"proc_gen_timestamps_cam{cam}.npy", self.generator_timestamps)

            # Per-step latency breakdowns
            np.save(self.out_folder / f"proc_queue_wait_cam{cam}.npy", self.queue_wait_latencies)
            np.save(self.out_folder / f"proc_store_get_cam{cam}.npy", self.store_get_latencies)
            np.save(self.out_folder / f"proc_resize_cam{cam}.npy", self.resize_latencies)
            np.save(self.out_folder / f"proc_dlc_cam{cam}.npy", self.dlc_latencies)
            np.save(self.out_folder / f"proc_kalman_cam{cam}.npy", self.kalman_latencies)
            np.save(self.out_folder / f"proc_postprocess_cam{cam}.npy", self.postprocess_latencies)
            np.save(self.out_folder / f"proc_queue_put_cam{cam}.npy", self.queue_put_latencies)
            np.save(self.out_folder / f"proc_total_cam{cam}.npy", self.total_latencies)

            # Queue drain tracking
            np.save(self.out_folder / f"proc_frames_dropped_cam{cam}.npy", self.frames_dropped)

            # Legacy file names for backward compatibility
            np.save(self.out_folder / f"latencies_cam{cam}.npy", self.total_latencies)
            np.save(self.out_folder / f"startLatencies_cam{cam}.npy", self.timestamps)
            np.save(self.out_folder / f"dlcLatencies_cam{cam}.npy", self.dlc_latencies)

            logger.info(f"Processor cam{cam}: Predictions and latencies saved to {self.out_folder}")
        logger.info(f"Processor {self.name} stopped")

    def runStep(self):
        frame_id = None
        self.prediction = None
        angle = None
        smoothed_prediction = None
        smoothed_angle = None  # Initialize smoothed_angle

        if self.pred_active:
            # --- Step 1: Queue drain — skip to the NEWEST frame ---
            t_queue_wait = time.perf_counter()
            try:
                msg = self.q_in.get(timeout=0.01)
                # ----------------------------
                # Drain any additional frames — keep only the newest
                dropped = 0
                while True:
                    try:
                        newer_msg = self.q_in.get_nowait()
                        dropped += 1
                        msg = newer_msg
                    except Exception:
                        break  # queue is empty, msg holds the newest frame

                if dropped > 0:
                    logger.debug(f"Cam {self.camera_num}: Skipped {dropped} stale frame(s) to reduce latency")
                #------------------------------

                # Support both old [data_id, timestamp] and new [data_id, timestamp, frame_num] formats
                if len(msg) == 3:
                    frame_id, camera_start, gen_frame_num = msg
                else:
                    frame_id, camera_start = msg
                    gen_frame_num = -1
                t_got = time.perf_counter()
                self.queue_wait_latencies.append(t_got - t_queue_wait)
                self.frames_dropped.append(dropped)
            except Exception as e:
                # No frame available, nothing to process
                return
                
            if frame_id is not None:
                self.done = False
                step_start = time.perf_counter()
                now = time.time()
                self.timestamps.append(now)
                self.frame_nums_received.append(gen_frame_num)
                self.generator_timestamps.append(camera_start)

                try:
                    # --- Step 2: Store get ---
                    t0 = time.perf_counter()
                    frame = self.client.get(frame_id)
                    if isinstance(frame, np.ndarray) and len(frame.shape) == 3:
                        pass
                    else:
                        # uncompressing the frame
                        frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)
                    self.store_get_latencies.append(time.perf_counter() - t0)

                    self.frame_num += 1

                    # --- Step 3: Resize ---
                    t0 = time.perf_counter()
                    frame = cv2.resize(frame, (int(frame.shape[1] * self.resize), int(frame.shape[0] * self.resize)))
                    self.resize_latencies.append(time.perf_counter() - t0)

                    # Quick check: log color channel info periodically
                    if self.frame_num % 100 == 0:
                        avg_channels = np.mean(frame, axis=(0, 1))
                        logger.debug(f"Frame shape: {frame.shape}, Average channel values: {avg_channels}")
                        if avg_channels[0] < avg_channels[2]:
                            logger.info(f"Frame from camera {self.camera_num} appears to be BGR format (channel 0 < channel 2)")
                        else:
                            logger.info(f"Frame from camera {self.camera_num} appears to be RGB format (channel 0 >= channel 2)")

                    # --- Step 4: DLC Inference ---
                    t0 = time.perf_counter()
                    try:
                        raw_prediction = self.pose_runner.inference([frame])
                    except Exception as e:
                        logger.error(f"DLC inference error: {e}")
                        logger.error(traceback.format_exc())
                        # Use previous prediction as fallback if available
                        if self.smoothed_prediction is not None:
                            smoothed_prediction = self.smoothed_prediction
                            smoothed_angle = self.prev_angle
                        self.dlc_latencies.append(time.perf_counter() - t0)
                        self.total_latencies.append(time.perf_counter() - step_start)
                        self.angles_sent.append(smoothed_angle if smoothed_angle is not None else np.nan)
                        self.q_out.put([smoothed_prediction, smoothed_angle, camera_start, gen_frame_num])
                        logger.info(f"Processor camera {self.camera_num}: Sent fallback prediction and angle")
                        return
                    self.dlc_latencies.append(time.perf_counter() - t0)

                    # Extract the bodyparts array from the prediction dictionary
                    self.prediction = raw_prediction[0]['bodyparts'][0]
                    # Select bodyparts based on camera number
                    if self.camera_num == 0:
                        self.prediction = self.prediction[:4]
                    elif self.camera_num == 2:
                        self.prediction = self.prediction[:1]
                    
                    self.raw_predictions.append(self.prediction.copy())

                    # --- Step 5: Kalman filter ---
                    t0 = time.perf_counter()
                    try:
                        assert False
                        smoothed_prediction = self.kalman_filter.process(self.prediction, frame_time=camera_start)
                    except Exception as e:
                        logger.error(f"Kalman filter processing error: {e}")
                        logger.error(traceback.format_exc())
                        smoothed_prediction = self.prediction  # fallback to raw prediction on error
                    self.kalman_latencies.append(time.perf_counter() - t0)
                    
                    # Save prediction for analysis
                    self.predictions.append(smoothed_prediction)
                    self.smoothed_prediction = smoothed_prediction

                    # --- Step 6: Post-processing (angle calculation + smoothing) ---
                    t0 = time.perf_counter()
                    if len(smoothed_prediction) >= 3:
                        angle = self.calculateAngle(smoothed_prediction)
                        
                        #Angle Smoothing
                        self.angle_queue.append(angle)
                        smoothed_angle = np.median(self.angle_queue) if len(self.angle_queue) > 0 else angle

                        # Apply sudden jump detection on the smoothed angle
                        if self.prev_angle is not None and np.abs(smoothed_angle - self.prev_angle) > 500000:
                            smoothed_angle = self.prev_angle  # ignore sudden large jumps
                        self.prev_angle = smoothed_angle
                    else:
                        self.angle_queue.append(smoothed_prediction[0][0])  # Just treat the x value as angle for queue
                        smoothed_angle = np.median(self.angle_queue) if len(self.angle_queue) > 0 else angle

                        # Apply sudden jump detection on the smoothed angle
                        if self.prev_angle is not None and np.abs(smoothed_angle - self.prev_angle) > 5000000:
                            smoothed_angle = self.prev_angle  # ignore sudden large jumps
                        self.prev_angle = smoothed_angle
                    self.postprocess_latencies.append(time.perf_counter() - t0)

                    if self.frame_num % self.frames_log == 0:
                        dlc_end = time.perf_counter()
                        total_time = dlc_end - self.time_start                    
                        logger.info(f"Frame number: {self.frame_num}")
                        logger.info(f"Overall Average FPS: {round(self.frames_log / total_time,2)}")
                        self.time_start = time.perf_counter()

                except ObjectNotFoundError:
                    logger.error("Processor: Frame unavailable from store, dropping")
                    return
                except Exception as e:
                    logger.error(f"Processing error: {e}")
                    logger.error(traceback.format_exc())
                    return

                # --- Step 7: Queue put (send to downstream) ---
                t0 = time.perf_counter()
                try:
                    smoothed_angle = smoothed_angle/self.resize if self.camera_num == 2 else smoothed_angle
                    # Pass along camera_start and frame_num so Sender can compute true end-to-end
                    self.q_out.put([smoothed_prediction, smoothed_angle, camera_start, gen_frame_num])
                except Exception as e:
                    logger.error(f"Processor Exception: {e}")
                    logger.error(traceback.format_exc())
                self.queue_put_latencies.append(time.perf_counter() - t0)
            
                self.total_latencies.append(time.perf_counter() - step_start)
                self.angles_sent.append(smoothed_angle if smoothed_angle is not None else np.nan)

        else:
            pass

    def calculateAngle(self,prediction):
        # Check if we have at least 3 points
        if len(prediction) < 3:
            logger.error(f"Cannot calculate angle: need 3 points, got {len(prediction)}")
            return None

        p2, p3, p4 = prediction[0,:2], prediction[2, :2], prediction[3, :2]
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