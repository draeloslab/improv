import numpy as np
import logging
import yaml
import time
import traceback
import cv2
# from dlclive import DLCLive
from pathlib import Path
from improv.actor import Actor
from collections import deque
# from .dlcProcessor import IndexAngles
# from deeplabcut.pose_estimation_pytorch import Task
# from deeplabcut.pose_estimation_pytorch.apis.analyze_videos import video_inference
from deeplabcut.pose_estimation_pytorch.config import read_config_as_dict
from deeplabcut.pose_estimation_pytorch.apis.utils import get_inference_runners
# from .kalmanfilter import KalmanFilterPredictor
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

    def setup(self):
        """Initializes all class variables."""

        if self.pred_active:
            logger.info("Beginning setup for Processor")

            # load the configuration file
            source_folder = Path(__file__).resolve().parent.parent

            with open(f'{source_folder}/config.yaml', 'r') as file:
                config = yaml.safe_load(file)

            self.resize = config['resize']


            train_dir = Path(config['model_path'])
            # train_dir = Path('/home/chesteklab/Desktop/napierNewManipulandum-jake-2025-11-04/dlc-models-pytorch/iteration-2/napierNewManipulandumNov4-trainset95shuffle1/train')
            # train_dir = Path("/home/chesteklab/Desktop/dlc-models-pytorch/iteration-2/manipulandum_pytorchMay13-trainset95shuffle1/train")
            # train_dir = Path("/home/chesteklab/Desktop/human-manipulandum-jake-2025-10-29/dlc-models-pytorch/iteration-0/human-manipulandumOct29-trainset95shuffle2/train")
            pytorch_config_path = train_dir / "pytorch_config.yaml"
            snapshot_path = train_dir / "snapshot-best-010.pt"

            # for top-down models, otherwise None
            detector_snapshot_path = None

            # video and inference parameters
            max_num_animals = 1
            batch_size = 1
            detector_batch_size = 8

            # read model configuration
            model_cfg = read_config_as_dict(pytorch_config_path)

            self.pose_runner, detector_runner = get_inference_runners(
                model_config=model_cfg,
                snapshot_path=snapshot_path,
                max_individuals=max_num_animals,
                batch_size=batch_size,
                detector_batch_size=detector_batch_size,
                detector_path=detector_snapshot_path,
            )

            # Initializing Kalman Filter with smoother parameters
            # self.kalman_filter = KalmanFilterPredictor(
            #     adapt=True,
            #     forward=0.002,
            #     fps=30,  
            #     nderiv=2,
            #     priors=[1, 1],
            #     initial_var=10,    
            #     process_var=1,     
            #     dlc_var=10,        
            #     lik_thresh=0.5     
            # )

            # self.model_path = f'{source_folder}/DLCLive/' + config['model_path']
            self.resize = config['resize']
            self.name = "Processor"
            self.frame = None
            # dlc_proc = IndexAngles()
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
            self.angle_queue = deque(maxlen=5)  # to store last 5 angles for smoothing
            # self.recent_predictions = [deque(maxlen=3) for _ in range(5)]  #want to keep this low to avoid lag
            self.recent_predictions = [None for _ in range(5)]
            self.alpha = config['alpha']
            # self.alpha = 0.6 #Smoothing factor for EMA
            self.interp_thresh = config['threshold']
            # self.interp_thresh = 0 #threshold below which to use last known good position
            self.prev_angle = None
            self.smoothed_prediction = None


            timestamp = time.strftime("%Y%m%d-%H%M")
            string  = config['output_path']
            self.out_folder = Path(f"{string}/{timestamp}")
            self.out_folder.mkdir(parents=True, exist_ok=True)
            logger.info(f"Output folder set to {self.out_folder}")
            logger.info(f"Using alpha: {self.alpha} and interp_thresh: {self.interp_thresh} and resize: {self.resize}")
            logger.info("Completed setup for Processor")

    def stop(self):
        """Stop function for saving results and cleaning up."""
        if self.pred_active:
            self.done = True
            np.save(self.out_folder / "latencies.npy", self.latencies)
            np.save(self.out_folder / "predictions.npy", self.predictions)
            np.save(self.out_folder / "startLatencies.npy", self.start_time)
            np.save(self.out_folder / "dlcLatencies.npy", self.dlc_latencies)
            np.save(self.out_folder / "latenciesFull.npy", self.latenciesFull)

            # np.save(self.out_folder / "grabLatencies.npy", self.grab_latencies)
            # np.save(self.out_folder / "putLatencies.npy", self.put_latencies)
            logger.info("Predictions and latencies saved")
        logger.info(f"Processor {self.name} stopped")

    def runStep(self):
        frame_id = None
        self.prediction = None
        angle = None
        smoothed_prediction = None
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
                    kalman_time = time.time()
                    # self.prediction = self.dlc_live.get_pose(frame)
                    frame = cv2.resize(frame, (int(frame.shape[1] * self.resize), int(frame.shape[0] * self.resize)))
                    # Convert BGR to RGB for the PyTorch model (trained with RGB images)
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    raw_prediction = self.pose_runner.inference([frame])  # this needs to be switched back to just frame for camera input
                    self.dlc_latencies.append(time.perf_counter() - dlc_start)
                    # Extract the bodyparts array from the prediction dictionary
                    # The format is [{'bodyparts': array([[[x, y, likelihood], ...]])}]
                    self.prediction = raw_prediction[0]['bodyparts'][0]  # Get the first (and only) frame's bodyparts
                    # logger.info(f"Raw prediction: {self.prediction}")
                    # logger.info(f' Shape of prediction: {self.prediction.shape}')
                    # logger.info(f' Prediction {self.prediction}')
                    # logger.info(f"Time {kalman_time}")
                    # smoothed_prediction = self.kalman_filter.process(self.prediction, frame_time=kalman_time)
                    # logger.info(f"Smoothed prediction: {smoothed_prediction.shape}")
                    # logger.info(f"Smoothed prediction: {smoothed_prediction}")
                    # Apply exponential moving average smoothing to predictions
                    smoothed_prediction= self.prediction
                    # smoothed_prediction = np.zeros_like(self.prediction)
                    
                    # for i, point in enumerate(self.prediction):
                    #     x, y, likelihood = point
                        
                    #     # If this is the first prediction for this bodypart, use it directly
                    #     if self.recent_predictions[i] is None:
                    #         smoothed_x, smoothed_y = x, y
                    #     # If confidence is low, use the last known good position
                    #     elif likelihood < self.interp_thresh:
                    #         smoothed_x, smoothed_y = self.recent_predictions[i]
                    #     # Otherwise, apply exponential moving average smoothing
                    #     else:
                    #         prev_x, prev_y = self.recent_predictions[i]
                    #         smoothed_x = self.alpha * x + (1 - self.alpha) * prev_x
                    #         smoothed_y = self.alpha * y + (1 - self.alpha) * prev_y
                        
                    #     # Store the smoothed position for next frame
                    #     self.recent_predictions[i] = (smoothed_x, smoothed_y)
                    #     smoothed_prediction[i] = [smoothed_x, smoothed_y, likelihood]
                    
                    # Save prediction for analysis
                    self.predictions.append(smoothed_prediction)

                    # Only calculate angle if we have at least 3 bodyparts
                    if len(smoothed_prediction) >= 3:
                        angle = self.calculateAngle(smoothed_prediction)
                    else:
                        angle = None
                        logger.warning(f"Not enough bodyparts for angle calculation. Got {len(smoothed_prediction)}, need 3.")

                    #Angle Smoothing
                    self.angle_queue.append(angle)
                    smoothed_angle = np.mean(self.angle_queue) if len(self.angle_queue) > 0 else angle

                    # Apply sudden jump detection on the smoothed angle
                    if self.prev_angle is not None and np.abs(smoothed_angle - self.prev_angle) > 5:
                        smoothed_angle = self.prev_angle  # ignore sudden large jumps
                    self.prev_angle = smoothed_angle

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
                    # # Log latency even on error
                    # if self.pred_active:
                    #     self.latencies.append(time.perf_counter())
                    # return
                self.latencies.append(time.perf_counter() - self.start_perf)
                
                
                try:
                    self.q_out.put([smoothed_prediction, smoothed_angle])

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