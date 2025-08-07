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
from .kalmanfilter import KalmanFilterPredictor

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


            train_dir = Path("/home/chesteklab/improv/demos/FES/dlc-models-pytorch/iteration-2/manipulandum_pytorchMay13-trainset95shuffle1/train")
            pytorch_config_path = train_dir / "pytorch_config.yaml"
            snapshot_path = train_dir / "snapshot-best-010.pt"

            # for top-down models, otherwise None
            detector_snapshot_path = None

            # video and inference parameters
            max_num_animals = 1
            batch_size = 16
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
            self.kalman_filter = KalmanFilterPredictor(
                adapt=True,
                forward=0.002,
                fps=30,  
                nderiv=2,
                priors=[10, 10],
                initial_var=5,    
                process_var=5,     
                dlc_var=20,        
                lik_thresh=0.5     
            )

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
            self.dlc_latencies = []
            self.grab_latencies = []
            self.put_latencies = []            
            self.time_start = time.perf_counter()
            self.frame_num = 0
            self.frame_sentTime = 0
            self.frames_log = 200 # num frames after which to log
            # self.recent_predictions = [deque(maxlen=3) for _ in range(5)]  #want to keep this low to avoid lag
            self.recent_predictions = [None for _ in range(5)]
            self.alpha = 0.3 #Smoothing factor for EMA


            timestamp = time.strftime("%Y%m%d-%H%M")
            self.out_folder = Path(f"/home/chesteklab/predictions/{timestamp}")
            self.out_folder.mkdir(parents=True, exist_ok=True)
            logger.info(f"Output folder set to {self.out_folder}")
            logger.info("Completed setup for Processor")

    def stop(self):
        """Stop function for saving results and cleaning up."""
        if self.pred_active:
            self.done = True
            np.save(self.out_folder / "latencies.npy", self.latencies)
            np.save(self.out_folder / "predictions.npy", self.predictions)
            np.save(self.out_folder / "dlcLatencies.npy", self.dlc_latencies)
            np.save(self.out_folder / "grabLatencies.npy", self.grab_latencies)
            np.save(self.out_folder / "putLatencies.npy", self.put_latencies)
            logger.info("Predictions and latencies saved")
            logger.info("Processor stopping")

    def runStep(self):
        frame_id = None
        self.prediction = None
        angle = None
        smoothed_prediction = None

        try:
            frame_id = self.q_in.get()
            start_time = time.perf_counter()
            # logger.info(f"Frame Id received: {frame_id}")
        except Exception as e:
            logger.error(f"Could not get frame! {e}")
            return

        if frame_id is not None:
            self.done = False

            if self.pred_active:
                # retrieving the compressed frame from the storage
                # frame = self.client.get(frame_id)
                frame_enc = self.client.get(frame_id)
                # uncompressing the frame
                frame = cv2.imdecode(frame_enc, cv2.IMREAD_COLOR)

                self.frame_num += 1

                # Perform inference
                dlc_start = time.time()
                # self.prediction = self.dlc_live.get_pose(frame)
                raw_prediction = self.pose_runner.inference([frame])
                # Extract the bodyparts array from the prediction dictionary
                # The format is [{'bodyparts': array([[[x, y, likelihood], ...]])}]
                self.prediction = raw_prediction[0]['bodyparts'][0]  # Get the first (and only) frame's bodyparts
                # smoothed_prediction = self.kalman_filter.process(self.prediction, frame_time=dlc_start)
                smoothed_prediction = self.prediction

                # smoothed_prediction = np.zeros_like(self.prediction)
                # for i, point in enumerate(self.prediction):
                #     x, y, likelihood = point
                #     if self.recent_predictions[i] is None:
                #         self.recent_predictions[i] = (x,y)

                #     prev_x, prev_y = self.recent_predictions[i]
                #     ema_x = self.alpha * x + (1 - self.alpha) * prev_x
                #     ema_y = self.alpha * y + (1 - self.alpha) * prev_y


                #     self.recent_predictions[i] = (ema_x, ema_y)
                #     # Calculate the moving average for x and y
                #     # avg_x = np.mean([p[0] for p in self.recent_predictions[i]])
                #     # avg_y = np.mean([p[1] for p in self.recent_predictions[i]])
                #     smoothed_prediction[i, :2] = ema_x, ema_y
                #     smoothed_prediction[i, 2] = likelihood
                #     if likelihood < 0.3 and len(self.predictions) > 0:
                #         smoothed_prediction[i,:2] = self.predictions[-1][i,:2]

                self.predictions.append(smoothed_prediction) #TODO might want to also store the raw prediction

                # Only calculate angle if we have at least 3 bodyparts
                if len(smoothed_prediction) >= 3:
                    angle = self.calculateAngle(smoothed_prediction)
                else:
                    angle = None
                    logger.warning(f"Not enough bodyparts for angle calculation. Got {len(smoothed_prediction)}, need 3.")

                # logger.info(f"Angle: {angle}") 
                dlc_end = time.perf_counter()

                self.dlc_latencies.append(dlc_end - dlc_start)
                self.grab_latencies.append(dlc_end - start_time)

                if self.frame_num % self.frames_log == 0:
                    total_time = dlc_end - self.time_start                    

                    logger.info(f"Frame number: {self.frame_num}")
                    # logger.info(f"Prediction: {prediction}")
                    logger.info(f"Overall Average FPS: {round(self.frames_log / total_time,2)}")
                    logger.info(f'Camera Grab Time Avg latency: {np.mean(self.grab_latencies)}')
                    logger.info(f'Pure DLC Inference Time Avg latency: {np.mean(self.dlc_latencies)}')
                    logger.info(f'Put Time Avg latency: {np.mean(self.put_latencies)}')

                    self.time_start = time.perf_counter() # reset the timer

                # logger.info(f'sent on this frame{frame}')
                # logger.info('Put prediction and index dict in store')

            try:
                self.q_out.put([frame_id,smoothed_prediction, angle])
                logger.info(f"Sent frame_id: {frame_id}, predictions: {smoothed_prediction is not None}, angle: {angle} to video screen")

                if self.pred_active:
                    self.put_latencies.append(time.perf_counter() - dlc_end)
                    self.latencies.append(time.perf_counter() - start_time)
            except Exception as e:
                logger.error(f"--------------------------------Generator Exception: {e}")
                logger.error(traceback.format_exc())


    def calculateAngle(self,prediction):
        # Check if we have at least 3 points
        if len(prediction) < 3:
            logger.error(f"Cannot calculate angle: need 3 points, got {len(prediction)}")
            return None
            
        p2, p3, p4 = prediction[0, :2], prediction[1, :2], prediction[2, :2]
        # Define vectors from point 3 to points 2 and 4
        v3_to_2 = p2 - p3
        v3_to_4 = p4 - p3

        # Calculate dot product and determinant
        dot_product = np.dot(v3_to_2, v3_to_4)
        determinant = v3_to_2[0] * v3_to_4[1] - v3_to_2[1] * v3_to_4[0]

        # Calculate angle in degrees at point 3
        angle = np.degrees(np.arctan2(determinant, dot_product)) % 360
        return angle