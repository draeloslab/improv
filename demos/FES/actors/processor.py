from improv.actor import Actor
import numpy as np
import logging
from dlclive import DLCLive
from pathlib import Path
import yaml
import time
import traceback

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

            self.model_path = f'{source_folder}/DLCLive/' + config['model_path']
            self.resize = config['resize']
            self.name = "Processor"
            self.frame = None
            self.dlc_live = DLCLive(self.model_path, resize=self.resize, dynamic=(True, 0.9, 30))
            frame = np.random.rand(1080, 1920, 3)
            self.dlc_live.init_inference(frame)  # putting in a random frame to initialize the model
            self.predictions = []
            self.latencies = []
            self.dlc_latencies = []
            self.grab_latencies = []
            self.put_latencies = []            
            self.time_start = time.perf_counter()
            self.frame_num = 0
            self.frame_sentTime = 0
            self.frames_log = 200 # num frames after which to log

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
        frame_key = None
        prediction = None

        try:
            frame_key = self.q_in.get()
            start_time = time.perf_counter()
            # logger.info(f"Frame Key received: {frame_key}")
        except Exception as e:
            logger.error(f"Could not get frame! {e}")
            return

        if frame_key is not None:
            self.done = False

            if self.pred_active:
                self.frame = self.client.get(frame_key)
                self.frame_num += 1

                # Perform inference
                dlc_start = time.perf_counter()
                prediction = self.dlc_live.get_pose(self.frame)
                dlc_end = time.perf_counter()

                self.predictions.append(prediction)
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

                # logger.info(f'sent on this frame{self.frame}')
                # logger.info('Put prediction and index dict in store')

            try:
                self.q_out.put([frame_key,prediction])

                if self.pred_active:
                    self.put_latencies.append(time.perf_counter() - dlc_end)
                    self.latencies.append(time.perf_counter() - start_time)
            except Exception as e:
                logger.error(f"--------------------------------Generator Exception: {e}")
                logger.error(traceback.format_exc())