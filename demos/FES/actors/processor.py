from improv.actor import Actor
import numpy as np
import logging
from dlclive import DLCLive
from pathlib import Path
import yaml
import time

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
            self.dlcLatencies = []
            self.grabLatencies = []
            self.putLatencies = []
            self.frame_num = 1
            self.frame_sentTime = 0

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
            np.save(self.out_folder / "dlcLatencies.npy", self.dlcLatencies)
            np.save(self.out_folder / "grabLatencies.npy", self.grabLatencies)
            np.save(self.out_folder / "putLatencies.npy", self.putLatencies)
            logger.info("Predictions and latencies saved")
            logger.info("Processor stopping")

    def runStep(self):
        frame = None
        try:
            frame_key = self.q_in.get()
            start_time = time.time()
            # logger.info(f"Frame Key received: {frame}")
        except Exception as e:
            logger.error(f"Could not get frame! {e}")
            return

        if frame is not None and self.frame_num is not None:
            self.done = False

            if self.pred_active:
                self.frame = self.client.get(frame_key)

                # logger.info(f"Got frame: {self.frame.shape}")

                self.frame_num += 1

                # Perform inference
                self.grabLatencies.append(time.time() -start_time)
                dlcStart = time.time()
                prediction = self.dlc_live.get_pose(self.frame)
                postInf = time.time()
                self.dlcLatencies.append(time.time()- dlcStart)
                self.predictions.append(prediction)
                if self.frame_num % 200 == 0:
                    logger.info(f"Frame number: {self.frame_num}")
                    logger.info(f"Prediction: {prediction}")
                    logger.info(f"Overall Average FPS: {1/np.mean(self.latencies)}")
                    logger.info(f'Camera Grab Time Avg FPS: {np.mean(self.grabLatencies)}')
                    logger.info(f'Pure DLC Inference Time Avg FPS: {1/np.mean(self.dlcLatencies)}')
                    logger.info(f'Put Time Avg FPS: {np.mean(self.putLatencies)}')

                data_id = self.client.put([frame_key, prediction])
                logger.info(f'sent on this frame{self.frame}')
                # logger.info('Put prediction and index dict in store')
                try:
                    self.q_out.put(data_id)
                    self.putLatencies.append(time.time() - postInf)
                    self.latencies.append(time.time() - start_time)
                    # logger.info("Sent message on")
                except Exception as e:
                    logger.error(f"--------------------------------Generator Exception: {e}")
            else:
                data_id = self.client.put([frame_key, np.zeros(1)])