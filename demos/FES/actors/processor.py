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

    def setup(self):
        """Initializes all class variables."""
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
        self.sentLatencies = []
        self.frame_num = 1
        self.frame_sentTime = 0

        timestamp = time.strftime("%Y%m%d-%H%M%S")
        self.out_folder = Path(f"/home/chesteklab/predictions/{timestamp}")
        self.out_folder.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output folder set to {self.out_folder}")
        logger.info("Completed setup for Processor")

    def stop(self):
        """Stop function for saving results and cleaning up."""
        self.done = True
        np.save(self.out_folder / "latencies.npy", self.latencies)
        np.save(self.out_folder / "predictions.npy", self.predictions)
        np.save(self.out_folder / "dlcLatencies.npy", self.dlcLatencies)
        logger.info("Predictions and latencies saved")
        logger.info("Processor stopping")

    def runStep(self):
        frame = None
        try:
            start_time = time.perf_counter()

            frame = self.q_in.get()
            # logger.info(f"Frame Key received: {frame}")

        except Exception as e:
            logger.error(f"Could not get frame! {e}")
            return

        if frame is not None and self.frame_num is not None:
            self.done = False
            self.frame,self.frame_time = self.client.get(frame)

            # logger.info(f"Got frame: {self.frame.shape}")

            self.frame_num += 1

            # Perform inference
            dlcStart = time.perf_counter()
            self.sentLatencies.append(dlcStart - self.frame_time)
            prediction = self.dlc_live.get_pose(self.frame)
            self.latencies.append(time.perf_counter() - start_time)
            self.dlcLatencies.append(time.perf_counter() - dlcStart)
            self.predictions.append(prediction)
            if self.frame_num % 100 == 0:
                logger.info(f"Prediction: {prediction}")
                logger.info(f"Frame number: {self.frame_num}")  
                logger.info(f"Overall Average latency: {np.mean(self.latencies)}")
                logger.info(f" Average DLC inference: {np.mean(self.dlcLatencies)}")
                logger.info(f"Average Frame sent time: {np.mean(self.sentLatencies)}")
            send_time = time.perf_counter()
            data_id = self.client.put({'prediction': prediction, 'frame': frame, 'timestamp': send_time})
            # logger.info('Put prediction and index dict in store')
            try:
                self.q_out.put(data_id)
                # logger.info("Sent message on")
            except Exception as e:
                logger.error(f"--------------------------------Generator Exception: {e}")