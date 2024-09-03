from improv.actor import Actor
import numpy as np
import logging
from dlclive import DLCLive
from pathlib import Path
import yaml
import time

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Processor(Actor):
    """ Applying DLC inference to each video frame
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def setup(self):
        """Initializes all class variables.

        self.name (string): name of the actor.
        self.frame (ObjectID): StoreInterface object id referencing data from the store.
        self.avg_list (list): list that contains averages of individual vectors.
        self.frame_num (int): index of current frame.
        """

        logger.info("Beginning setup for Processor")

         # load the configuration file
        source_folder = Path(__file__).resolve().parent.parent

        with open(f'{source_folder}/config.yaml', 'r') as file:
            config = yaml.safe_load(file)

        self.model_path = f'{source_folder}/DLCLive/' + config['model_path']
        self.resize = config['resize']
        self.name = "Processor"
        self.frame = None
        self.dlc_live = DLCLive(self.model_path, resize = self.resize, dynamic = (True, 0.9, 30))
        frame = np.random.rand(1080, 1920, 3)
        self.dlc_live.init_inference(frame)  #putting in a random frame to initialize the model
        self.predictions = []
        self.latencies = []
        self.frame_num = 1
        logger.info("Completed setup for Processor")

    def stop(self):
        """Trivial stop function for testing purposes."""
        self.done=True
        np.save("latencies.npy", self.latencies)
        np.save("predictions.npy", self.predictions)
        logger.info("Processor stopping")

    def runStep(self):

        
        frame = None
        try:
            start_time = time.perf_counter()

            frame = self.q_in.get()
            logger.info(f"Frame Key received: {frame}")

        except Exception as e:
            logger.error(f"Could not get frame! {e}")
            pass

        if frame is not None and self.frame_num is not None:
            self.done = False
            # start_time = time.perf_counter() #NOTE might want to put this where I get the key instead of here
            self.frame = self.client.get(frame)

            logger.info(f"Got frame: {self.frame.shape}")

            self.frame_num += 1

            # Perform inference
            # self.dlc_live.init_inference(self.frame)
            # start_time = time.perf_counter()
            prediction = self.dlc_live.get_pose(self.frame)
            self.latencies.append( time.perf_counter() - start_time)
            self.predictions.append(prediction)
            logger.info(f"Prediction: {prediction}")
            logger.info(f"Frame number: {self.frame_num}")  
            # logger.info(f"Latency: {latency:.4f} seconds")
            logger.info(f"Overall Average FPS: {1/np.mean(self.latencies)}")
