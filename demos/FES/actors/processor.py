from improv.actor import Actor
import numpy as np
import logging
from dlclive import DLCLive
from pathlib import Path
import yaml

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

        params = config['camera_params']

        self.model_path = params['model_path']

        self.name = "Processor"
        self.frame = None
        self.dlc_live = DLCLive(self.model_path)
        self.predictions = []
        self.frame_num = 1
        logger.info("Completed setup for Processor")

    def stop(self):
        """Trivial stop function for testing purposes."""
        self.done=True
        logger.info("Processor stopping")

    def runStep(self):

        try:
            frame = self.frame_in.get(timeout=0.001) #NOTE: might need to change timeout

        except Exception:
            logger.error("Could not get frame!")
            pass

        if frame is not None and self.frame_num is not None:
            self.done = False
            self.dlc_live.init_inference(frame) 
            prediction = self.dlc_live.get_pose(frame)
            # if self.store_loc:
            #     self.frame = self.client.getID(frame[0][0])
            # else:
            #     self.frame = self.client.get(frame)
            # avg = np.mean(self.frame[0])

            # logger.info(f"Average: {avg}")
            self.predictions.append(prediction)
            # logger.info(f"Overall Average: {np.mean(self.avg_list)}")
            logger.info(f"Frame number: {self.frame_num}")
            logger.info(f"Prediction: {prediction}")

            self.frame_num += 1
