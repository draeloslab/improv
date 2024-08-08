from improv.actor import Actor
import numpy as np
import logging
import cv2
import time
from pathlib import Path
import yaml



logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Generator(Actor):
    """Sample actor to generate data to pass into a sample processor.

    Intended for use along with sample_processor.py.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def setup(self):
        logger.info("Beginning setup for Generator")

         # load the configuration file
        source_folder = Path(__file__).resolve().parent.parent

        with open(f'{source_folder}/config.yaml', 'r') as file:
            config = yaml.safe_load(file)

        params = config['camera_params']

        self.video_path = params['video_path']
        self.cap = None
        self.frame_interval = 1.0 / params['fps']
        self.name = "Generator"
        self.frame_num = 1

        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            logger.error("Error opening video file")
            return 
        logger.info("Completed setup for Generator")

    def stop(self):
        """Save current randint vector to a file."""

        logger.info("Generator stopping")
        if self.cap:
            self.cap.release()
        return 0

    def runStep(self):

        if self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                logger.info("End of video")
                self.stop()
                return

            try:
                if self.store_loc:
                    data_id = self.client.put(frame, str(f"Frame: {self.frame_num}"))
                    self.q_out.put([[data_id, str(self.frame_num)]])
                else:
                    data_id = self.client.put(frame)
                    self.q_out.put(data_id)

                self.frame_num += 1

            except Exception as e:
                logger.error(f"--------------------------------Generator Exception: {e}")

            time.sleep(self.frame_interval)

        else:
            logger.error("Video capture not opened")

