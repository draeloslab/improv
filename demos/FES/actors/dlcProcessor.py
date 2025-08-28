import numpy as np
from pathlib import Path
import yaml
import logging

from dlclive import Processor



class IndexAngles(Processor):
    def __init__(self):

        super().__init__()

        # load the configuration file
        source_folder = Path(__file__).resolve().parent.parent

        with open(f'{source_folder}/config.yaml', 'r') as file:
            config = yaml.safe_load(file)


        self.threshold = config['threshold']


    def process(self, pose, **kwargs):
        ### bodyparts
        # 0. DIP
        # 1. PIP
        # 2. MCP
        # 3. Wrist
        # 4. Forearm

        # Extract (x, y) coordinates of points 2, 3, and 4
        p2, p3, p4 = pose[1, :2], pose[2, :2], pose[3, :2]

        logger.info(f"p2: {p2.shape}, p3: {p3.shape}, p4: {p4.shape}")

        # Define vectors from point 3 to points 2 and 4
        v3_to_2, v3_to_4 = p2 - p3, p4 - p3

        logger.info(f"v3_to_2: {v3_to_2.shape}, v3_to_4: {v3_to_4.shape}")

        # Calculate dot product and magnitudes
        dot_product = np.dot(v3_to_2, v3_to_4)
        magnitude_3_to_2 = np.linalg.norm(v3_to_2)
        magnitude_3_to_4 = np.linalg.norm(v3_to_4)

        # Calculate angle in degrees at point 3
        angle_degrees = np.degrees(np.arccos(np.clip(dot_product / (magnitude_3_to_2 * magnitude_3_to_4), -1.0, 1.0)))

        return pose, angle_degrees

    def save(self):
        pass