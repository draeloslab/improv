import yaml
import time
from multiprocessing import Value, RawArray, Process

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Create a file handler
log_file = "camera_reader.log"
file_handler = logging.FileHandler(log_file)
file_handler.setLevel(logging.INFO)

# Create a formatter and set it for the handler
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)

# Add the handler to the logger
logger.addHandler(file_handler)

from pathlib import Path
from improv.actor import ManagedActor
from .TIS import *

class CameraReader(ManagedActor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.camera_num = kwargs['camera_num']

    def setup(self):
        """ Initializes the camera reading process """
        logger.info("setup init")

        # store init
        self._getStoreInterface()

        # load the configuration file
        source_folder = Path(__file__).resolve().parent.parent

        with open(f'{source_folder}/camera_config.yaml', 'r') as file:
            config = yaml.safe_load(file)

        camera_params = config['camera_params']

        self.frame_w = camera_params['resolution']['width'] # frame width
        self.frame_h = camera_params['resolution']['height'] # frame height
        self.fps = camera_params['fps'] # FPS
        
        self.start_camera_read = False # flag to start the camera reading
        self.stop_program = Value('b', False) # flag to stop the program
        self.total_frames = 0 # total frames read

        # camera setup
        self.camera_interface = None

        camera_config = config['active_cameras'][self.camera_num]['camera']
        self.camera_name = camera_config['name']

        logger.info(f'Initializing camera {self.camera_name}')

        # initializing the camera frame and socket for communication
        shared_frame = RawArray('B', self.frame_w * self.frame_h * 4) # shared memory for frames reading

        # initializing each camera TIS interface and opening the device
        logger.info(f'Opening device: {camera_config}')

        self.camera_interface = TIS(self.camera_name, self.client, self.q_out)
        self.camera_interface.open_device(camera_config['serial_id'], shared_frame, self.frame_w, self.frame_h, self.fps, SinkFormats.RGB, showvideo=False)
        
        logger.info(f'Device {self.camera_name} opened')

        # starting the camera pipeline
        self.camera_interface.start_pipeline()
        logger.info(f'Device {self.camera_name} pipeline started')

    def runStep(self):
        if not self.start_camera_read:
            self.camera_interface.start_sharing()
            self.start_camera_read = True

            start_time = time.perf_counter()
            logger.info(f"[Camera {self.camera_name}] Run started at {start_time}")

    def stop(self):
        """Trivial stop function for testing purposes."""
        logger.info(f"[Camera {self.camera_name}] - CameraReader stopping")   

        self.stop_program.value = True
        self.camera_interface.stop_pipeline()