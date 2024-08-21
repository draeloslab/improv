from improv.actor import ManagedActor
import time
import threading
import yaml
from pathlib import Path
import zarr

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Create a file handler
log_file = "camera_recoder.log"
file_handler = logging.FileHandler(log_file)
file_handler.setLevel(logging.INFO)

# Create a formatter and set it for the handler
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)

# Add the handler to the logger
logger.addHandler(file_handler)

class CameraSaver(ManagedActor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.camera_num = kwargs['camera_num']

    def read_frames_process(self):
        total_delay = 0

        while not self.stop_program:
            frame_num = 1
            try:
                frame_id = self.q_in.get(timeout=1)

                if frame_id is not None:
                    start_time = time.perf_counter()
                    self.array_store[self.frame_count] = self.client.get(frame_id)
                    end_time = time.perf_counter()

                    total_delay += end_time - start_time
                    frame_num += 1

                    self.total_frames += 1
                    self.frame_count += 1

                    if self.frame_count % 60 == 0:
                        time_end = time.perf_counter()
                        total_time = time_end - self.time_start

                        logger.info(f"[Camera {self.camera_name}] General FPS: {round(self.frame_count / total_time,2)} - avg delay: {round(total_delay / self.frame_count, 4)}")
                        # logger.info(f"frame size on disk: {round(frame.nbytes/(1024**2),2)}MB")
                        self.frame_count = 0
                        total_delay = 0

                        self.time_start = time.perf_counter()
            except Exception as e:
                logger.error(f"[Camera {self.camera_name}] Could not get frame! {e}")
                self.stop_program = True
                pass

    def setup(self):
        # store init
        self._getStoreInterface()

        # load the configuration file
        source_folder = Path(__file__).resolve().parent.parent

        with open(f'{source_folder}/config/camera_config.yaml', 'r') as file:
            config = yaml.safe_load(file)

        camera_params = config['camera_params']
        frame_w = camera_params['resolution']['width'] # frame width
        frame_h = camera_params['resolution']['height'] # frame height

        self.array_store = zarr.open('/home/matteo/camera_video/frames_1.zarr', mode='w', shape=(120, 768, 1024, 4), dtype='u1')

        camera_config = config['active_cameras'][self.camera_num]['camera']
        self.camera_name = camera_config['name']

        # control variables
        self.stop_program = False
        self.total_frames = 0

        self.frame_count = 0
        self.total_delay = 0
        self.max_delay = 0
        self.time_start = time.perf_counter()

        self.start_program = False

        # store process
        self.store_frame_proc = threading.Thread(target=self.read_frames_process)

        logger.info(f"[Camera {self.camera_name}] saver setup completed")

    def runStep(self):      
        if not self.start_program:
            self.start_program = True
            self.store_frame_proc.start()

    def stop(self):
        logger.info(f"[Camera {self.camera_name}] waiting for camera saver thread to finish")

        # wait until the store thread has finished it's execution
        self.store_frame_proc.join()            

        logger.info(f"[Camera {self.camera_name}] total frames received: {self.total_frames}")

        self.start_program = False