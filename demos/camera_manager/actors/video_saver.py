from improv.actor import ManagedActor
import time
import threading
import yaml
import cv2
from pathlib import Path

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

class VideoSaver(ManagedActor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.camera_num = kwargs['camera_num']

    def read_frames_process(self):
        while not self.stop_program:
            frame_num = 1

            try:
                frame_id = self.q_in.get(timeout=1)

                if frame_id is not None:
                    frame = self.client.get(frame_id)

                    frame_num += 1

                    self.video_out.write(frame)

                    self.total_frames += 1
                    self.frame_count += 1

                    if self.frame_count % 120 == 0:
                        time_end = time.perf_counter()
                        total_time = time_end - self.time_start

                        logger.info(f"[Camera {self.camera_name}] General FPS: {round(self.frame_count / total_time,2)}")
                        self.frame_count = 0

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

        camera_config = config['active_cameras'][self.camera_num]['camera']
        self.camera_name = camera_config['name']

        # exctract from the string "60/1" the fps value
        fps = int(camera_params['fps'].split('/')[0])

        # video saving parameters
        video_codec = cv2.VideoWriter_fourcc(*'XVID')
        out_file_name = f"camera_video_{self.camera_num+1}.avi"
        self.video_out = cv2.VideoWriter(f'/home/matteo/camera_video/{out_file_name}', video_codec, fps, (frame_w, frame_h), True)
        
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