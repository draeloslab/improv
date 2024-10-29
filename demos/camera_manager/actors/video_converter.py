import os
import time
import threading
import yaml
import numpy as np
import subprocess
import pickle
import cv2
from skvideo.io import FFmpegWriter

from improv.actor import ManagedActor
from pathlib import Path
from multiprocessing import Pool, Array, shared_memory, Process

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Create a file handler
log_file = "camera_video_saver.log"
file_handler = logging.FileHandler(log_file)
file_handler.setLevel(logging.INFO)

# Create a formatter and set it for the handler
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)

# Add the handler to the logger
logger.addHandler(file_handler)

class VideoConverter(ManagedActor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.camera_num = kwargs['camera_num']

    def convert_video_process(self):      
        out_file_name = f'{self.out_folder}/camera_{self.camera_num}.mp4'

        input_dict = {
            '-pix_fmt':'rgb24',
            '-r':str(self.fps)
        }

        output_dict = {
            '-c:v':'libopenjpeg',
            '-pix_fmt':'yuv420p',
            '-r':str(self.fps),
            '-vcodec':'libx264',
            '-threads': '2'
        }

        try:
            video_proc = FFmpegWriter(out_file_name, inputdict=input_dict, outputdict=output_dict)

            while not self.stop_program:            
                # List all files in the output folder
                files = os.listdir(self.out_folder)
                raw_files = [f for f in files if f.endswith('.jpg')]

                # Sort the files in ascending order
                raw_files.sort()

                for raw_file in raw_files:
                    # open the raw_file jpg and write it in video_proc
                    frame_path = os.path.join(self.out_folder, raw_file)
                    frame = cv2.imread(frame_path)

                    video_proc.writeFrame(frame)

                    # delete the raw_file jpg
                    os.remove(os.path.join(self.out_folder, raw_file))
                
                # Sleep for a while before checking again
                time.sleep(.1)
            
            video_proc.close()
        except Exception as e:
            logger.error(f"[Camera {self.camera_name}] Video converter: Error {e}")
            self.stop_program = True

    def setup(self):
        # store init
        self._getStoreInterface()

        source_folder = Path(__file__).resolve().parent.parent

        # load the camera configuration params
        with open(f'{source_folder}/config/camera_config.yaml', 'r') as file:
            camera_config = yaml.safe_load(file)

        camera_params = camera_config['camera_params']
        self.frame_w = camera_params['resolution']['width'] # frame width
        self.frame_h = camera_params['resolution']['height'] # frame height

        camera_config = camera_config['active_cameras'][self.camera_num]['camera']
        self.camera_name = camera_config['name']

        # calculate from the string "60/1" the fps value
        self.fps = int(camera_params['fps'].split('/')[0])

        # load the video configuration params
        with open(f'{source_folder}/config/video_config.yaml', 'r') as file:
            video_config = yaml.safe_load(file)

        self.num_convert_processes = video_config['num_convert_processes']

        # control variables
        self.stop_program = False
        self.total_frames = 0

        self.frame_count = 0
        self.total_delay = 0
        self.max_delay = 0
        self.time_start = time.perf_counter()

        self.start_program = False

        # get the output video folder from the VideoSaver actor
        self.out_folder = self.q_in.get(timeout=5)

        # store process
        self.video_converter_proc = threading.Thread(target=self.convert_video_process)

    def runStep(self):      
        if not self.start_program:
            self.start_program = True
            self.video_converter_proc.start()

    def stop(self):
        logger.info(f"[Camera {self.camera_name}] waiting for video converter thread to finish")

        # wait until the converter thread has finished it's execution
        self.video_converter_proc.join()            

        self.start_program = False
        self.stop_program = True