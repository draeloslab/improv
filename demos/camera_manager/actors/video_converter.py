import os
import time
import threading
import yaml
import numpy as np
import subprocess
import pickle

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

# Function to save frames using ffmpeg subprocess
def convert_video_raw(in_file_name, frame_w, frame_h, fps):        
    out_file_name = in_file_name.split('.')[0] + '.mp4'
    
    video_save_command = [
        'ffmpeg', '-y', '-f', 'rawvideo', '-vcodec', 'rawvideo',
        '-s', f'{frame_w}x{frame_h}', '-pix_fmt', 'rgb24', '-r', str(fps),
        '-i', '-', '-an', '-vcodec', 'libx264', '-pix_fmt', 'yuv420p', out_file_name,
        '-crf', '15',  # CRF value for high quality
        '-preset', 'slow',
        '-loglevel', 'error',  # Suppress all output except for errors
        '-threads', '1'  # Limit to 1 thread
    ]

    try: 
        video_proc = subprocess.Popen(video_save_command, stdin=subprocess.PIPE)

        # read the raw-video file and write it to the ffmpeg process
        with open(in_file_name, 'rb') as f:
            for frame in f:
                video_proc.stdin.write(frame)                

        video_proc.stdin.close()
        video_proc.wait()

        # delete the raw video file
        os.remove(in_file_name)
    except Exception as e:
        logger.error(f"[convert_video_raw] Error converting video | {e}")

class VideoConverter(ManagedActor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.camera_num = kwargs['camera_num']

    def convert_video_process(self):        
        with Pool(processes=self.num_convert_processes) as pool:
            workers = []
            processed_files = set()

            while not self.stop_program:
                try:
                    # List all files in the output folder
                    files = os.listdir(self.out_folder)
                    raw_files = [f for f in files if f.endswith('.raw') and f.startswith(f'camera_video_{self.camera_num}')]

                    # Sort the files in ascending order
                    raw_files.sort()

                    if len(raw_files) >= 2:
                        for raw_file in raw_files:
                            video_name = os.path.join(self.out_folder, raw_file)
                            file_hash = hash(video_name)

                            if file_hash not in processed_files:
                                worker = pool.apply_async(convert_video_raw, args=(video_name, self.frame_w, self.frame_h, self.fps,))
                                workers.append(worker)
                                processed_files.add(file_hash)
                                
                                # Break after sending the older file to the worker
                                break
                    
                    # Sleep for a while before checking again
                    time.sleep(.5)
                except Exception as e:
                    logger.error(f"[Camera {self.camera_name}] Video converter: Error {e}")
                    self.stop_program = True

            # Wait for the last worker to finish if it exists
            for worker in workers:
                worker.get()

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