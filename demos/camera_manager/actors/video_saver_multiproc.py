import os
import time
import threading
import yaml
import numpy as np
import subprocess
import pickle

from copy import deepcopy
from improv.actor import ManagedActor
from pathlib import Path
from multiprocessing import Pool, Array, shared_memory, Process
from redis import Redis
from redis.retry import Retry
from redis.backoff import ConstantBackoff
from collections import deque

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

# Function to save chunks of frames to a video file
def save_buffer_frames(buffer, out_file_name):    
    try:
        redis_store = Redis(host='localhost', port=6379)
    except Exception:
        logger.exception("Cannot connect to redis datastore localhost:6379")

    try:
        with open(out_file_name, 'wb') as f:
            for frame_id in buffer:
                if frame_id is not None:
                    f.write(pickle.loads(redis_store.get(frame_id)))
                    redis_store.expire(frame_id, 5) # remove the frame from the redis store after x seconds
    except Exception as e:
        logger.error(f"Error saving video | {e}")
    finally:
        redis_store.close()

class VideoSaver(ManagedActor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.camera_num = kwargs['camera_num']

    def read_frames_process(self):
        num_buf_frames = self.fps * self.buffer_length  # number of frames to buffer 

        # initialize the buffer to store the frames
        buffer = deque(maxlen=num_buf_frames)

        buffer_index = 0 # current index in the buffer
        num_buffer = 0 # number of buffers saved
        workers = []
        
        with Pool(processes=self.num_save_processes) as pool:
            while not self.stop_program:
                try:
                    frame_id = self.q_in.get(timeout=1)

                    if frame_id is not None:
                        # Insert the frame_id into the buffer at the current index
                        buffer.append(frame_id)

                        buffer_index += 1
                        self.total_frames += 1
                        self.frame_count += 1

                        # If the buffer is full, start the worker to save the buffer
                        if buffer_index == num_buf_frames:

                            # Start a new worker to save the buffer
                            out_file_name = f"{self.out_folder}/camera_video_{self.camera_num}_{num_buffer}.raw"

                            worker = pool.apply_async(save_buffer_frames, args=(deepcopy(buffer), out_file_name,))
                            workers.append(worker)

                            # Reset the buffer index and buffer
                            buffer = deque(maxlen=num_buf_frames)
                            buffer_index = 0
                            num_buffer += 1

                        # Reset the time and frame count to calculate FPS
                        if self.frame_count % 300 == 0:                            
                            time_end = time.perf_counter()
                            total_time = time_end - self.time_start
                            logger.info(f"[Camera {self.camera_name}] General FPS: {round(self.frame_count / total_time,2)}")
                            self.frame_count = 0
                            self.time_start = time.perf_counter()

                except Exception as e:
                    logger.error(f"[Camera {self.camera_name}] No more frames | {e}")
                    self.stop_program = True

            if self.stop_program:
                # send the last buffer to the video converter
                logger.info(f"[Camera {self.camera_name}] saving the last frames")
                out_file_name = f"{self.out_folder}/camera_video_{self.camera_num}_{num_buffer}.raw"
                worker = pool.apply_async(save_buffer_frames, args=(deepcopy(buffer), out_file_name,))
                workers.append(worker)

                # Wait every worker to finish
                for worker in workers:
                    worker.get()

            # Close the pool
            pool.close()

    def setup(self):
        # store init
        self._getStoreInterface()

        source_folder = Path(__file__).resolve().parent.parent
        home_dir = os.path.expanduser('~')

        # load the camera configuration params
        with open(f'{source_folder}/config/camera_config.yaml', 'r') as file:
            camera_config = yaml.safe_load(file)

        camera_params = camera_config['camera_params']
        self.frame_w = camera_params['resolution']['width'] # frame width
        self.frame_h = camera_params['resolution']['height'] # frame height        
        self.fps = int(camera_params['fps'].split('/')[0]) # extract the fps value

        camera_config = camera_config['active_cameras'][self.camera_num]['camera']
        self.camera_name = camera_config['name']

        # load the video configuration params
        with open(f'{source_folder}/config/video_config.yaml', 'r') as file:
            video_config = yaml.safe_load(file)

        raw_chunks_path = video_config['raw_chunks_path']
        self.buffer_length = video_config['buffer_length']
        self.num_save_processes = video_config['num_save_processes']

        # create the dest folder
        date = time.strftime("%Y-%m-%d")
        timestamp = time.strftime("%H%M%S")

        self.out_folder = f"{home_dir}/{raw_chunks_path}/{date}/{timestamp}"

        # send the out_folder to the VideoConverter
        self.q_out.put(self.out_folder)

        if not Path(self.out_folder).exists():
            Path(self.out_folder).mkdir(parents=True, exist_ok=True)

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
        self.stop_program = True