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

# Function to save frames using ffmpeg subprocess
def save_buffer_frames(buffer, out_file_name):    
    try:
        redis_store = Redis(host='localhost', port=6379)
    except Exception:
        logger.exception("Cannot connect to redis datastore localhost:6379")

    # video_save_command = [
    #     'ffmpeg', '-y', '-f', 'rawvideo', '-vcodec', 'rawvideo',
    #     '-s', f'{1920}x{1080}', '-pix_fmt', 'rgb24', '-r', str(60),
    #     '-i', '-', '-an', '-vcodec', 'mpeg4', '-pix_fmt', 'yuv420p', out_file_name,
    #     '-loglevel', 'error'
    # ]
    # video_proc = subprocess.Popen(video_save_command, stdin=subprocess.PIPE)

    with open(out_file_name, 'wb') as f:
        for frame_id in buffer:
            if frame_id is not None:
                f.write(pickle.loads(redis_store.get(frame_id)))
                # video_proc.stdin.write(pickle.loads(redis_store.get(frame_id)))

                redis_store.expire(frame_id, 1) # set expiration on the key from the store to 1 second (TODO: move this to config file)

    # video_proc.stdin.close()
    redis_store.close()

class VideoSaver(ManagedActor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.camera_num = kwargs['camera_num']

    def read_frames_process(self):
        buffer_length = 7 # time length of the buffer in seconds (TODO: move this to config file)
        num_buf_frames = self.fps * buffer_length  # number of frames to buffer 
        frame_shape = (self.frame_h, self.frame_w, 3)

        # initialize the buffer to store the frames
        buffer = deque(maxlen=num_buf_frames)

        buffer_index = 0 # current index in the buffer
        num_buffer = 0 # number of buffers saved
        # workers = []
        worker = None
        frame = None

        home_dir = os.path.expanduser('~')
        
        with Pool(processes=3) as pool:
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
                            if worker is not None:
                                worker.get()  # Wait for the previous worker to finish
                                
                                # send signal to the video converter to start converting the video file
                                data_id = self.client.put(out_file_name)
                                self.q_out.put(data_id)

                            # Start a new worker to save the buffer
                            out_file_name = f"{home_dir}/camera_video/chunks/camera_video_{self.camera_num}_{num_buffer}.raw"

                            worker = pool.apply_async(save_buffer_frames, args=(deepcopy(buffer), out_file_name,))
                            # workers.append(worker)

                            logger.info(f"[Camera {self.camera_name}] Saving buffer {num_buffer} with {len(buffer)} frames")

                            # Reset the buffer index and buffer
                            buffer = deque(maxlen=num_buf_frames)
                            buffer_index = 0
                            num_buffer += 1

                        # Reset the time and frame count to calculate FPS
                        if self.frame_count % 300 == 0:
                            if frame is not None:
                                logger.info(type(frame))
                                logger.info(frame.shape)
                            
                            time_end = time.perf_counter()
                            total_time = time_end - self.time_start
                            logger.info(f"[Camera {self.camera_name}] General FPS: {round(self.frame_count / total_time,2)}")
                            self.frame_count = 0
                            self.time_start = time.perf_counter()

                except Exception as e:
                    logger.error(f"[Camera {self.camera_name}] No more frames | {e}")
                    self.stop_program = True

            if self.stop_program:
                # Wait for the last worker to finish if it exists
                # for worker in workers:
                #     worker.get()
                worker.get()

                # send the last buffer to the video converter
                logger.info(f"[Camera {self.camera_name}] saving the last frames")
                out_file_name = f"{home_dir}/camera_video/chunks/camera_video_{self.camera_num}_{num_buffer}.raw"
                worker = pool.apply_async(save_buffer_frames, args=(deepcopy(buffer), out_file_name,))
                worker.get()

                data_id = self.client.put(out_file_name)
                self.q_out.put(data_id)

    def setup(self):
        # store init
        self._getStoreInterface()

        # load the configuration file
        source_folder = Path(__file__).resolve().parent.parent

        with open(f'{source_folder}/config/camera_config.yaml', 'r') as file:
            config = yaml.safe_load(file)

        camera_params = config['camera_params']
        self.frame_w = camera_params['resolution']['width'] # frame width
        self.frame_h = camera_params['resolution']['height'] # frame height

        camera_config = config['active_cameras'][self.camera_num]['camera']
        self.camera_name = camera_config['name']

        # exctract from the string "60/1" the fps value
        self.fps = int(camera_params['fps'].split('/')[0])

        # control variables
        self.stop_program = False
        self.total_frames = 0

        self.frame_count = 0
        self.total_delay = 0
        self.max_delay = 0
        self.time_start = time.perf_counter()

        self.start_program = False

        # store process
        # self.store_frame_proc = Process(target=self.read_frames_process)
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