import os
import time
import threading
import yaml
import numpy as np
import subprocess
import struct
from skvideo.io import FFmpegWriter
from copy import deepcopy
from improv.actor import ManagedActor
from pathlib import Path
from multiprocessing import Pool, Process
from collections import deque
from queue import Queue, Empty
from .video_converter import VideoConverter

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

class VideoSaver(ManagedActor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.camera_num = kwargs['camera_num']

    # Function to save chunks of frames to a video file
    def save_buffer_frames(self, buffer, num_buffer):
        try:
            # Define the output file path
            file_name = f'buffer_{num_buffer:05d}.bin'
            file_path = os.path.join(self.out_folder_buffer, file_name)

            # Save all compressed frames to the binary file
            with open(file_path, 'wb') as f:
                for frame_id in buffer:
                    frame_enc = self.client.get(frame_id)
                    frame_bytes = frame_enc.tobytes()
                    f.write(len(frame_bytes).to_bytes(4, byteorder='little'))
                    f.write(frame_bytes)

                    # Set the expiration for the frame in the client
                    # self.client.expire(frame_id, 5)

        except Exception as e:
            logger.error(f"Error saving frames | {e}")

    def save_buffer_frames_loop(self):
        while not self.stop_program or not self.save_queue.empty():
            if self.start_program:
                try:
                    prev_buffer, num_buffer = self.save_queue.get(timeout=1)
                    self.save_buffer_frames(prev_buffer, num_buffer)
                    self.save_queue.task_done()
                except:
                    continue
            else:
                time.sleep(self.wait_time)

    def read_frames_process(self):
        num_buf_frames = self.fps * self.buffer_length  # number of frames to buffer 

        # initialize the buffer to store the frames
        buffer = deque(maxlen=num_buf_frames)

        buffer_index = 0 # current index in the buffer
        num_buffer = 0 # number of buffers saved
        workers = []
        
        while not self.stop_program:
            if self.start_program:
                try:
                    self.start_times.append(time.time())
                    start_perf = time.perf_counter()
                    msg = self.q_in.get(timeout=1)
                    # Support both old [data_id, timestamp] and new [data_id, timestamp, frame_num] formats
                    if len(msg) >= 2:
                        frame_id = msg[0]
                        camera_start = msg[1]
                    else:
                        frame_id = msg[0]
                        camera_start = None

                    if frame_id is not None:
                        # Insert the frame_id into the buffer at the current index
                        buffer.append(frame_id)

                        buffer_index += 1
                        self.total_frames += 1
                        self.frame_count += 1

                        # If the buffer is full, save the buffer
                        if buffer_index == num_buf_frames:
                            self.save_queue.put((deepcopy(buffer), num_buffer))

                            # Reset the buffer index and buffer
                            buffer = deque(maxlen=num_buf_frames)
                            buffer_index = 0
                            num_buffer += 1

                        # Reset the time and frame count to calculate FPS
                        if self.frame_count % 900 == 0:                            
                            time_end = time.perf_counter()
                            total_time = time_end - self.time_start
                            logger.info(f"[Camera {self.camera_name}] writer FPS: {round(self.frame_count / total_time,2)}")
                            self.frame_count = 0
                            self.time_start = time.perf_counter()
                        
                    self.latencies.append(time.perf_counter() - start_perf)
                except Empty:
                    # Timeout occurred, just continue waiting for frames
                    logger.info(f"[Camera {self.camera_name}] Queue timeout, continuing...")
                    continue
                except Exception as e:
                    logger.error(f"[Camera {self.camera_name}] Error reading frames: {e}")
                    self.stop_program = True
            else:
                time.sleep(self.wait_time)

            if self.stop_program:
                # send the last buffer to the video converter
                logger.info(f"[Camera {self.camera_name}] saving the last frames")

                worker = self.save_buffer_frames(buffer, num_buffer)

    def setup(self):
        # store init
        self._getStoreInterface()

        source_folder = Path(__file__).resolve().parent.parent
        home_dir = os.path.expanduser('~')

        # load the camera configuration params
        with open(f'{source_folder}/config/camera_config.yaml', 'r') as file:
            camera_config = yaml.safe_load(file)

        # load the configuration file
        with open(f'{source_folder}/config.yaml', 'r') as file:
            config = yaml.safe_load(file)

        camera_params = camera_config['camera_params']
        self.frame_w = camera_params['resolution']['width'] # frame width
        self.frame_h = camera_params['resolution']['height'] # frame height        
        self.fps = int(camera_params['fps'].split('/')[0]) # extract the fps value

        self.wait_time = np.round(1/self.fps/2, 3) # wait time when recordin not active (twice fast the frame rate)

        camera_config = camera_config['active_cameras'][self.camera_num]['camera']
        self.camera_name = camera_config['name']

        # load the video configuration params
        with open(f'{source_folder}/config/video_config.yaml', 'r') as file:
            video_config = yaml.safe_load(file)

        raw_chunks_path = video_config['raw_chunks_path']
        self.buffer_length = video_config['buffer_length']
        self.num_save_processes = video_config['num_save_processes']
        self.compression_quality = video_config['compression_quality']

        # create the dest folder
        date = time.strftime("%Y-%m-%d")
        timestamp = time.strftime("%H%M%S")

        self.out_folder_buffer = f"{home_dir}/{raw_chunks_path}/{date}/{timestamp}/camera_{self.camera_num}/"
        self.out_folder_video = f"{home_dir}/{raw_chunks_path}/{date}/{timestamp}/"

        if not Path(self.out_folder_buffer).exists():
            Path(self.out_folder_buffer).mkdir(parents=True, exist_ok=True)

        timestamp_hhmm = time.strftime("%H%M")
        self.output_video = os.path.join(self.out_folder_video, f"camera_video_{self.camera_num+1}_{timestamp_hhmm}.mp4")

        # Initialize latency tracking variables
        self.latencies = []
        self.start_times = []


        date = time.strftime("%Y%m%d")
        timestamp = time.strftime("%Y%m%d-%H%M")
        string = config['output_path']
        self.out_folder = Path(f"{string}/{date}/{timestamp}")
        self.out_folder.mkdir(parents=True, exist_ok=True)
        logger.info(f"Latency Output folder set to {self.out_folder}")

        # video converter setup
        video_params = {
            'frame_w': self.frame_w,
            'frame_h': self.frame_h,
            'fps': self.fps
        }

        self.video_converter = VideoConverter(
            compression_quality = self.compression_quality,
            video_params = video_params,
            output_video = self.output_video,
            out_folder_buffer = self.out_folder_buffer,
            log_progress = True
            # msg_out = self.links['msg_out']
        )

        # control variables
        self.stop_program = False
        self.start_program = False
        self.start_conversion = Queue()
    
        self.total_frames = 0

        self.frame_count = 0
        self.total_delay = 0
        self.max_delay = 0
        self.time_start = time.perf_counter()

        # Initialize save queue and start save thread
        self.save_queue = Queue()
        self.save_thread = threading.Thread(target=self.save_buffer_frames_loop, daemon=True)
        self.save_thread.start()

        # store process
        self.store_frame_proc = threading.Thread(target=self.read_frames_process)

        # video conversion process (for saving the video at the end of the recording)
        self.video_conv_queue = Queue(maxsize=1000) # video conversion queue

        self.wait_conversion_proc = threading.Thread(target=self.wait_conversion_process)
        self.convert_saved_frames_proc = threading.Thread(target=self.convert_saved_frames)

        logger.info(f"[Camera {self.camera_name}] saver setup completed")

    def runStep(self):      
        if not self.start_program:
            # clear the frames in queue
            while not self.q_in.empty():
                self.q_in.get_nowait()

            self.start_program = True
            self.store_frame_proc.start()
            logger.info(f"[Camera {self.camera_name}] recording started")

    def stop(self):
        self.stop_program = True
        self.start_program = False
        self.conversion_started = False

        logger.info(f"[Camera {self.camera_name}] waiting for camera saver thread to finish")

        # wait until the store thread has finished it's execution
        self.store_frame_proc.join()     

        # wait until all buffers are saved
        self.save_queue.join()

        logger.info(f"[Camera {self.camera_name}] total frames received: {self.total_frames}")

        np.save(self.out_folder / f"saverlatencies_cam_{self.camera_num}.npy", self.latencies)
        np.save(self.out_folder / f"saverstarts_cam_{self.camera_num}.npy", self.start_times)
        logger.info(f"[Camera {self.camera_name}] Latencies saved to {self.out_folder}")

        self.wait_conversion_proc.start()
        self.wait_conversion_proc.join()

        if self.conversion_started:
            # wait until the video conversion process has finished
            self.convert_saved_frames_proc.join()
            # self.writer_video_proc.join()
            logger.info(f"[Camera {self.camera_name}] video conversion completed")

    def wait_conversion_process(self):
        program_quit = False

        # wait until the user send the conversion start request
        while not self.conversion_started and not program_quit:   
            try:
                msg = self.links["msg_in"].get(timeout=0.5)
            except KeyboardInterrupt:
                program_quit = True
            except:
                msg = None        

            if msg is not None:
                logger.info(f"[Camera {self.camera_name}] received message: {msg}")

                if msg['type'] == 'video_conversion':
                    if msg['value']:
                        self.conversion_started = True
            
                        # start conversion
                        logger.info(f"[Camera {self.camera_name}] video conversion started")
                        self.convert_saved_frames_proc.start()
                    else:
                        program_quit = True

            time.sleep(1)

    def convert_saved_frames(self):
        self.video_converter.convert_saved_frames()

    # Function to map the OpenCV imwrite quality to FFmpeg quality
    def __map_cv_quality_to_ffmpeg_q(self, imwrite_quality):
        return max(2, min(31, int(31 - (imwrite_quality * 29 / 100))))