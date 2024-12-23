import os
import time
import threading
import yaml
import numpy as np
import subprocess
import pickle
import cv2
import struct
from skvideo.io import FFmpegWriter
from copy import deepcopy
from improv.actor import ManagedActor
from pathlib import Path
from multiprocessing import Pool, Process
from redis import Redis
from collections import deque
from queue import Queue

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
            compressed_frames = [] # TODO: this could be a fixed vector for reducing memory usage
            
            for frame_id in buffer:
                frame_enc = self.client.get(frame_id)
                compressed_frames.append(frame_enc.tobytes())

                # Set the expiration for the frame in the client
                self.client.expire(frame_id, 5)

            # Define the output file path
            file_name = f'buffer_{num_buffer:05d}.bin'
            file_path = os.path.join(self.out_folder_buffer, file_name)
            
            # Save all compressed frames to the binary file
            with open(file_path, 'wb') as f:
                for cf in compressed_frames:
                    # Write the length of the compressed frame
                    f.write(len(cf).to_bytes(4, byteorder='little'))
                    # Write the compressed frame data
                    f.write(cf)

        except Exception as e:
            logger.error(f"Error saving frames | {e}")

    def save_buffer_frames_loop(self):
        while not self.stop_program or not self.save_queue.empty():
            try:
                prev_buffer, num_buffer = self.save_queue.get(timeout=1)
                self.save_buffer_frames(prev_buffer, num_buffer)
                self.save_queue.task_done()
            except:
                continue

    def read_frames_process(self):
        num_buf_frames = self.fps * self.buffer_length  # number of frames to buffer 

        # initialize the buffer to store the frames
        buffer = deque(maxlen=num_buf_frames)

        buffer_index = 0 # current index in the buffer
        num_buffer = 0 # number of buffers saved
        workers = []
        
        while not self.stop_program:
            try:
                frame_id = self.q_in.get(timeout=1)

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
                    if self.frame_count % 300 == 0:                            
                        time_end = time.perf_counter()
                        total_time = time_end - self.time_start
                        logger.info(f"[Camera {self.camera_name}] General FPS: {round(self.frame_count / total_time,2)}")
                        self.frame_count = 0
                        self.time_start = time.perf_counter()
            except Exception as e:
                logger.info(f"[Camera {self.camera_name}] No more frames {e}")
                self.stop_program = True

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
        self.compression_quality = video_config['compression_quality']

        # create the dest folder
        date = time.strftime("%Y-%m-%d")
        timestamp = time.strftime("%H%M%S")

        self.out_folder_buffer = f"{home_dir}/{raw_chunks_path}/{date}/{timestamp}/camera_{self.camera_num}/"
        self.out_folder_video = f"{home_dir}/{raw_chunks_path}/{date}/{timestamp}/"

        if not Path(self.out_folder_buffer).exists():
            Path(self.out_folder_buffer).mkdir(parents=True, exist_ok=True)

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
        self.output_video = os.path.join(self.out_folder_video, f"camera_video_{self.camera_num+1}.avi")

        self.convert_saved_frames_proc = threading.Thread(target=self.convert_saved_frames)
        self.writer_video_proc = threading.Thread(target=self.save_video_process)

        logger.info(f"[Camera {self.camera_name}] saver setup completed")

    def runStep(self):      
        if not self.start_program:
            self.start_program = True
            self.store_frame_proc.start()

    def stop(self):
        self.stop_program = True
        self.start_program = False

        logger.info(f"[Camera {self.camera_name}] waiting for camera saver thread to finish")

        # wait until the store thread has finished it's execution
        self.store_frame_proc.join()     

        # wait until all buffers are saved
        self.save_queue.join()

        logger.info(f"[Camera {self.camera_name}] total frames received: {self.total_frames}")
        
        # start conversion
        self.convert_saved_frames_proc.start()

        # wait until the video conversion process has finished
        self.convert_saved_frames_proc.join()
        self.writer_video_proc.join()
        logger.info(f"[Camera {self.camera_name}] video conversion completed")

    def save_video_process(self):        
        logger.info(f"[Camera {self.camera_name}] save_video_process started")
        input_dict = {
            '-pix_fmt': 'rgb24',
            '-r': str(self.fps),
            '-threads': '0'
        }

        video_compression_quality = self.__map_cv_quality_to_ffmpeg_q(self.compression_quality)

        output_dict = {
            '-c:v': 'mjpeg',                          # Use MJPEG codec
            '-q:v': str(video_compression_quality),   # Quality level (lower is higher quality)
            '-pix_fmt': 'yuvj420p',
            '-r': str(self.fps),
            '-threads': '0'
        }

        logger.info(f"[Camera {self.camera_name}] Saving video to {self.output_video}")

        saving_error = False

        try:
            video_proc = FFmpegWriter(self.output_video, inputdict=input_dict, outputdict=output_dict)

            while True:
                frame = self.video_conv_queue.get()

                if frame is None:
                    break        

                video_proc.writeFrame(frame)
        except Exception as e:
            logger.error(f"[Camera {self.camera_name}] Error writing frame to video | {e}")
            saving_error = True

        video_proc.close()

        if not saving_error:
            # Delete binary files after successful video creation
            buffer_files = sorted(Path(self.out_folder_buffer).glob('buffer_*.bin'))

            for buffer_file in buffer_files:
                try:
                    os.remove(buffer_file)
                except Exception as e:
                    logger.error(f"[Camera {self.camera_name}] Failed to delete {buffer_file} | {e}")

        self.video_conv_queue.close()

    def convert_saved_frames(self):
        # Gather and sort all binary files
        buffer_files = sorted(Path(self.out_folder_buffer).glob('buffer_*.bin'))

        self.writer_video_proc.start()

        msg = {'type': 'num_buffer_files', 'value': len(buffer_files)}
        self.links['msg_out'].put(msg)

        for idx, buffer_file in enumerate(buffer_files):
            logger.info(f"[Camera {self.camera_name}] Processing {idx+1}/{len(buffer_files)}")

            with open(buffer_file, 'rb') as f:                
                while True:
                    # Read the length of the compressed frame (4 bytes)
                    length_bytes = f.read(4)

                    if not length_bytes:
                        break

                    frame_length = struct.unpack('I', length_bytes)[0]

                    # Read the compressed frame data
                    frame_data = f.read(frame_length)

                    if len(frame_data) != frame_length:
                        logger.warning(f"[Camera {self.camera_name}] Unexpected frame length in {buffer_file}")
                        break

                    # Decompress the frame (assuming PNG compression)
                    nparr = np.frombuffer(frame_data, np.uint8)
                    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                    if frame is not None:
                        self.video_conv_queue.put(frame)
                    else:
                        logger.warning(f"[Camera {self.camera_name}] Failed to decode frame in {buffer_file}")

            msg = {'type': 'buffer_conv_progress', 'value': idx+1}
            self.links['msg_out'].put(msg)

        self.video_conv_queue.put(None)

        # Signal the end of the video conversion
        msg = {'type': 'video_conversion_done'}
        self.links['msg_out'].put(msg)

    # Function to map the OpenCV imwrite quality to FFmpeg quality
    def __map_cv_quality_to_ffmpeg_q(self, imwrite_quality):
        return max(2, min(31, int(31 - (imwrite_quality * 29 / 100))))