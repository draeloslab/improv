import os
import time
import threading
import yaml
import numpy as np
import subprocess
import struct
from improv.actor import ManagedActor
from pathlib import Path
from multiprocessing import Pool, Process
from collections import deque
from queue import Queue
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

class OfflineVideoConverter(ManagedActor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.camera_num = kwargs['camera_num']

    # Function to save chunks of frames to a video file
    def save_buffer_frames(self, buffer, num_buffer):
        try:
            compressed_frames = [None] * len(buffer) # Preallocate list to reduce memory usage
            
            for i, frame_id in enumerate(buffer):
                frame_enc = self.client.get(frame_id)
                compressed_frames[i] = frame_enc.tobytes()

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

        # control variables
        self.start_program = False
        self.conversion_started = False

        self.wait_conversion_proc = threading.Thread(target=self.wait_conversion_process)
        self.convert_saved_frames_proc = threading.Thread(target=self.convert_saved_frames)
        self.stop_program_event = threading.Event()

        self.wait_conversion_proc.start()

        logger.info(f"[Camera {self.camera_name}] saver setup completed")

    def runStep(self):      
        if not self.start_program:            
            self.start_program = True
            self.convert_saved_frames_proc.start()
        else:
            time.sleep(1)

    def stop(self):
        self.start_program = False
        self.stop_program_event.set()
        
        self.wait_conversion_proc.join()

        if self.conversion_started:
            logger.info("Stop signal requested - quitting the conversion before the end")

            # wait until the video conversion process has finished
            if self.convert_saved_frames_proc.is_alive():
                self.convert_saved_frames_proc.join()       # Wait for the process to terminate
                logger.info("Conversion process has been terminated.")

            logger.info(f"[Camera {self.camera_name}] video conversion killed")

    def wait_conversion_process(self):
        # wait until the user send the conversion start request
        try:
            while not self.conversion_started and not self.stop_program_event.is_set():   
                try: 
                    msg = self.links["msg_in"].get(timeout=0.5)
                except KeyboardInterrupt:
                    self.stop_program_event.set()
                except:
                    msg = None        

                if msg is not None:
                    logger.info(f"[Camera {self.camera_name}] received message: {msg}")

                    if msg['type'] == 'buffer_folder':
                        self.out_folder_buffer = msg['value'] + f"/camera_{self.camera_num}/"
                        self.output_video = msg['value'] + f"/camera_video_{self.camera_num+1}.avi"
                    elif msg['type'] == 'video_conversion':
                        if msg['value']:
                            self.conversion_started = True
                
                            # start conversion
                            logger.info(f"[Camera {self.camera_name}] video conversion started")
                        else:
                            self.stop_program_event.set()

                time.sleep(.5)
        except Exception as e:
            logger.error(f"Error waiting for conversion start | {e}")

        logger.info(f"[Camera {self.camera_name}] waiting process terminated")

    def convert_saved_frames(self):
        """
        Converts saved frames into a video file using the VideoConverter class.
        
        This method initializes the VideoConverter with the necessary parameters
        and starts the conversion process. It logs the progress and handles any
        exceptions that may occur during the conversion.
        """

        if not self.stop_program_event.is_set():
            try:
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
                    log_progress = False,
                    msg_out = self.links['msg_out'],
                    stop_program_event = self.stop_program_event
                )

                self.video_converter.convert_saved_frames()
            except Exception as e:
                logger.error(f"Error converting saved frames | {e}")

        logger.info(f"[Camera {self.camera_name}] conversion process terminated")