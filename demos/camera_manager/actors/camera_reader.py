import yaml
import time
import threading
import zmq
from multiprocessing import Process, RawArray

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

    # TODO: you might need a lock for reading and writing on the shared memory - to check this
    def read_frames(self, camera_name, socket, shared_frame):
        logger.info(f"[Camera {camera_name}] reading frames thread started")

        time_start = time.perf_counter()
        frame_count = 0
        total_delay = 0
        max_delay = 0

        while not self.stop_program:
            # receive the frame timestamp   
            frame_time = socket.recv_pyobj() 

            # pull the frame from the shared memory
            frame = np.frombuffer(shared_frame).reshape(self.frame_h, self.frame_w, 4).astype(np.uint8)
            
            received_time = time.perf_counter() 

            frame_count += 1
            total_delay += received_time - frame_time

            if received_time - frame_time > max_delay:
                max_delay = received_time - frame_time

            if frame_count % 240 == 0:
                time_end = time.perf_counter()
                total_time = time_end - time_start

                logger.info(f"[Camera {camera_name}] - General FPS: {round(frame_count / total_time,2)} - avg delay: {total_delay/frame_count:.3f} - max delay: {max_delay:.3f}")

                # frame_size = round(frame[:,:,:3].nbytes/(1024**2),4)
                # print(f"Frame shape {frame.shape} - size on memory: {frame_size}MB")

                frame_count = 0
                total_time = 0
                total_delay = 0
                max_delay = 0

                time_start = time.perf_counter()

    def setup(self):
        """ Initializes the camera reading process """
        logger.info("setup init")

        self._getStoreInterface()

        # load the configuration file
        source_folder = Path(__file__).resolve().parent.parent

        with open(f'{source_folder}/config/camera_config.yaml', 'r') as file:
            config = yaml.safe_load(file)

        camera_params = config['camera_params']

        self.frame_w = camera_params['resolution']['width'] # frame width
        self.frame_h = camera_params['resolution']['height'] # frame height
        self.fps = camera_params['fps'] # FPS
        
        self.num_devices = len(config['active_cameras'])
        self.start_camera_read = False # flag to start the camera reading
        self.stop_program = False # flag to stop the program

        # zmq intialization
        zmq_frames_ip = config['zmq_config']['ip']
        zmq_context = zmq.Context()

        # camera setup
        self.camera_processes = []
        self.camera_interfaces = []
        self.camera_names = []

        for i,c in enumerate(config['active_cameras']):
            camera = c['camera']

            logger.info(f'Initializing camera {camera["name"]}')

            # initializing the camera frame and socket for communication
            shared_frame = RawArray('d', self.frame_w * self.frame_h * 4) # shared memory for frames reading
            zmq_port = camera['zmq_port']

            socket = zmq_context.socket(zmq.PULL)
            socket.connect(f"tcp://{zmq_frames_ip}:{zmq_port}")

            # initializing each camera TIS interface and opening the device
            logger.info(f'Opening device: {camera}')

            tis_camera = TIS(zmq_port, shared_frame)
            tis_camera.open_device(camera['serial_id'], self.frame_w, self.frame_h, self.fps, SinkFormats.BGR, showvideo=False)

            self.camera_interfaces.append(tis_camera)
            self.camera_names.append(camera['name'])
            
            logger.info(f'Device {camera} opened')

            # starting the camera pipeline
            tis_camera.start_pipeline()

            logger.info(f'Device {camera} pipeline started')

            thread = threading.Thread(target=self.read_frames, args=(self.camera_names[i], socket, shared_frame))
            thread.start()

            logger.info(f'Device {camera} reading thread started')

            print(f'\tdevice {camera} pipeline started')

        logger.info("Camera reading setup completed")

        logger.info("setup completed")

    def runStep(self):
        if not self.start_camera_read:
            for tis_camera in self.camera_interfaces:
                tis_camera.start_sharing()

            self.start_camera_read = True

    def stop(self):
        """Trivial stop function for testing purposes."""
        logger.info("CameraReader stopping")   
        self.stop_program = True

        for tis_camera in self.camera_interfaces:
            tis_camera.stop_pipeline()

    def quit(self):
        logger.info("Quit signal function")   
        self.close_logger()    
        time.sleep(1)

    def close_logger(self):
        """Close the logger and its handlers."""
        handlers = logger.handlers[:]

        for handler in handlers:
            handler.close()
            logger.removeHandler(handler)    
        