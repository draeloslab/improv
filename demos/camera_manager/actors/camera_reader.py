import yaml
import time
import threading
import zmq
from multiprocessing import Value, RawArray, Pool, Process

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
from improv.actor import AsyncActor, ManagedActor
from .TIS import *

# making the shared control variables accessible from the camera_thread
def init_camera_thread(start,stop):
    global start_camera_read, stop_program

    start_camera_read = start
    stop_program = stop

# TODO: you might need a lock for reading and writing on the shared memory - to check this
def camera_thread(camera_name, config):
    logger.info(f"[Camera {camera_name}] handling camera thread started")

    frame_w = config['frame_w']
    frame_h = config['frame_h']
    fps = config['fps']

    # initializing the camera frame and socket for communication
    socket = zmq.Context().socket(zmq.PULL)
    socket.connect(f"tcp://{config['zmq_ip']}:{config['zmq_port']}")

    # initializing shared memory for frames reading
    shared_frame = RawArray('B', frame_w * frame_h * 4)

    # opening the camera device
    logger.info(f'[Camera {camera_name}] - opening device')
    tis_camera = TIS(config['zmq_port'])
    tis_camera.open_device(config['serial_id'], shared_frame, frame_w, frame_h, fps, SinkFormats.BGR, showvideo=False)
    logger.info(f'[Camera {camera_name}] - device opened')

    # starting the camera pipeline
    logger.info(f'[Camera {camera_name}] - starting pipeline')
    tis_camera.start_pipeline()
    logger.info(f'[Camera {camera_name}] - pipeline started')

    # TODO: find a smarter way to handle the start signal
    while not start_camera_read.value:
        time.sleep(0.001)

    tis_camera.start_sharing()
    time_start = time.perf_counter()

    logger.info(f"[Camera {camera_name}] - time start: {time_start}")
    frame_count = 0
    total_delay = 0
    max_delay = 0
    
    while not stop_program.value:
        # receive the frame timestamp   
        frame_time = socket.recv_pyobj() 

        # pull the frame from the shared memory
        frame = np.frombuffer(shared_frame, dtype=np.uint8).reshape(frame_h, frame_w, 4)
        
        received_time = time.perf_counter() 

        frame_count += 1
        total_delay += received_time - frame_time

        if received_time - frame_time > max_delay:
            max_delay = received_time - frame_time

        if frame_count % 120*30 == 0:
            time_end = time.perf_counter()
            total_time = time_end - time_start

            logger.info(f"[Camera {camera_name}] - General FPS: {round(frame_count / total_time,2)} - avg delay: {total_delay/frame_count:.3f} - max delay: {max_delay:.3f}")

            # frame_size = round(frame[:,:,:3].nbytes/(1024**2),4)
            # logger.info(f"Frame shape {frame.shape} - size on memory: {frame_size}MB")

            frame_count = 0
            total_time = 0
            total_delay = 0
            max_delay = 0

            time_start = time.perf_counter()

class CameraReader(ManagedActor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def start_camera_threads(self):
        with Pool(initializer=init_camera_thread, initargs=(self.start_camera_read, self.stop_program,)) as pool:
            workers = [pool.apply_async(camera_thread, args=(s,self.camera_configs[i])) for i,s in enumerate(self.camera_names)]

            # letting the threads run until the program is stopped
            for w in workers:
                w.get()

    def setup(self):
        """ Initializes the camera reading process """
        logger.info("Setup init")

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
        self.start_camera_read = Value('b', False) # flag to start the camera reading
        self.stop_program = Value('b', False) # flag to stop the program

        # cameras setup
        self.camera_names = []
        self.camera_configs = []

        for i,c in enumerate(config['active_cameras']):
            camera = c['camera']

            self.camera_names.append(camera['name'])

            camera_config = {                                                                                       
                'serial_id': camera['serial_id'],
                'zmq_ip': config['zmq_config']['ip'],
                'zmq_port': camera['zmq_port'],
                'frame_w': self.frame_w,
                'frame_h': self.frame_h,
                'fps': self.fps
            }
                                    
            self.camera_configs.append(camera_config)

        self.camera_process = threading.Thread(target=self.start_camera_threads)
        self.camera_process.start()

    def runStep(self):
        if not self.start_camera_read.value:
            self.start_camera_read.value = True
            logger.info("CameraReader run started")

    def stop(self):
        """Trivial stop function for testing purposes."""
        logger.info("CameraReader stopping")   
        self.stop_program.value = True

        for tis_camera in self.camera_interfaces:
            tis_camera.stop_pipeline()

    # TODO: understand how to add this to the quit function
    # def quit(self):
    #     logger.info("Quit signal function")   
    #     self.close_logger()    
    #     time.sleep(1)

    def close_logger(self):
        """Close the logger and its handlers."""
        handlers = logger.handlers[:]

        for handler in handlers:
            handler.close()
            logger.removeHandler(handler)