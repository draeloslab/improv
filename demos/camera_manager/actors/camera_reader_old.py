import yaml
import multiprocessing
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

from pathlib import Path
from improv.actor import ManagedActor
from .TIS import *

# Needed packages:
# pyhton-gst-1.0
# python-opencv
# tiscamera (+ pip install pycairo PyGObject)

class CameraReader(ManagedActor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def setup(self):
        """ Initializes the camera reading process """

        # load the configuration file
        source_folder = Path(__file__).resolve().parent.parent

        with open(f'{source_folder}/config/camera_config.yaml', 'r') as file:
            camera_config = yaml.safe_load(file)

        camera_params = camera_config['camera_params']

        self.frame_w = camera_params['resolution']['width'] # frame width
        self.frame_h = camera_params['resolution']['height'] # frame height
        self.fps = camera_params['fps'] # FPS
        self.num_devices = len(camera_config['active_cameras'])

        self.stop_program = False # flag to stop the program

        # instantiating the camera processes and synching
        self.frames_queue = multiprocessing.Queue()
        self.camera_interfaces = []
        self.camera_names = []
        self.camera_synch = multiprocessing.Barrier(self.num_devices + 1)  # +1 for the main process

        for c in camera_config['active_cameras']:
            camera = c['camera']
            logger.info(f'Opening device {camera}')

            tis_camera = TIS(self.frames_queue, self.camera_synch)
            tis_camera.open_device(camera['serial_id'], self.frame_w, self.frame_h, self.fps, SinkFormats.BGRA, showvideo=False)

            self.camera_interfaces.append(tis_camera)
            self.camera_names.append(camera['name'])
            
            logger.info(f'Device {camera} opened')

        logger.info("Camera reading setup completed")

    def worker(self):
        time.sleep(2000)
        logger.info("Worker")

    def runStep(self):
        """ Reads frames from the camera """

        logger.info("Camera reading run")

        self.camera_processes = []

        # start the pipeline reading process for each camera
        for tis_camera in self.camera_interfaces:
            cam_process = multiprocessing.Process(target=tis_camera.process_buffer)
            #tis_camera.start_pipeline()
            self.camera_processes.append(cam_process)
            cam_process.start()

        while not self.stop_program:
            # wait until all the cameras have read a frame
            self.camera_synch.wait()

            # process item
            frames = []
            
            for _ in range(self.num_devices):
                frame = self.frames_queue.get()
                frames.append(frame)
                logger.info(f"{frame.shape} - {time.time()}")

    def stop(self):
        """ Stops the camera reading process """

        self.stop_program = True

        for tis_camera in self.camera_interfaces:
            tis_camera.stop_pipeline()

        for cam_process in self.camera_processes:
            cam_process.join()

        logger.info("Camera reading stop completed")