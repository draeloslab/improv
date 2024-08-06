from improv.actor import Actor
import yaml
import os
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

import TIS
# Needed packages:
# pyhton-gst-1.0
# python-opencv
# tiscamera

class CameraReading(Actor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def setup(self):
        """ Initializes the camera reading process """

        # load the configuration file
        source_folder = os.path.dirname(os.path.abspath(__file__))

        with open(f'{source_folder}/config/camera_config.yaml', 'r') as file:
            camera_config = yaml.safe_load(file)

        camera_params = camera_config['camera_params']

        self.frame_w = camera_params['resolution']['width'] # frame width
        self.frame_h = camera_params['resolution']['height'] # frame height
        self.fps = camera_params['fps'] # FPS
        self.num_devices = len(camera_config['active_cameras'])

        # instantiating the camera interface objects (using TIS api)
        self.camera_interfaces = {}

        for camera in camera_config['active_cameras']:
            TisCamera = TIS.TIS()
            TisCamera.open_device(camera['serial_number'], self.frame_w, self.frame_h, self.fps, TIS.SinkFormats.BGRA, showvideo=False)
            
            self.camera_interfaces[camera['name']] = TisCamera

        logger.info("Camera reading setup completed")

    def run(self):
        """ Reads frames from the camera """

        for camera in self.camera_interfaces:
            self.camera_interfaces[camera].start_pipeline()

        # wait untile the operator stop the program
        input("\nPress a key to stop\n")

    def stop(self):
        """ Stops the camera reading process """

        for camera in self.camera_interfaces:
            self.camera_interfaces[camera].stop_pipeline()

        logger.info("Camera reading stopped")