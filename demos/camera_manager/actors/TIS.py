import time
import numpy as np
from enum import Enum
from collections import namedtuple

import gi
gi.require_version("Gst", "1.0")
gi.require_version("Tcam", "1.0")

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

from gi.repository import GLib, Gst, Tcam

# Needed packages:
# pyhton-gst-1.0
# python-opencv
# tiscamera (+ pip install pycairo PyGObject)

class SinkFormats(Enum):
    GRAY8 = "GRAY8"
    GRAY16_LE = "GRAY16_LE"
    BGRx = "BGRx"
    RGB = "RGB"

class TIS:
    def __init__(self, camera_name, client, q_out):
        try:
            if not Gst.is_initialized():
                Gst.init(())  # Usually better to call in the main function.
        except gi.overrides.Gst.NotInitialized:
            # Older gst-python overrides seem to have a bug where Gst needs to
            # already be initialized to call Gst.is_initialized
            Gst.init(())

        self.camera_name = camera_name

        self.sharing_on = False
        self.stop_program = False

        # Gst.debug_set_default_threshold(Gst.DebugLevel.WARNING)
        self.serialnumber = ""
        self.height = 0
        self.width = 0
        self.framerate = ""
        self.sinkformat = None
        self.img_mat = None
        self.ImageCallback = None
        self.pipeline = None
        self.source = None
        self.appsink = None
        
        # Array which will receive the images.
        self.image_data = []
        self.image_caps = None

        # buffer processing management
        self.client = client
        self.q_out = q_out

    def open_device(self, serial,
                    shared_frame,
                    width, height,
                    framerate,
                    sinkformat: SinkFormats,
                    showvideo: bool,
                    conversion: str = ""):
        ''' Inialize a device, e.g. camera.
        :param serial: Serial number of the camera to be used.
        :param width: Width of the wanted video format
        :param height: Height of the wanted video format
        :param framerate: Frame rate of the wanted video format
        :param sinkformat: Color format to use for the sink
        :param showvideo: Whether to always open a live video preview
        :param conversion: Optional pipeline string to add a conversion before the appsink
        :return: none
        '''
        if serial is None:
            raise RuntimeError("No serial number given on device initialization")

        self.serialnumber = serial
        self.width = width
        self.height = height
        self.framerate = framerate
        self.sinkformat = sinkformat

        if self.sinkformat == SinkFormats.GRAY8:
            self.bpp = 1
        elif self.sinkformat == SinkFormats.RGB:
            self.bpp = 3
        elif self.sinkformat == SinkFormats.BGRx:
            self.bpp = 4

        self.num_bytes = self.height * self.width * (self.bpp - 1)

        self._create_pipeline(conversion, showvideo)
        self.source.set_property("serial", self.serialnumber)
        self.pipeline.set_state(Gst.State.READY)
        self.pipeline.get_state(40000000)

    def _create_pipeline(self, conversion: str, showvideo: bool):
        if conversion and not conversion.strip().endswith("!"):
            conversion += " !"
        p = 'tcambin name=source ! videoconvert ! capsfilter name=caps'
        
        if showvideo:
            p += " ! tee name=t"
            p += " t. ! queue ! videoconvert ! ximagesink"
            p += f" t. ! queue ! appsink name=sink"
        else:
            p += f" ! queue ! appsink name=sink"

        logger.info(f'\tPipeline starting command: {p}')

        try:
            self.pipeline = Gst.parse_launch(p)
        except GLib.Error as error:
            logger.info("Error creating pipeline: {0}".format(error))
            raise

        # Quere the source module.
        self.source = self.pipeline.get_by_name("source")

        # Query a pointer to the appsink, so we can assign the callback function.
        appsink = self.pipeline.get_by_name("sink")
        appsink.set_property("max-buffers", 5)
        appsink.set_property("drop", True)
        appsink.set_property("emit-signals", True)
        appsink.set_property("enable-last-sample", True)
        appsink.connect('new-sample', self.__on_new_buffer)
        self.appsink = appsink

    def _setcaps(self):
        """
        Set pixel and sink format and frame rate
        """
        caps = Gst.Caps.from_string('video/x-raw,format=%s,width=%d,height=%d,framerate=%s' % (self.sinkformat.value, self.width, self.height, self.framerate))

        logger.info(f"\tcaps command: {caps.to_string()}")

        capsfilter = self.pipeline.get_by_name("caps")
        capsfilter.set_property("caps", caps)

    def start_pipeline(self):
        """ Start the pipeline, so the video start running """
        self.start_time = time.perf_counter()
        self.total_frame_count = 0
        self.frame_count = 0
        self.total_delay = 0
        self.max_delay = 0

        self.image_data = []
        self.image_caps = None

        self._setcaps()
        self.pipeline.set_state(Gst.State.PLAYING)

        cam_state = self.pipeline.get_state(5000000000)

        if cam_state[1] != Gst.State.PLAYING:
            logger.info("Error starting pipeline. {0}".format(""))
            return False
        
        return True
    
    # starting sharing the frames received from the camera
    def start_sharing(self):
        self.sharing_on = True

        self.total_start_time = time.perf_counter()
    
    # @profile
    def __on_new_buffer(self, appsink):
        frame_time = time.perf_counter()
        sample = appsink.get_property('last-sample')

        if sample is not None and self.sharing_on:
            buf = sample.get_buffer()

            frame = self.__convert_to_numpy(buf.extract_dup(0, buf.get_size()), sample.get_caps())

            try:
                data_id = self.client.put(frame)
                self.q_out.put(data_id)

                delay = time.perf_counter() - frame_time

                if delay > self.max_delay:
                    self.max_delay = delay

                self.total_delay += delay
            except Exception as e:
                pass

            self.frame_count += 1
            self.total_frame_count += 1

            if self.frame_count % 600 == 0:               
                total_time = time.perf_counter() - self.start_time

                logger.info(f"[Camera {self.camera_name}] reader FPS: {round(self.frame_count / total_time,2)} - avg delay: {self.total_delay/self.frame_count:.4f} - max delay: {self.max_delay:.4f}")                
                # logger.info(f"{frame.shape} - size on memory: {round(frame.nbytes/(1024**2),2)}MB")

                self.total_delay = 0
                self.max_delay = 0
                self.frame_count = 0
                self.start_time = time.perf_counter()
        
        return Gst.FlowReturn.OK

    # @profile
    def __convert_to_numpy(self, data, caps):
        ''' Convert a GStreamer sample to a numpy array
            Sample code from https://gist.github.com/cbenhagen/76b24573fa63e7492fb6#file-gst-appsink-opencv-py-L34
        '''
        s = caps.get_structure(0)

        return np.ndarray((s.get_value('height'), s.get_value('width'),self.bpp), buffer=data, dtype=np.uint8)
    
    def stop_pipeline(self):
        stop_time = time.perf_counter()

        self.sharing_on = False
        self.stop_program = True
        
        self.pipeline.set_state(Gst.State.PAUSED)
        self.pipeline.set_state(Gst.State.READY)

        recording_duration = stop_time - self.total_start_time

        logger.info(f"[Camera {self.camera_name}] reader stopped. Total frames: {self.total_frame_count} - Recording duration: {recording_duration:.2f}s ({round(recording_duration/60,1)} min)")

    def get_source(self):
        '''
        Return the source element of the pipeline.
        '''
        return self.source

    def list_properties(self):
        property_names = self.source.get_tcam_property_names()

        for name in property_names:
            try:
                base = self.source.get_tcam_property(name)
                print("{}\t{}".format(base.get_display_name(),
                                      name))
            except Exception as error:
                raise RuntimeError(f"Failed to get property '{name}'") from error

    def get_property(self, property_name):
        """
        Return the value of the passed property.
        If something fails an
        exception is thrown.
        :param property_name: Name of the property to set
        :return: Current value of the property
        """
        try:
            baseproperty = self.source.get_tcam_property(property_name)
            val = baseproperty.get_value()
            return val

        except Exception as error:
            raise RuntimeError(f"Failed to get property '{property_name}'") from error

        return None

    def set_property(self, property_name, value):
        '''
        Pass a new value to a camera property. If something fails an
        exception is thrown.
        :param property_name: Name of the property to set
        :param value: Property value. Can be of type int, float, string and boolean
        '''
        try:
            baseproperty = self.source.get_tcam_property(property_name)
            baseproperty.set_value(value)
        except Exception as error:
            raise RuntimeError(f"Failed to set property '{property_name}'") from error

    def execute_command(self, property_name):
        '''
        Execute a command property like Software Trigger
        If something fails an exception is thrown.
        :param property_name: Name of the property to set
        '''
        try:
            baseproperty = self.source.get_tcam_property(property_name)
            baseproperty.set_command()
        except Exception as error:
            raise RuntimeError(f"Failed to execute '{property_name}'") from error