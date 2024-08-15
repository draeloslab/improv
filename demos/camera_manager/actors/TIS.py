import time
import numpy as np
import zmq
from enum import Enum
from collections import namedtuple

import gi
gi.require_version("Gst", "1.0")
gi.require_version("Tcam", "1.0")

from gi.repository import GLib, Gst, Tcam

# Needed packages:
# pyhton-gst-1.0
# python-opencv
# tiscamera (+ pip install pycairo PyGObject)

class SinkFormats(Enum):
    GRAY8 = "GRAY8"
    GRAY16_LE = "GRAY16_LE"
    BGR = "BGRx"
    RGB = "RGBx64"

class TIS:
    def __init__(self, zmq_port=5555, shared_frame=None):
        try:
            if not Gst.is_initialized():
                Gst.init(())  # Usually better to call in the main function.
        except gi.overrides.Gst.NotInitialized:
            # Older gst-python overrides seem to have a bug where Gst needs to
            # already be initialized to call Gst.is_initialized
            Gst.init(())

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
        zmq_context = zmq.Context()
        self.zmq_socket = zmq_context.socket(zmq.PUSH)
        self.zmq_socket.bind(f"tcp://*:{zmq_port}")

        self.shared_frame = shared_frame

    def open_device(self, serial,
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

        self.dest_frame = np.frombuffer(self.shared_frame).reshape((self.height, self.width, 4))

        if self.sinkformat == SinkFormats.GRAY8:
            self.bpp = 1
        elif self.sinkformat == SinkFormats.BGR:
            self.bpp = 4

        self._create_pipeline(conversion, showvideo)
        self.source.set_property("serial", self.serialnumber)
        self.pipeline.set_state(Gst.State.READY)
        self.pipeline.get_state(40000000)

    def _create_pipeline(self, conversion: str, showvideo: bool):
        if conversion and not conversion.strip().endswith("!"):
            conversion += " !"
        p = 'tcambin name=source ! capsfilter name=caps'
        
        if showvideo:
            p += " ! tee name=t"
            p += " t. ! queue ! videoconvert ! ximagesink"
            p += f" t. ! queue ! {conversion} appsink name=sink"
        else:
            p += f" ! queue ! {conversion} appsink name=sink"

        print(f'\tPipeline starting command: {p}')

        try:
            self.pipeline = Gst.parse_launch(p)
        except GLib.Error as error:
            print("Error creating pipeline: {0}".format(error))
            raise

        # Quere the source module.
        self.source = self.pipeline.get_by_name("source")

        # Query a pointer to the appsink, so we can assign the callback function.
        appsink = self.pipeline.get_by_name("sink")
        appsink.set_property("max-buffers", 4)
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

        print(f"\tcaps command: {caps.to_string()}")

        capsfilter = self.pipeline.get_by_name("caps")
        capsfilter.set_property("caps", caps)

    def start_pipeline(self):
        """ Start the pipeline, so the video start running """
        self.start_time = time.perf_counter()
        self.frame_count = 0

        self.image_data = []
        self.image_caps = None

        self._setcaps()
        self.pipeline.set_state(Gst.State.PLAYING)

        cam_state = self.pipeline.get_state(5000000000)

        if cam_state[1] != Gst.State.PLAYING:
            print("Error starting pipeline. {0}".format(""))
            return False
        
        return True
    
    # starting sharing the frames received from the camera
    def start_sharing(self):
        self.sharing_on = True
    
    # @profile
    def __on_new_buffer(self, appsink):
        current_time = time.perf_counter()
        sample = appsink.get_property('last-sample')

        if sample is not None:
            buf = sample.get_buffer()

            self.__convert_to_numpy(buf.extract_dup(0, buf.get_size()), sample.get_caps())

            if self.sharing_on:
                self.zmq_socket.send_pyobj(current_time)

            self.frame_count += 1

            # if self.frame_count % 120 == 0:               
            #     total_time = current_time - self.start_time

            #     # print(f"Local FPS: {round(self.frame_count / total_time,2)}")                
            #     # print(f"{img.shape} - size on memory: {round(img.nbytes/(1024**2),2)}MB")

            #     self.frame_count = 0
            #     self.start_time = time.perf_counter()
        
        return Gst.FlowReturn.OK

    # @profile
    def __convert_to_numpy(self, data, caps):
        ''' Convert a GStreamer sample to a numpy array
            Sample code from https://gist.github.com/cbenhagen/76b24573fa63e7492fb6#file-gst-appsink-opencv-py-L34

            The result is in self.img_mat.
        '''
        s = caps.get_structure(0)

        # save in the shared memory
        self.dest_frame[:] = np.frombuffer(data, dtype=np.uint8).reshape((s.get_value('height'), s.get_value('width'), self.bpp))
    
    def stop_pipeline(self):
        self.stop_program = True
        self.sharing_on = False
        
        self.pipeline.set_state(Gst.State.PAUSED)
        self.pipeline.set_state(Gst.State.READY)

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