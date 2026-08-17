import time
import numpy as np
import cv2
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

from pathlib import Path


from gi.repository import GLib, Gst, Tcam

# Needed packages:
# python-gst-1.0
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

        self.camera_latencies = []
        self.camera_latenciesFull = []
        self.cameraStarts = []

        # --- Timing logs ---
        self.frame_num = 0
        # Per-step breakdown (perf_counter durations in seconds)
        self.convert_latencies = []    # GStreamer buffer → numpy
        self.encode_latencies = []     # unused (JPEG encode removed); kept for file-schema compat
        self.store_put_latencies = []  # client.put
        self.queue_put_latencies = []  # q_out.put
        
        date = time.strftime("%Y%m%d")
        timestamp = time.strftime("%Y%m%d-%H%M")
        string = '/home/chesteklab/predictions'
        self.out_folder = Path(f"{string}/{date}/{timestamp}")
        self.out_folder.mkdir(parents=True, exist_ok=True)
        # logger.info(f"Output folder set to {self.out_folder}")
        logger.info("Completed setup for TIS")

    def open_device(self, serial,
                    shared_frame,
                    width, height,
                    framerate,
                    sinkformat: SinkFormats,
                    showvideo: bool,
                    conversion: str = "",
                    out_width: int = None,
                    out_height: int = None):
        ''' Inialize a device, e.g. camera.
        :param serial: Serial number of the camera to be used.
        :param width: Width of the wanted video format (native capture)
        :param height: Height of the wanted video format (native capture)
        :param framerate: Frame rate of the wanted video format
        :param sinkformat: Color format to use for the sink
        :param showvideo: Whether to always open a live video preview
        :param conversion: Optional pipeline string to add a conversion before the appsink
        :param out_width: If given, GStreamer downscales to this width before appsink
            (defaults to `width`, i.e. no scaling)
        :param out_height: If given, GStreamer downscales to this height before appsink
            (defaults to `height`, i.e. no scaling)
        :return: none
        '''
        if serial is None:
            raise RuntimeError("No serial number given on device initialization")

        self.serialnumber = serial
        self.width = width
        self.height = height
        self.framerate = framerate
        self.sinkformat = sinkformat
        # Output dimensions delivered to __on_new_buffer / the numpy array.
        # Cached once here instead of being re-queried from GObject caps on
        # every single frame (see __convert_to_numpy).
        self.out_width = out_width if out_width is not None else width
        self.out_height = out_height if out_height is not None else height

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
        # Downscale in GStreamer (C, off the Python critical path) instead of
        # decoding a full-res JPEG and cv2.resize-ing it in the processor.
        # When out_width/out_height == width/height this is a same-size
        # negotiation and effectively a no-op.
        p += ' ! videoscale ! capsfilter name=caps_out'

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
        Set pixel and sink format and frame rate (native capture), plus the
        downscaled format requested of the videoscale element before appsink.
        """
        caps = Gst.Caps.from_string('video/x-raw,format=%s,width=%d,height=%d,framerate=%s' % (self.sinkformat.value, self.width, self.height, self.framerate))

        logger.info(f"\tcaps command: {caps.to_string()}")

        capsfilter = self.pipeline.get_by_name("caps")
        capsfilter.set_property("caps", caps)

        caps_out = Gst.Caps.from_string('video/x-raw,format=%s,width=%d,height=%d' % (self.sinkformat.value, self.out_width, self.out_height))

        logger.info(f"\tcaps_out command: {caps_out.to_string()}")

        capsfilter_out = self.pipeline.get_by_name("caps_out")
        capsfilter_out.set_property("caps", caps_out)

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
            logger.info("Error starting pipeline. {0}".format(cam_state[1]))
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
            camera_start = time.time()
            self.cameraStarts.append(camera_start)

            # --- Step 1: Convert GStreamer buffer to numpy ---
            # Dimensions are cached on self (out_width/out_height/bpp, set once in
            # open_device) instead of being re-derived from sample.get_caps() /
            # get_structure() / get_value() on every single frame.
            t0 = time.perf_counter()
            frame = self.__convert_to_numpy(buf.extract_dup(0, buf.get_size()))
            self.convert_latencies.append(time.perf_counter() - t0)

            # --- Step 2: (JPEG encode removed) ---
            # Frame is already downscaled to out_width x out_height by GStreamer
            # (see _create_pipeline/_setcaps), so it goes into the store raw --
            # no encode here, no imdecode + cv2.resize downstream in the
            # processor. encode_latencies is kept (always empty) so downstream
            # tooling that expects the file to exist doesn't break.

            try:
                # --- Step 3: Store put ---
                t0 = time.perf_counter()
                data_id = self.client.put(frame)
                self.store_put_latencies.append(time.perf_counter() - t0)

                # --- Step 4: Queue put (with frame_num for cross-actor correlation) ---
                t0 = time.perf_counter()
                self.q_out.put([data_id, camera_start, self.frame_num])
                self.queue_put_latencies.append(time.perf_counter() - t0)

                self.camera_latencies.append(time.perf_counter() - frame_time)

                delay = time.perf_counter() - frame_time

                if delay > self.max_delay:
                    self.max_delay = delay

                self.total_delay += delay
                self.frame_count += 1
                self.total_frame_count += 1
                self.frame_num += 1

            except Exception as e:
                logger.warning(f"[Camera {self.camera_name}] Could not put frame in the store | {e}")
                pass

            if self.frame_count % 600 == 0 and self.frame_count > 0:
                total_time = time.perf_counter() - self.start_time

                logger.info(f"[Camera {self.camera_name}] reader FPS: {round(self.frame_count / total_time,2)} - avg delay: {self.total_delay/self.frame_count:.4f} - max delay: {self.max_delay:.4f}")

                self.total_delay = 0
                self.max_delay = 0
                self.frame_count = 0
                self.start_time = time.perf_counter()
        self.camera_latenciesFull.append(time.perf_counter() - frame_time)
            
        
        return Gst.FlowReturn.OK

    # @profile
    def __convert_to_numpy(self, data):
        ''' Convert a GStreamer sample to a numpy array.
            Dimensions come from self.out_height/self.out_width/self.bpp, cached
            once in open_device() -- the frame is always negotiated to exactly
            that size by the caps_out capsfilter, so there is nothing to look up
            per-frame. (Previously queried via caps.get_structure(0).get_value(),
            which was ~1.9 ms of GObject introspection per frame.)
            Adapted from https://gist.github.com/cbenhagen/76b24573fa63e7492fb6#file-gst-appsink-opencv-py-L34
        '''
        return np.ndarray((self.out_height, self.out_width, self.bpp), buffer=data, dtype=np.uint8)
    
    def stop_pipeline(self):
        stop_time = time.perf_counter()

        self.sharing_on = False
        self.stop_program = True
        
        self.pipeline.set_state(Gst.State.PAUSED)
        self.pipeline.set_state(Gst.State.READY)
        self.pipeline.set_state(Gst.State.NULL)

        if hasattr(self, 'total_start_time'):
            recording_duration = stop_time - self.total_start_time
            logger.info(f"[Camera {self.camera_name}] reader stopped. Total frames: {self.total_frame_count} - Recording duration: {recording_duration:.2f}s ({round(recording_duration/60,1)} min)")
        else:
            logger.info(f"[Camera {self.camera_name}] reader stopped. Total frames: {self.total_frame_count}")

        # Total latencies (legacy)
        np.save(self.out_folder / f"TISlatencies_{self.camera_name}.npy", self.camera_latencies)
        np.save(self.out_folder / f"TISstarts_{self.camera_name}.npy", self.cameraStarts)
        np.save(self.out_folder / f"TISlatenciesFull_{self.camera_name}.npy", self.camera_latenciesFull)

        # Per-step breakdowns
        np.save(self.out_folder / f"TIS_convert_{self.camera_name}.npy", self.convert_latencies)
        np.save(self.out_folder / f"TIS_encode_{self.camera_name}.npy", self.encode_latencies)
        np.save(self.out_folder / f"TIS_store_put_{self.camera_name}.npy", self.store_put_latencies)
        np.save(self.out_folder / f"TIS_queue_put_{self.camera_name}.npy", self.queue_put_latencies)

        logger.info(f"TIS latencies saved to {self.out_folder}")

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