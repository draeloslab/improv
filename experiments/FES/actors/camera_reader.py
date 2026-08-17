import yaml
import math
import time
from multiprocessing import Value, RawArray, Process

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
from . import cpu_affinity
from .TIS import *

class CameraReader(ManagedActor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.camera_num = kwargs['camera_num']

    def setup(self):
        """ Initializes the camera reading process """
        logger.info("setup init")

        # Claim a physical P-core before GStreamer builds its pipeline, so the
        # debayer + videoscale threads inherit it. These run at 30 Hz on
        # 1920x1080 Bayer and sit directly in the measured end-to-end path, so
        # they want P-core clocks too -- but they claim cores *after* the
        # processors, which get first pick of the fastest ones.
        cpu_affinity.pin_actor(cpu_affinity.CAPTURE, slot=self.camera_num,
                               label=f"CameraReader cam{self.camera_num}")

        # store init
        self._getStoreInterface()

        # load the configuration file
        source_folder = Path(__file__).resolve().parent.parent

        with open(f'{source_folder}/config/camera_config.yaml', 'r') as file:
            config = yaml.safe_load(file)

        camera_params = config['camera_params']

        self.frame_w = camera_params['resolution']['width'] # native capture width
        self.frame_h = camera_params['resolution']['height'] # native capture height
        # Downscaled resolution GStreamer delivers to the rest of the pipeline.
        # Falls back to the native resolution (no scaling) if not configured.
        stream_res = camera_params.get('stream_resolution', camera_params['resolution'])
        self.stream_w = stream_res['width']
        self.stream_h = stream_res['height']
        self.fps = camera_params['fps'] # FPS
        
        self.start_camera_read = False # flag to start the camera reading
        self.stop_program = Value('b', False) # flag to stop the program
        self.total_frames = 0 # total frames read

        # camera setup
        self.camera_interface = None

        camera_config = config['active_cameras'][self.camera_num]['camera']
        self.camera_name = camera_config['name']

        logger.info(f'Initializing camera {self.camera_name}')

        # initializing the camera frame and socket for communication
        shared_frame = RawArray('B', self.frame_w * self.frame_h * 4) # shared memory for frames reading

        # initializing each camera TIS interface and opening the device
        logger.info(f'Opening device: {camera_config}')

        self.camera_interface = TIS(self.camera_name, self.client, self.q_out)
        self.camera_interface.open_device(camera_config['serial_id'], shared_frame, self.frame_w, self.frame_h, self.fps, SinkFormats.RGB, showvideo=False, out_width=self.stream_w, out_height=self.stream_h)
        
        logger.info(f'Device {self.camera_name} opened')

        # --- Capture phase stagger -------------------------------------------
        # The cameras free-run, and whatever relative phase they happen to start
        # in, they hold (measured: circular concentration 0.98 over a 52 s run).
        # In run 20260804-1432 they landed badly clustered -- three of the four
        # submitted their frame to the GPU within 4.2 ms of each other:
        #
        #   cam2 @ 7.68 ms   cam3 @ 8.48 ms   cam1 @ 11.90 ms   cam0 @ 19.98 ms
        #
        # All four then blocked until the GPU had drained the whole pile. Their
        # inference calls STARTED across a 12.3 ms window but FINISHED within a
        # 3.1 ms one, so the earliest submitter waited longest and its timer
        # recorded that wait as "inference time":
        #
        #   cam2 22.13 ms > cam3 21.31 > cam1 17.89 > cam0 11.07
        #   corr(start phase, measured inference time) = -0.999
        #
        # That ordering is queue position, not a property of any camera. Pure
        # GPU time is 5.4 ms per frame, so four cameras need 21.6 ms of the
        # 33.3 ms cycle -- 65% utilisation, which fits comfortably if the
        # arrivals are spread out instead of bunched. Starting each camera one
        # Nth of a frame period apart is what spreads them.
        #
        # NOTE for the 3D work: this is the right thing *today*, when each
        # camera independently produces its own 2D angle. It is exactly wrong
        # for triangulation, which needs the views to be simultaneous. Once
        # cameras are hardware-triggered in sync, every frame arrives at once by
        # construction and the pile-up comes back -- at which point the answer
        # is one process doing a single batched forward pass, not N processes.
        # See MULTICAM_3D_PLAN.md.
        stagger_cfg = camera_params.get('capture_stagger', {}) or {}
        if stagger_cfg.get('enabled', True):
            n_slots = int(stagger_cfg.get('slots',
                                          cpu_affinity.load_config().get('max_camera_slots', 4)))
            fps_val = float(str(self.fps).split('/')[0]) / float(
                str(self.fps).split('/')[1]) if '/' in str(self.fps) else float(self.fps)
            period = 1.0 / fps_val
            offset = (self.camera_num % max(n_slots, 1)) * period / max(n_slots, 1)

            # Align to a wall-clock epoch every reader computes identically, so
            # the offsets are relative to a shared origin rather than to each
            # actor's own (variable) setup time.
            lead = float(stagger_cfg.get('lead_seconds', 2.0))
            target = math.ceil(time.time() + lead) + offset
            now = time.time()
            if now > target:  # setup ran long; keep the phase, take a later cycle
                target += math.ceil((now - target) / period) * period
            logger.info(f'Camera {self.camera_name} (cam{self.camera_num}): staggering '
                        f'capture start by {offset * 1000:.1f} ms, waiting '
                        f'{(target - now) * 1000:.0f} ms')
            time.sleep(max(0.0, target - time.time()))

        # starting the camera pipeline
        ret_sp = self.camera_interface.start_pipeline()
        logger.info(f'Camera {self.camera_name} (cam{self.camera_num}): pipeline PLAYING '
                    f'at t={time.time():.6f} (phase '
                    f'{(time.time() % (1.0 / 30)) * 1000:.2f} ms) -- compare these across '
                    f'cameras to confirm the stagger took effect')
        if ret_sp:
            logger.info(f'Device {self.camera_name} pipeline started')
        else:
            # tcambin will happily "open" a camera that is not attached, so a failure to
            # reach PLAYING is the first point at which a missing device is detectable.
            # Fail here instead of running on with a dead pipeline that yields 0 frames.
            msg = (f'Device {self.camera_name} (camera_num {self.camera_num}, '
                   f"serial {camera_config['serial_id']}) pipeline could not be started - "
                   f'is the camera plugged in?')
            logger.error(msg)
            raise RuntimeError(msg)

    def runStep(self):
        if not self.start_camera_read:
            self.camera_interface.start_sharing()
            self.start_camera_read = True

            start_time = time.perf_counter()
            logger.info(f"[Camera {self.camera_name}] Run started at {start_time}")

    def stop(self):
        """Trivial stop function for testing purposes."""
        logger.info(f"[Camera {self.camera_name}] - CameraReader stopping")   

        self.stop_program.value = True
        if self.camera_interface is not None:
            self.camera_interface.stop_pipeline()
        logger.info(f"[Camera {self.camera_name}] - CameraReader stopped")   