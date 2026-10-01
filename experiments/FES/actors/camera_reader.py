"""CameraReader: one improv actor per TIS (The Imaging Source) USB camera.

Opens the camera named in config/camera_config.yaml `active_cameras[camera_num]`, applies the fixed exposure / gain /
white-balance settings, and from the first runStep on puts every frame into the improv store and sends
[store id, camera_start, frame_num, capture_time] on q_out (see actors/TIS.py for what each field means).
"""
import math
import re
import time
from pathlib import Path

import yaml
from improv.actor import ManagedActor

from . import cpu_affinity
from .TIS import TIS, SinkFormats
from .run_paths import get_logger

logger = get_logger(__name__, "camera_reader.log")


class CameraReader(ManagedActor):
    """Streams one camera into the store. Graph kwargs: camera_num (index into camera_config.yaml active_cameras)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.camera_num = kwargs['camera_num']
        self.camera_interface = None
        self.started = False

    def setup(self):
        """Pin a CPU core, open the camera's GStreamer pipeline, optionally stagger its start, apply settings."""
        # Claim a P-core before GStreamer builds its pipeline so the debayer/scale threads inherit it. The slot
        # is the wiring order (the N in GeneratorN), which is dense 0..k-1 unlike camera numbers.
        m = re.search(r"(\d+)$", str(self.name))
        slot = int(m.group(1)) if m else self.camera_num
        cpu_affinity.pin_actor(cpu_affinity.CAPTURE, slot=slot, label=f"CameraReader cam{self.camera_num}")

        self._getStoreInterface()
        with open(Path(__file__).resolve().parent.parent / 'config' / 'camera_config.yaml') as f:
            config = yaml.safe_load(f)
        params = config['camera_params']
        native = params['resolution']
        stream = params.get('stream_resolution') or native          # GStreamer downscales to this before Python
        self.fps = params['fps']
        camera_config = config['active_cameras'][self.camera_num]['camera']
        self.camera_name = camera_config['name']
        logger.info(f'Opening camera {self.camera_name}: {camera_config}')

        self.camera_interface = TIS(self.camera_name, self.client, self.q_out)
        self.camera_interface.open_device(camera_config['serial_id'], native['width'], native['height'], self.fps,
                                          SinkFormats.RGB, showvideo=False, out_width=stream['width'],
                                          out_height=stream['height'], max_buffers=int(params.get('appsink_max_buffers', 5)))
        self._stagger(params.get('capture_stagger') or {})

        if not self.camera_interface.start_pipeline():
            # tcambin "opens" a camera that is not attached; reaching PLAYING is the first real check.
            logger.error(f"Camera {self.camera_name} (camera_num {self.camera_num}, serial {camera_config['serial_id']}) "
                         f"pipeline could not be started - is the camera plugged in?")
            return
        logger.info(f'Camera {self.camera_name}: pipeline PLAYING at t={time.time():.6f} '
                    f'(phase {(time.time() % (1.0 / 30)) * 1000:.2f} ms)')
        # Fixed exposure/gain/white balance: on auto, exposure drifts with the scene and can outrun the frame period.
        settings = dict(params.get('camera_settings') or {})
        settings.update(camera_config.get('settings') or {})
        if settings:
            self.camera_interface.apply_settings(settings)

    def _stagger(self, cfg):
        """Start this camera 1/slots of a frame period after the others (capture_stagger in camera_config.yaml).

        Only useful for per-camera 2D pipelines sharing one GPU (spreads the inference requests). Keep it off for 3D:
        triangulation wants the cameras as close in phase as possible.
        """
        if not cfg.get('enabled', False):
            return
        n = max(int(cfg.get('slots', cpu_affinity.load_config().get('max_camera_slots', 4))), 1)
        num, _, den = str(self.fps).partition('/')
        period = float(den or 1) / float(num)
        offset = (self.camera_num % n) * period / n
        target = math.ceil(time.time() + float(cfg.get('lead_seconds', 2.0))) + offset   # shared wall-clock origin
        now = time.time()
        if now > target:
            target += math.ceil((now - target) / period) * period
        logger.info(f'Camera {self.camera_name}: staggering capture start by {offset * 1000:.1f} ms')
        time.sleep(max(0.0, target - time.time()))

    def runStep(self):
        """The first step starts sending frames (frames before the run starts are dropped)."""
        if not self.started:
            self.camera_interface.start_sharing()
            self.started = True
            logger.info(f"[Camera {self.camera_name}] run started")

    def stop(self):
        """Stop the pipeline and save the camera's timing logs."""
        if self.camera_interface is not None:
            self.camera_interface.stop_pipeline()
        logger.info(f"[Camera {self.camera_name}] CameraReader stopped")
