"""improv actor pair backing the 3D GUI (actors/front_end_3d.Camera3DWidget).

Same shape as video_screen.Visual/VideoScreen, but the prediction side is a
single link carrying one dict for all cameras (from ProcessorBatch3D) rather
than one [pred, angle] message per camera.
"""

import os
import queue
import time
import traceback
import logging
from pathlib import Path

import cv2
import numpy as np
import yaml

from improv.actor import ManagedActor, Actor, Signal
from . import cpu_affinity
from .front_end_3d import Camera3DWidget

os.environ['QT_API'] = 'pyqt5'
from PyQt5 import QtWidgets  # noqa: E402

# cv2 (imported above) points QT_QPA_PLATFORM_PLUGIN_PATH at its own bundled
# Qt plugins, which are built against cv2's private vendored Qt5 libs -- not
# the PyQt5 Qt5 build QApplication actually runs on. Loading cv2's "xcb"
# plugin into a PyQt5 app fails with "found but could not be initialized"
# because of the ABI mismatch, so QApplication() never returns and this actor
# never sends ready. Point it back at PyQt5's own plugins before any
# QApplication is constructed.
os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = str(
    Path(QtWidgets.__file__).resolve().parent / "Qt5" / "plugins"
)

from .run_paths import get_logger, run_folder

logger = get_logger(__name__, "video_screen.log", level=logging.DEBUG,
                    handler_level=logging.INFO)


class Visual3D(Actor):
    def setup(self, visual):
        self.visual = visual
        self.visual.setup()
        logger.info("Running setup for " + self.name)

    def run(self):
        import os as _os
        def _chk(msg):
            _os.write(2, f"[CHK-run] {msg}\n".encode())
        _chk("run entered")
        logger.info("Loading 3D FrontEnd")
        _chk("before QApplication")
        self.app = QtWidgets.QApplication([])
        _chk("after QApplication, before widget")
        self.viewer = Camera3DWidget(self.visual, self.q_comm, self.q_sig)
        _chk("after widget, before show")
        self.viewer.show()
        _chk("after show, before ready puts")
        self.q_comm.put([Signal.ready()])
        self.visual.q_comm.put([Signal.ready()])
        _chk("before exec_")
        self.app.exec_()
        logger.info("3D GUI ready")


class VideoScreen3D(ManagedActor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_cameras = kwargs['num_active_cameras']

    def setup(self):
        cpu_affinity.pin_actor(cpu_affinity.BACKGROUND, label="VideoScreen3D")
        self._getStoreInterface()

        self.stop_program = False
        self.start_program = False

        source_folder = Path(__file__).resolve().parent.parent
        with open(f'{source_folder}/config/camera_config.yaml', 'r') as file:
            config = yaml.safe_load(file)

        camera_params = config['camera_params']
        stream_res = camera_params.get('stream_resolution', camera_params['resolution'])
        self.frame_w = stream_res['width']
        self.frame_h = stream_res['height']

        fps_str = str(camera_params.get('fps', 30))
        if '/' in fps_str:
            num, den = fps_str.split('/')
            self.frame_rate_update = int(int(num) / int(den))
        else:
            self.frame_rate_update = int(fps_str)

        self.frame_latencies = []
        self.pred_latencies = []
        self.videoStarts = []

        self.out_folder = run_folder()
        logger.info(f"3D Video GUI setup completed, {self.num_cameras} cameras")

    def getLastFrame(self, camera_id):
        """Newest image for one camera. Returns the frame or None."""
        self.videoStarts.append(time.time())
        frame_start = time.perf_counter()
        try:
            msg = self.links[f"images{camera_id}_in"].get(timeout=0.005)
            frame_id = msg[0]
            if frame_id is None:
                return None
            frame = self.client.get(frame_id)
            if not (isinstance(frame, np.ndarray) and frame.ndim == 3):
                frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)
            self.frame_latencies.append(time.perf_counter() - frame_start)
            return frame
        except (queue.Empty, KeyError):
            return None
        except Exception:
            logger.debug(f"error getting frame for camera {camera_id}: "
                         f"{traceback.format_exc()}")
            return None

    def getLastPrediction(self):
        """Newest ProcessorBatch3D message: one dict covering every camera.

        Drained to the most recent one -- the GUI redraws on its own timer and
        only ever shows the latest state, so queued older frames are dead weight
        and would otherwise make the display lag further behind the more it
        falls behind.
        """
        pred_start = time.perf_counter()
        msg = None
        try:
            link = self.links["preds_in"]
        except KeyError:
            return None
        try:
            msg = link.get(timeout=0.005)
        except (queue.Empty, KeyError):
            return None
        except Exception:
            logger.debug(f"error getting prediction: {traceback.format_exc()}")
            return None
        while True:
            try:
                msg = link.get_nowait()
            except Exception:
                break
        self.pred_latencies.append(time.perf_counter() - pred_start)
        return msg

    def runStep(self):
        self.start_program = True

    def stopMe(self):
        logger.info(f"{self.name}: Stopping 3D Video GUI")
        self.stop_program = True
        try:
            np.save(self.out_folder / "vizframelatencies.npy", self.frame_latencies)
            np.save(self.out_folder / "vizpredictionslatencies.npy", self.pred_latencies)
            np.save(self.out_folder / "vizStarts.npy", self.videoStarts)
            logger.info(f"{self.name}: 3D Video GUI stopped")
        except Exception:
            logger.info(f'Could not save latencies: {traceback.format_exc()}')
