"""GUI for the batched 3D pipeline: N camera views + all joint angles + a live 3D skeleton.

Layout (3 x 2 grid):

    cam0        cam1        joint-angle plot   (one curve per joint)
    cam2        cam3        3D skeleton        (software-projected, no GL)

Differs from front_end5.CameraStreamWidget in what it consumes: that one reads
a per-camera [prediction, angle] message from one Processor per camera, this one
reads a single dict from ProcessorBatch3D carrying every camera's 2D keypoints,
the shared 3D keypoints, and the full joint-angle set.

The 3D panel deliberately does NOT use pyqtgraph.opengl / GLViewWidget.
improv.nexus always launches the GUI actor with a plain fork()'d
multiprocessing.Process (Nexus.createActor's GUI branch hardcodes
Process(target=...), ignoring the yaml's `method:` -- there is no way to make
this actor's process use spawn instead). Forking a process and then creating an
EGL/GLX context in the child is a well-known deadlock hazard (GL drivers hold
internal locks/threads that fork() does not carry over cleanly), and that is
exactly what happened here: GLViewWidget() inside a forked GUI process hung
forever before ever reaching Qt's event loop, well before any exception could
even be logged. A few dozen 3D points don't need a GPU anyway, so this instead
projects them with a small numpy rotation matrix and draws the result as a
plain 2-D pyqtgraph plot -- no GL context, no fork hazard, identical visual
result for a skeleton this size.
"""

import os
import traceback
from pathlib import Path

import numpy as np
import yaml
import cv2

# pyqtgraph auto-detects a Qt binding from whatever's already imported, and
# this env also has PySide6 installed -- left to its own devices it grabs
# PySide6, while video_screen_3d.py builds the actual QApplication with
# PyQt5. Two separate Qt bindings in one process means pyqtgraph's widgets
# belong to a binding with no QApplication, which crashes with "QWidget:
# Must construct a QApplication before a QWidget". Pin it before importing.
os.environ.setdefault('PYQTGRAPH_QT_LIB', 'PyQt5')
import pyqtgraph as pg
from PyQt5.QtWidgets import QLabel, QWidget, QGridLayout, QSizePolicy
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QBrush, QFont

import logging
from .run_paths import get_logger

logger = get_logger(__name__, "video_screen.log")
logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)


def _rotation_matrix(elevation_deg, azimuth_deg):
    """Camera-style rotation: azimuth about Z, then elevation about the new X.
    Matches matplotlib's Axes3D.view_init(elev, azim) convention closely enough
    for a static viewing angle -- this is not meant to be user-orbitable."""
    el, az = np.radians(elevation_deg), np.radians(azimuth_deg)
    cz, sz = np.cos(az), np.sin(az)
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    ce, se = np.cos(el), np.sin(el)
    Rx = np.array([[1, 0, 0], [0, ce, -se], [0, se, ce]])
    return Rx @ Rz

# Distinct-enough colours for up to ~16 joint-angle curves. Assigned in the
# order joints are first seen and never reassigned, so a curve's colour is
# stable for the whole run.
_JOINT_PALETTE = [
    '#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300',
    '#4a3aa7', '#e34948', '#00a6b5', '#9b5de5', '#8d6e3a', '#d81b8c',
    '#5a8f00', '#0072b2', '#cc6677', '#44aa99',
]


class Camera3DWidget(QWidget):
    """Camera streams + every joint angle + the triangulated hand in 3D."""

    def __init__(self, visual, comm, q_sig):
        import os as _os
        def _chk(msg):
            _os.write(2, f"[CHK] {msg}\n".encode())
        _chk("__init__ entered")
        try:
            super().__init__()
            _chk("init start")

            self.visual = visual
            self.comm = comm
            self.q_sig = q_sig
            self.stop_program = False
            _chk("basic attrs set")

            n = self.visual.num_cameras
            self.last_frame = [None] * n
            self.last_points_2d = [None] * n

            # Joint-angle plot state, keyed by joint name.
            self.angle_history = {}
            self.angle_curves = {}
            self._palette_idx = 0
            self.angle_y_min = np.inf
            self.angle_y_max = -np.inf
            self.history_len = 150

            self.last_points_3d = None
            self.skeleton_idx = []

            source_folder = Path(__file__).resolve().parent.parent
            _chk("before config read")
            with open(f'{source_folder}/config.yaml', 'r') as file:
                config = yaml.safe_load(file)
            self.resize = config['resize']
            self.threshold = config['threshold']

            _chk("before setWindowTitle")
            self.setWindowTitle('3D Hand Tracking')
            self.setGeometry(100, 100, 1920, 1080)

            layout = QGridLayout()
            for r in range(2):
                layout.setRowStretch(r, 1)
            for c in range(3):
                layout.setColumnStretch(c, 1)

            # --- camera views: left 2x2 block ---
            _chk("before camera labels")
            self.camera_labels = [QLabel(self) for _ in range(n)]
            for label in self.camera_labels:
                label.setMinimumSize(320, 240)
                # Without an expanding policy the label keeps its minimum size
                # and the 960x540 pixmap is scaled down into a corner of the
                # cell, wasting most of the window.
                label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
                label.setAlignment(Qt.AlignCenter)
            camera_positions = [(0, 0), (0, 1), (1, 0), (1, 1)]
            for i, label in enumerate(self.camera_labels):
                if i < len(camera_positions):
                    row, col = camera_positions[i]
                    layout.addWidget(label, row, col)

            # --- joint angle plot: top right ---
            _chk("before angle plot widget")
            self.angle_plot_widget = pg.PlotWidget()
            self.angle_plot_widget.setMinimumSize(480, 320)
            self.angle_plot_widget.setBackground('w')
            self.angle_plot_widget.setTitle("Joint Angles (deg, 0 = extended)", color='k')
            self.angle_plot_widget.setLabel('bottom', 'Frame', color='k')
            self.angle_plot_widget.setLabel('left', 'Angle (deg)', color='k')
            # 15 joints in a single-column legend covers a third of the plot and
            # hides the curves it is meant to explain. Lay it out across columns
            # so it becomes a short band instead of a tall block.
            legend = self.angle_plot_widget.addLegend(offset=(10, 8), labelTextSize='7pt',
                                                      colCount=3)
            try:
                legend.setBrush(pg.mkBrush(255, 255, 255, 200))
            except Exception:
                pass  # older pyqtgraph: legend just stays transparent
            layout.addWidget(self.angle_plot_widget, 0, 2)

            # --- 3D skeleton: bottom right. Software-projected (see module
            # docstring for why this isn't pyqtgraph.opengl.GLViewWidget). ---
            _chk("before 3d plot widget")
            self.plot3d_widget = pg.PlotWidget()
            self.plot3d_widget.setMinimumSize(480, 320)
            self.plot3d_widget.setBackground('k')
            self.plot3d_widget.setTitle("3D Hand Skeleton", color='w')
            self.plot3d_widget.setAspectLocked(True)
            self.plot3d_widget.showGrid(x=True, y=True, alpha=0.2)
            self.plot3d_widget.getPlotItem().hideAxis('left')
            self.plot3d_widget.getPlotItem().hideAxis('bottom')
            # Fixed viewing angle (matches the calibration notebook's
            # elev=18,azim=-70 static view) -- not orbitable, but a live hand
            # skeleton doesn't need to be, and it keeps this simple.
            self._proj_R = _rotation_matrix(elevation_deg=18, azimuth_deg=-70)
            self.kp_scatter = pg.ScatterPlotItem(
                pen=pg.mkPen('k'), brush=pg.mkBrush(255, 80, 80, 255), size=10)
            self.plot3d_widget.addItem(self.kp_scatter)
            # One PlotCurveItem per bone -- like the old GLLinePlotItem pool,
            # each is independently shown/hidden depending on whether both its
            # keypoints triangulated this frame.
            self.bone_items = []
            layout.addWidget(self.plot3d_widget, 1, 2)

            _chk("before setLayout")
            self.setLayout(layout)

            fps = getattr(self.visual, 'frame_rate_update', 30)
            update_interval = int(1000 / fps) if fps > 0 else 33
            _chk("before QTimer")
            self.timer = QTimer()
            self.timer.timeout.connect(self.update_frames)
            self.timer.start(update_interval)

            _chk("before final log")
            logger.info("3D front end setup completed")
        except Exception as e:
            logger.error(f"3D front end setup failed: {e}")
            logger.error(traceback.format_exc())

    # ------------------------------------------------------------------ update

    def update_frames(self):
        # Camera images first -- these arrive on their own per-camera links and
        # are independent of whether a prediction landed this tick.
        for camera_id in range(self.visual.num_cameras):
            try:
                frame = self.visual.getLastFrame(camera_id)
                if frame is not None:
                    self.last_frame[camera_id] = frame
                self.display_frame(self.last_frame[camera_id],
                                   self.last_points_2d[camera_id],
                                   self.camera_labels[camera_id],
                                   camera_id)
            except Exception:
                logger.debug(f"no frame for camera {camera_id}")

        # Then the single prediction message covering all cameras at once.
        try:
            msg = self.visual.getLastPrediction()
        except Exception:
            logger.error(f"prediction fetch failed: {traceback.format_exc()}")
            return
        if msg is None:
            return

        points_2d = msg.get('points_2d')
        if points_2d is not None:
            for i in range(min(len(points_2d), self.visual.num_cameras)):
                self.last_points_2d[i] = points_2d[i]

        skel = msg.get('skeleton_idx')
        if skel:
            self.skeleton_idx = skel

        angles = msg.get('joint_angles') or {}
        names = msg.get('joint_names') or sorted(angles)
        if angles:
            self.update_angle_plot(angles, names)

        p3d = msg.get('points_3d')
        if p3d is not None:
            self.last_points_3d = np.asarray(p3d, dtype=float)
            self.update_3d_plot(self.last_points_3d)

    # -------------------------------------------------------------- 2D drawing

    def display_frame(self, frame, points, label, camera_id):
        """Draw one camera view with its 2D keypoints overlaid."""
        if frame is None:
            return
        frame = np.ascontiguousarray(frame)
        height, width, channel = frame.shape
        q_img = QImage(frame.data, width, height, channel * width,
                       QImage.Format_RGB888).copy()
        painter = QPainter(q_img)

        if points is not None:
            pts = np.asarray(points, dtype=float)
            # 21 keypoints in a 960x540 view: small dots, no per-point text.
            # Labelling every joint here is unreadable and the 3D view is the
            # place to identify them anyway.
            radius = 5
            drawn = [None] * len(pts)
            for i, point in enumerate(pts):
                x, y, likelihood = point[0], point[1], point[2]
                if not (np.isfinite(x) and np.isfinite(y)) or x < -1.5 or y < -1.5:
                    continue
                x = x / self.resize
                y = y / self.resize
                drawn[i] = (x, y)
                confident = likelihood > self.threshold
                colour = QColor(255, 0, 0) if confident else QColor(255, 165, 0)
                painter.setBrush(QBrush(colour))
                painter.setPen(QPen(colour, 1))
                painter.drawEllipse(int(x) - radius, int(y) - radius, radius * 2, radius * 2)

            painter.setPen(QPen(QColor(255, 255, 255), 2))
            for a, b in self.skeleton_idx:
                if a < len(drawn) and b < len(drawn) and drawn[a] and drawn[b]:
                    painter.drawLine(int(drawn[a][0]), int(drawn[a][1]),
                                     int(drawn[b][0]), int(drawn[b][1]))

        painter.setPen(QPen(QColor(0, 255, 0), 2))
        painter.setFont(QFont("Arial", 28))
        painter.drawText(10, 40, f"cam{camera_id}")
        painter.end()

        pixmap = QPixmap.fromImage(q_img)
        label.setPixmap(pixmap.scaled(label.size(), Qt.KeepAspectRatio))

    # ----------------------------------------------------------- angle plotting

    def _new_angle_curve(self, joint):
        colour = _JOINT_PALETTE[self._palette_idx % len(_JOINT_PALETTE)]
        self._palette_idx += 1
        self.angle_curves[joint] = self.angle_plot_widget.plot(
            pen=pg.mkPen(colour, width=2), name=joint)
        self.angle_history[joint] = []

    def update_angle_plot(self, angles, names):
        """One curve per joint, all sharing a single y-axis (they're all degrees)."""
        for joint in names:
            value = angles.get(joint)
            if value is None:
                continue
            if joint not in self.angle_curves:
                self._new_angle_curve(joint)
            hist = self.angle_history[joint]
            # A NaN is a real event here -- the joint had fewer than two
            # confident views this frame. Keep it in the series so the curve
            # breaks rather than drawing a straight line across the gap.
            hist.append(float(value))
            if len(hist) > self.history_len:
                hist.pop(0)
            if np.isfinite(value):
                self.angle_y_min = min(self.angle_y_min, value)
                self.angle_y_max = max(self.angle_y_max, value)

        for joint, curve in self.angle_curves.items():
            curve.setData(np.asarray(self.angle_history[joint], dtype=float),
                          connect='finite')

        if np.isfinite(self.angle_y_min) and np.isfinite(self.angle_y_max):
            self.angle_plot_widget.setYRange(self.angle_y_min - 5, self.angle_y_max + 5)

    # -------------------------------------------------------------- 3D plotting

    def _project(self, pts):
        """Orthographic projection of (N,3) mm points to 2D screen coords."""
        return (self._proj_R @ pts.T).T[:, :2]   # rotated X,Y; drop rotated Z (depth)

    def update_3d_plot(self, points_3d):
        """Scatter the triangulated keypoints and draw the bones between them,
        via a fixed-angle numpy projection instead of a GPU-rendered scene --
        see the module docstring for why."""
        finite = np.isfinite(points_3d).all(axis=1)
        if not finite.any():
            self.kp_scatter.setData(pos=np.zeros((0, 2)))
            for item in self.bone_items:
                item.setVisible(False)
            return

        # Recentre on the hand so it stays in view regardless of where the rig's
        # origin ended up -- the calibration origin is camera 0, which can be a
        # metre away.
        centre = points_3d[finite].mean(axis=0)
        centred = points_3d - centre
        proj = self._project(centred)   # (K, 2), NaN rows where not finite

        self.kp_scatter.setData(pos=proj[finite])

        # Lazily grow the pool of bone curves to match the skeleton.
        while len(self.bone_items) < len(self.skeleton_idx):
            item = pg.PlotCurveItem(pen=pg.mkPen(255, 255, 255, 220, width=2))
            self.plot3d_widget.addItem(item)
            self.bone_items.append(item)

        for k, (a, b) in enumerate(self.skeleton_idx):
            item = self.bone_items[k]
            if a < len(finite) and b < len(finite) and finite[a] and finite[b]:
                item.setData(x=[proj[a, 0], proj[b, 0]], y=[proj[a, 1], proj[b, 1]])
                item.setVisible(True)
            else:
                item.setVisible(False)
        for k in range(len(self.skeleton_idx), len(self.bone_items)):
            self.bone_items[k].setVisible(False)

    def closeEvent(self, event):
        logger.info("Camera3DWidget closeEvent triggered")
        self.visual.stopMe()
        self.comm.put(['stop'])
        event.accept()
