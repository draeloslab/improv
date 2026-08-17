import sys
import numpy as np
import threading
import queue  
from PyQt5.QtWidgets import QApplication, QLabel, QWidget, QGridLayout, QMessageBox
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QBrush, QFont
import cv2  
import time
import traceback
from pathlib import Path
import yaml
import pyqtgraph as pg
from collections import deque

import logging
from .run_paths import get_logger

logger = get_logger(__name__, "video_screen.log")
logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)


class CameraStreamWidget(QWidget):
    """PyQt Widget for displaying multiple camera streams."""

    def __init__(self, visual, comm, q_sig):
        
        try:
            super().__init__()
            self.visual = visual
            self.comm = comm  
            self.q_sig = q_sig
            self.stop_program = False
            self.last_frame_ids = [None for _ in range(self.visual.num_cameras)]
            self.recent_angles = deque(maxlen=5)
            self.predictions = None
            self.last_frame = [None for _ in range(self.visual.num_cameras)]
            self.last_predictions = [None for _ in range(self.visual.num_cameras)]
            self.last_angles = [0.0 for _ in range(self.visual.num_cameras)]

            # Angle plot state, keyed by real camera_num (not GUI slot index --
            # see getLastFrame's camera_num return). Populated lazily the first
            # time each camera_num is seen, so this handles 1..4 cameras with no
            # hardcoded "camera 0" / "camera 2" special-casing. All cameras
            # share one y-axis (see update_angle_plot) rather than the old
            # two-viewbox hack, which does not generalize past 2 series.
            self.angle_history = {}   # camera_num -> list of recent angles (capped at 100)
            self.angle_curves = {}    # camera_num -> pg.PlotDataItem
            self.angle_y_min = np.inf
            self.angle_y_max = -np.inf
            # Fixed categorical order, assigned to camera_nums in the order
            # they're first seen -- never reassigned/cycled once a camera has a
            # colour, so a curve's identity/colour is stable for the whole run.
            self._angle_palette = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100',
                                   '#e87ba4', '#008300', '#4a3aa7', '#e34948']
            self._angle_palette_idx = 0


            # Load the configuration file
            source_folder = Path(__file__).resolve().parent.parent
            with open(f'{source_folder}/config.yaml', 'r') as file:
                config = yaml.safe_load(file)

            self.resize = config['resize']
            self.threshold = config['threshold']

            # Set up GUI layout
            self.setWindowTitle('Camera Streams')
            self.setGeometry(100, 100, 1920, 1080)

            # Layout to hold the camera labels in a 2x3 Grid
            layout = QGridLayout()
            
            # 2 Rows
            layout.setRowStretch(0, 1)
            layout.setRowStretch(1, 1)
            
            # 3 Columns
            layout.setColumnStretch(0, 1)
            layout.setColumnStretch(1, 1)
            layout.setColumnStretch(2, 1)

            # Create labels to show camera frames
            self.camera_labels = [QLabel(self) for _ in range(self.visual.num_cameras)]
            for label in self.camera_labels:
                label.setMinimumSize(320, 240)
            
            # Map up to 5 cameras to the grid layout dynamically
            # (Row, Column) tuples for the first 5 slots
            camera_positions = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1)]
            for i, label in enumerate(self.camera_labels):
                if i < len(camera_positions):
                    row, col = camera_positions[i]
                    layout.addWidget(label, row, col)

            # Add a PyQtGraph PlotWidget for the angle plot. One shared y-axis
            # for every active camera (up to 4), each camera as its own
            # coloured curve with a legend -- generalizes cleanly to however
            # many cameras this run has, unlike a per-camera secondary y-axis.
            self.angle_plot_widget = pg.PlotWidget()
            self.angle_plot_widget.setMinimumSize(640, 480)
            self.angle_plot_widget.setBackground('w')
            self.angle_plot_widget.setTitle("Live Angle Plot", color='k')
            self.angle_plot_widget.setLabel('bottom', 'Frame', color='k')
            self.angle_plot_widget.setLabel('left', 'Angle (°)', color='k')
            self.angle_plot_widget.addLegend()

            # Always place the plot in the bottom-right corner (Row 1, Col 2)
            layout.addWidget(self.angle_plot_widget, 1, 2)

            self.setLayout(layout)

            # Initialize a QTimer to update frames
            fps = getattr(self.visual, 'frame_rate_update', 30)
            update_interval = int(1000 / fps) if fps > 0 else 33
            self.timer = QTimer()
            self.timer.timeout.connect(self.update_frames)
            self.timer.start(update_interval)

            logger.info(f'Front End Setup completed')
        except Exception as e:
            logger.info(f'Setup failed due to {e}')
            traceback.format_exc()

    def _new_angle_curve(self, cam):
        """Lazily create a curve for a camera_num the first time its angle is seen."""
        colour = self._angle_palette[self._angle_palette_idx % len(self._angle_palette)]
        self._angle_palette_idx += 1
        curve = self.angle_plot_widget.plot(pen=pg.mkPen(colour, width=2), name=f"Camera {cam}")
        self.angle_curves[cam] = curve
        self.angle_history[cam] = []

    def update_frames(self):
        """Update frames from each camera"""
        for camera_id in range(self.visual.num_cameras):
            frame = None
            predictions = None
            angle = None
            camera_num = camera_id
            try:
                frame, predictions, angle, camera_num = self.visual.getLastFrame(camera_id)
                if frame is not None:
                    self.last_frame[camera_id] = frame

                # Cache the predictions if they're valid
                if predictions is not None:
                    self.last_predictions[camera_id] = predictions

                # Cache the angle if it's valid
                if angle is not None:
                    self.last_angles[camera_id] = angle

                # Use cached predictions if current ones are None
                display_predictions = predictions if predictions is not None else self.last_predictions[camera_id]

                # Use cached angle if current one is None
                display_angle = angle if angle is not None else self.last_angles[camera_id]

                self.display_frame(self.last_frame[camera_id], display_predictions, self.camera_labels[camera_id], display_angle, camera_id)

                # Update the angle plot if an angle is provided. Keyed by real
                # camera_num (not the GUI slot loop index camera_id), so this
                # handles however many cameras are active -- 1 to 4 -- with no
                # per-slot special-casing.
                if angle is not None and np.isfinite(angle):
                    if camera_num not in self.angle_curves:
                        self._new_angle_curve(camera_num)
                    hist = self.angle_history[camera_num]
                    hist.append(angle)
                    if len(hist) > 100:
                        hist.pop(0)
                    if angle > self.angle_y_max:
                        self.angle_y_max = angle
                    if angle < self.angle_y_min:
                        self.angle_y_min = angle
                    self.update_angle_plot()

            except Exception as e:
                blank_frame = np.zeros((self.visual.frame_h, self.visual.frame_w, 3), dtype=np.uint8)
                self.display_frame(blank_frame, None, self.camera_labels[camera_id], 0.0, camera_id)
                logger.info(f"No frame available for camera {camera_id}")

    def display_frame(self, frame, predictions, label, angle, camera_id=None):
        """Convert frame to QImage, plot predictions if available, and display it in QLabel."""
        if frame is None:
            return
        
        # Make a copy to ensure data isn't garbage collected
        frame = np.ascontiguousarray(frame)
        height, width, channel = frame.shape
        bytes_per_line = channel * width
        q_img = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888).copy()
        
        # Initialize painter for drawing on the frame
        painter = QPainter(q_img)
        
        # Draw predictions if available
        if predictions is not None:
            painter.setBrush(QBrush(QColor(255, 0, 0)))

            # Bodypart labels are chosen from how many keypoints the model
            # actually returned this frame, not from camera_id -- this is
            # self-describing and needs no camera-specific config, and it
            # doesn't break if a camera_num<->model mapping ever changes.
            labels = ["MRS"] if len(predictions) == 1 else ["DIP", "PIP", "MCP", "Wrist"]

            prev_point = None
            for i, point in enumerate(predictions):
                x, y, likelihood = point
                # Draw every keypoint the model returns, regardless of confidence.
                # Only genuinely undrawable values (NaN from a failed PAF assembly,
                # or DLC's -1/-2 "no detection" sentinel) are skipped -- there is no
                # coordinate to draw in those cases. Low-confidence keypoints are
                # drawn hollow and dimmer so they are still visibly distinguishable
                # from confident ones.
                if not (np.isfinite(x) and np.isfinite(y)) or x < -1.5 or y < -1.5:
                    prev_point = None  # break the skeleton line across a missing joint
                    continue
                x = x/self.resize
                y = y/self.resize
                confident = likelihood > self.threshold
                colour = QColor(255, 0, 0) if confident else QColor(255, 165, 0)
                painter.setPen(QPen(colour, 2 if confident else 1))
                painter.drawEllipse(int(x), int(y), 50, 50)
                painter.setPen(QPen(QColor(255, 255, 255) if confident else QColor(200, 200, 200), 2))
                painter.setFont(QFont("Arial", 50))
                painter.drawText(int(x) + 20, int(y) + 20, labels[i % len(labels)])
                # Draw lines between points
                if prev_point is not None:
                    painter.drawLine(int(prev_point[0]), int(prev_point[1]), int(x), int(y))
                prev_point = (x, y)

        # Always draw angle text on every frame
        painter.setPen(QPen(QColor(0, 255, 0), 2))
        painter.setFont(QFont("Arial", 50))
        # `angle is not None` alone let NaN through and rendered a literal "nan°";
        # check for a real number instead.
        angle_ok = angle is not None and np.isfinite(angle)
        angle_text = f"Angle: {angle:.2f}°" if angle_ok else "Angle: N/A"
        painter.drawText(10, 50, angle_text)
        painter.end()

        pixmap = QPixmap.fromImage(q_img)
        scaled_pixmap = pixmap.scaled(label.size(), Qt.KeepAspectRatio)
        label.setPixmap(scaled_pixmap)

    def update_angle_plot(self):
        """Update the live plot of angles using PyQtGraph -- one curve per
        active camera_num, sharing a single y-axis."""
        for cam, curve in self.angle_curves.items():
            curve.setData(self.angle_history[cam])

        if self.angle_y_min != np.inf and self.angle_y_max != -np.inf:
            self.angle_plot_widget.setYRange(self.angle_y_min - 5, self.angle_y_max + 5)

    def closeEvent(self, event):
        '''Clicked x/close on window - save latencies before closing'''
        logger.info("CameraStreamWidget closeEvent triggered")
        self.visual.stopMe()

        self.comm.put(['stop'])
        logger.info("Closing CameraStreamWidget")
        event.accept()