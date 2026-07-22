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
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Create a file handler
log_file = "video_screen.log"
file_handler = logging.FileHandler(log_file)
file_handler.setLevel(logging.INFO)

# Create a formatter and set it for the handler
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)

# Add the handler to the logger
logger.addHandler(file_handler)
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
            self.angles = [0]  
            self.angles_cam2 = [0]  
            self.recent_angles = deque(maxlen=5) 
            self.y_max = -np.inf  
            self.y_min = np.inf  
            self.y_max_cam2 = -np.inf  
            self.y_min_cam2 = np.inf  
            self.predictions = None
            self.last_frame = [None for _ in range(self.visual.num_cameras)]
            self.last_predictions = [None for _ in range(self.visual.num_cameras)]  
            self.last_angles = [0.0 for _ in range(self.visual.num_cameras)]  
            

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

            # Add a PyQtGraph PlotWidget for the angle plot
            self.angle_plot_widget = pg.PlotWidget()
            self.angle_plot_widget.setMinimumSize(640, 480) 
            self.angle_plot_widget.setBackground('w')
            self.angle_plot_widget.setTitle("Live Angle Plot", color='k')
            self.angle_plot_widget.setLabel('bottom', 'Frame', color='k')
            self.angle_plot_widget.setLabel('left', 'Camera 0 Angle (°)', color='r')
            
            # Create second ViewBox for camera 2 with separate y-axis
            self.viewbox2 = pg.ViewBox()
            self.angle_plot_widget.scene().addItem(self.viewbox2)
            self.angle_plot_widget.getAxis('right').linkToView(self.viewbox2)
            self.viewbox2.setXLink(self.angle_plot_widget)
            self.angle_plot_widget.getAxis('right').setLabel('Camera 2 Angle (°)', color='b')
            self.angle_plot_widget.showAxis('right')
            
            # Initialize plot curves
            self.curve_cam0 = self.angle_plot_widget.plot(pen=pg.mkPen('r', width=2))
            self.curve_cam2 = pg.PlotCurveItem(pen=pg.mkPen('b', width=2))
            self.viewbox2.addItem(self.curve_cam2)
            
            # Connect view resize to update secondary viewbox
            self.angle_plot_widget.getViewBox().sigResized.connect(self._update_viewbox2)
            
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

    def _update_viewbox2(self):
        """Keep secondary viewbox geometry in sync with primary plot."""
        self.viewbox2.setGeometry(self.angle_plot_widget.getViewBox().sceneBoundingRect())

    def update_frames(self):
        """Update frames from each camera"""
        for camera_id in range(self.visual.num_cameras):
            frame = None
            predictions = None
            angle = None
            try:
                frame, predictions, angle = self.visual.getLastFrame(camera_id)
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
                
                # Update the angle plot if an angle is provided
                if angle is not None:
                    if camera_id == 0:
                        self.angles.append(angle)
                        if len(self.angles) > 100:
                            self.angles.pop(0)
                        if angle > self.y_max:
                            self.y_max = angle
                        if angle < self.y_min:
                            self.y_min = angle
                    elif camera_id == 2 and self.visual.num_cameras > 2: 
                        self.angles_cam2.append(angle)
                        if len(self.angles_cam2) > 100: 
                            self.angles_cam2.pop(0)
                        if angle > self.y_max_cam2:
                            self.y_max_cam2 = angle
                        if angle < self.y_min_cam2:
                            self.y_min_cam2 = angle
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

            # Set labels based on camera_id
            if camera_id == 2:
                labels = ["MRS"]
            else:
                labels = ["DIP", "PIP", "MCP", "Wrist"]

            prev_point = None
            for i, point in enumerate(predictions):
                x, y, likelihood = point
                x = x/self.resize
                y = y/self.resize
                if likelihood > 0:
                    painter.setPen(QPen(QColor(255, 0, 0), 2)) 
                    painter.drawEllipse(int(x), int(y), 50, 50)
                    painter.setPen(QPen(QColor(255, 255, 255), 2)) 
                    painter.setFont(QFont("Arial", 50)) 
                    painter.drawText(int(x) + 20, int(y) + 20, labels[i % len(labels)]) 
                    # Draw lines between points
                    if prev_point is not None:
                        painter.drawLine(int(prev_point[0]), int(prev_point[1]), int(x), int(y))
                    prev_point = (x, y)

        # Always draw angle text on every frame
        painter.setPen(QPen(QColor(0, 255, 0), 2)) 
        painter.setFont(QFont("Arial", 50)) 
        angle_text = f"Angle: {angle:.2f}°" if angle is not None else "Angle: N/A"
        painter.drawText(10, 50, angle_text)
        painter.end()

        pixmap = QPixmap.fromImage(q_img)
        scaled_pixmap = pixmap.scaled(label.size(), Qt.KeepAspectRatio)
        label.setPixmap(scaled_pixmap)

    def update_angle_plot(self):
        """Update the live plot of angles using PyQtGraph."""
        # Update camera 0 curve
        self.curve_cam0.setData(self.angles)
        
        # Update camera 2 curve
        self.curve_cam2.setData(self.angles_cam2)
        
        # Update y-axis ranges
        if self.y_min != np.inf and self.y_max != -np.inf:
            self.angle_plot_widget.setYRange(self.y_min - 5, self.y_max + 5)
        
        if self.y_min_cam2 != np.inf and self.y_max_cam2 != -np.inf:
            self.viewbox2.setYRange(self.y_min_cam2 - 5, self.y_max_cam2 + 5)

    def closeEvent(self, event):
        '''Clicked x/close on window - save latencies before closing'''
        logger.info("CameraStreamWidget closeEvent triggered")
        self.visual.stopMe()

        self.comm.put(['stop'])
        logger.info("Closing CameraStreamWidget")
        event.accept()