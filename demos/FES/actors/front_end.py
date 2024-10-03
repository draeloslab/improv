import sys
import numpy as np
import threading
import queue  # Import the queue module
from PyQt5.QtWidgets import QApplication, QLabel, QWidget, QGridLayout
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QBrush
import cv2  # Import cv2 for image processing

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

class CameraStreamWidget(QWidget):
    """PyQt Widget for displaying multiple camera streams."""

    def __init__(self, visual, comm, q_sig):
        super().__init__()

        self.visual = visual
        self.comm = comm  # Link back to Nexus for transmitting signals
        self.q_sig = q_sig
        self.stop_program = False
        self.last_frame_ids = [None for _ in range(self.visual.num_cameras)]
        
        # Set up GUI layout
        self.setWindowTitle('Camera Streams')
        self.setGeometry(100, 100, 1920, 1080)  # Adjust window size to fit aspect ratio

        # Layout to hold the camera labels
        layout = QGridLayout()

        # Create labels to show camera frames
        self.camera_labels = [QLabel(self) for _ in range(self.visual.num_cameras)]
        for i, label in enumerate(self.camera_labels):
            label.setAlignment(Qt.AlignCenter)
            label.setScaledContents(False)  # Maintain aspect ratio
            if i < 2:
                layout.addWidget(label, 0, i)  # First row, two columns
            else:
                layout.addWidget(label, 1, 0, 1, 2)  # Second row, one column spanning two cells

        self.setLayout(layout)

        # Initialize a QTimer to update frames
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frames)
        self.timer.start(100)  # Adjust the timer interval to match the frame rate [ms]

    def update_frames(self):
        """Update frames from each camera"""
        for camera_id in range(self.visual.num_cameras):
            try:
                frame, predictions = self.visual.getLastFrame(camera_id)
                # logger.info(f"Received frame for camera {camera_id}")

                self.display_frame(frame, predictions, self.camera_labels[camera_id])
            except Exception as e:
                logger.error(f"Error updating frame for camera {camera_id}: {e}")
                # Display a blank frame if there's an error
                blank_frame = np.zeros((self.visual.frame_h, self.visual.frame_w, 3), dtype=np.uint8)
                self.display_frame(blank_frame, None, self.camera_labels[camera_id])

    def display_frame(self, frame, predictions, label):
        """Convert frame to QImage, plot predictions if available, and display it in QLabel."""
        # Convert frame to RGB format
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        height, width, channel = rgb_frame.shape
        bytes_per_line = channel * width
        q_img = QImage(rgb_frame.data, width, height, bytes_per_line, QImage.Format_RGB888)
        
        if predictions is not None:
            # logger.info(f"Prediction recieved: {predictions}")
            painter = QPainter()
            painter.begin(q_img)
            painter.setPen(QPen(QColor(255, 0, 0), 2))  # Red color, 2px width
            painter.setBrush(QBrush(QColor(255, 0, 0)))

            
            for point in predictions:
                x, y, likelihood = point
                if likelihood > 0:  # Only plot points with high likelihood
                        painter.drawEllipse(int(x), int(y), 40, 40)
            
            painter.end()

        pixmap = QPixmap.fromImage(q_img)
        scaled_pixmap = pixmap.scaled(label.size(), Qt.KeepAspectRatio)  # Keep aspect ratio
        label.setPixmap(scaled_pixmap)