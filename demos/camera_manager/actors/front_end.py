import sys
import numpy as np
import threading
import queue
from PyQt5.QtWidgets import QApplication, QLabel, QWidget, QGridLayout
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QImage, QPixmap

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
                frame = self.visual.getLastFrame(camera_id)
        
                self.display_frame(frame, self.camera_labels[camera_id])
            except Exception as e:
                logger.error(f"Error: {e}")

    def display_frame(self, frame, label):
        """Convert frame to QImage and display it in QLabel."""
        # Convert frame to RGB format
        # rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width, channel = frame.shape
        bytes_per_line = channel * width
        q_img = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)
        scaled_pixmap = pixmap.scaled(label.size(), Qt.KeepAspectRatio)  # Keep aspect ratio
        label.setPixmap(scaled_pixmap)