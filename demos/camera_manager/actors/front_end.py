import sys
import numpy as np
import threading
import queue
import time
from improv.actor import Signal
from PyQt5.QtWidgets import (
    QApplication, QWidget, QGridLayout, QLabel, QPushButton, QVBoxLayout, QSizePolicy
)
from PyQt5.QtCore import QTimer, Qt, QSize
from PyQt5.QtGui import QImage, QPixmap, QIcon, QScreen

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
        
        # Dynamically set window size based on screen resolution
        screen = QScreen.availableGeometry(QApplication.primaryScreen())
        screen_width = screen.width()
        screen_height = screen.height()

        # Adjust window size to fit screen resolution
        self.setWindowTitle('Camera Streams')
        self.setGeometry(0, 0, int(screen_width), int(screen_height))  # 80% of screen resolution

        # Layout to hold the camera labels
        layout = QGridLayout()
        layout.setSpacing(1)  # Add some padding between widgets

        # Calculate label size (half the screen width and height minus padding)
        label_width = int((screen_width) // 2 - 20)
        label_height = int((screen_height) // 2 - 20)

        # Create labels to show camera frames
        self.camera_labels = [QLabel(self) for _ in range(self.visual.num_cameras)]

        for label in self.camera_labels:
            label.setAlignment(Qt.AlignCenter)
            label.setStyleSheet("background-color: white; border: 1px solid black;")
            label.setScaledContents(True) # scale the image to fit the box
            label.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
            label.setFixedSize(label_width, label_height)

        # Add the three camera widgets
        layout.addWidget(self.camera_labels[0], 0, 0)  # Top-left
        layout.addWidget(self.camera_labels[1], 0, 1)  # Top-right
        layout.addWidget(self.camera_labels[2], 1, 0)  # Bottom-left

        # Buttons column layout
        buttons_layout = QVBoxLayout()
        buttons_layout.setSpacing(10)  # Add spacing between buttons

        # Run button with green background and icon
        self.run_button = QPushButton('Start recording', self)
        self.run_button.setIcon(QIcon('/path/to/run_icon.png'))  # Replace with your icon path
        self.run_button.setStyleSheet("background-color: green; color: white;")
        self.run_button.setIconSize(QSize(24, 24))  # Set icon size
        self.run_button.clicked.connect(self.btn_run_action)
        buttons_layout.addWidget(self.run_button)

        # Stop button with red background and icon
        self.stop_button = QPushButton('Stop recording', self)
        self.stop_button.setIcon(QIcon('/path/to/stop_icon.png'))  # Replace with your icon path
        self.stop_button.setStyleSheet("background-color: red; color: white;")
        self.stop_button.setIconSize(QSize(24, 24))  # Set icon size
        self.stop_button.clicked.connect(self.btn_stop_action)
        buttons_layout.addWidget(self.stop_button)

        # Quit button with black background and icon
        self.quit_button = QPushButton('Quit', self)
        self.quit_button.setIcon(QIcon('/path/to/quit_icon.png'))  # Replace with your icon path
        self.quit_button.setStyleSheet("background-color: black; color: white;")
        self.quit_button.setIconSize(QSize(24, 24))  # Set icon size
        self.quit_button.clicked.connect(self.btn_quit_action)
        buttons_layout.addWidget(self.quit_button)

        # Add spacer to push buttons to the left or right if desired
        buttons_layout.addStretch()

        # Add the buttons layout to the grid
        # Add buttons layout to the grid
        layout.addLayout(buttons_layout, 1, 1, alignment=Qt.AlignLeft | Qt.AlignHCenter)  # Second row, second column

        self.setLayout(layout)

        # Initialize a QTimer to update frames
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frames)
        self.timer.start(50)  # Adjust the timer interval to match the frame rate [ms]
        
        # Sending the signal for starting the setup of the cameras
        time.sleep(1) # Wait for the GUI to be ready
        self.comm.put([Signal.setup()])

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

    def btn_run_action(self):
        """Action to execute when the run control button is clicked."""
        self.comm.put([Signal.run()])  # Starting the run of the cameras

    def btn_stop_action(self):
        """Action to execute when the stop control button is clicked."""
        self.comm.put([Signal.stop()])  # Stopping the cameras

    def btn_quit_action(self):
        """Action to execute when the quit control button is clicked."""
        self.comm.put([Signal.quit()])  # Quitting the application