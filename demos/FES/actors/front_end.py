import sys
import numpy as np
import threading
import queue  # Import the queue module
from PyQt5.QtWidgets import QApplication, QLabel, QWidget, QGridLayout
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QBrush, QFont
import cv2  # Import cv2 for image processing
import time
import traceback
from pathlib import Path
import yaml
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
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
            self.comm = comm  # Link back to Nexus for transmitting signals
            self.q_sig = q_sig
            self.stop_program = False
            self.last_frame_ids = [None for _ in range(self.visual.num_cameras)]
            self.angles = [0]  # Store angles for live plotting
            self.recent_angles = deque(maxlen=5) #TODO make this a parameter
            self.y_max = -np.inf  # Initialize y_max to negative infinity
            self.y_min = np.inf  # Initialize y_min to positive infinity
            self.predictions = None
            

            # Load the configuration file
            source_folder = Path(__file__).resolve().parent.parent
            with open(f'{source_folder}/config.yaml', 'r') as file:
                config = yaml.safe_load(file)

            self.threshold = config['threshold']

            # Set up GUI layout
            self.setWindowTitle('Camera Streams')
            self.setGeometry(100, 100, 1920, 1080)  # Adjust window size to fit aspect ratio

            # Layout to hold the camera labels
            layout = QGridLayout()

            # Create labels to show camera frames
            self.camera_labels = [QLabel(self) for _ in range(self.visual.num_cameras)]
            layout.addWidget(self.camera_labels[0], 0, 0)  # Top-left
            layout.addWidget(self.camera_labels[1], 0, 1)  # Top-right
            layout.addWidget(self.camera_labels[2], 1, 0)  # Bottom-left

            # Add a QLabel for the angle plot in the bottom-right
            self.angle_plot_label = QLabel(self)
            layout.addWidget(self.angle_plot_label, 1, 1)  # Bottom-right

            self.setLayout(layout)

            # Initialize a QTimer to update frames
            self.timer = QTimer()
            self.timer.timeout.connect(self.update_frames)
            self.timer.start(30)  # Adjust the timer interval to match the frame rate [ms]

            logger.info(f'Front End Setup completed')
        except Exception as e:
            logger.error(f'Setup failed due to {e}')
            traceback.format_exc()


    def update_frames(self):
        """Update frames from each camera"""
        for camera_id in range(self.visual.num_cameras):
            frame = None
            predictions = None
            angle = None
            try:
                frame, predictions, angle = self.visual.getLastFrame(camera_id)
                if frame is not None:
                    self.display_frame(frame, predictions, self.camera_labels[camera_id], angle)
                
                # Update the angle plot if an angle is provided
                
                if angle is not None:
                    # #Applying a moving average
                    # self.recent_angles.append(angle)
                    # smooted_angle = np.mean(self.recent_angles)
                    # self.angles.append(smooted_angle)
                    # if predictions[1:3,2].mean() < 0.5:  #Linear interpolation if the likelihood is low
                    #     self.angles.append(self.angles[-1])
                    # else:
                    self.angles.append(angle)
                    if len(self.angles) > 100:  # Limit to the latest 100 angles
                        self.angles.pop(0)
                    if angle > self.y_max:
                        self.y_max = angle
                    if angle < self.y_min:
                        self.y_min = angle
                    self.update_angle_plot()
                    
            except Exception as e:
                blank_frame = np.zeros((self.visual.frame_h, self.visual.frame_w, 3), dtype=np.uint8)
                self.display_frame(blank_frame, None, self.camera_labels[camera_id], 0.0)
                if camera_id == 0:  # Only log errors for camera 0 to reduce spam
                    logger.error(f"Error updating frame for camera {camera_id}: {e}")
                    # pass
                elif camera_id > 0:
                    # Expected error for cameras 1 and 2 - no need to log as error
                    pass

    def display_frame(self, frame, predictions, label, angle):
        """Convert frame to QImage, plot predictions if available, and display it in QLabel."""
        height, width, channel = frame.shape
        bytes_per_line = channel * width
        q_img = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888)
        
        # Only log predictions when they are actually present to reduce log spam
        if predictions is not None:
            logger.debug(f"PREDICTIONS: {predictions}")
            painter = QPainter()
            painter.begin(q_img)
            painter.setBrush(QBrush(QColor(255, 0, 0)))

            # labels = ["DIP", "PIP", "MCP", "Wrist", "Forearm"]
            labels = ["Wrist", "MCP", "End"]

            prev_point = None
            for i, point in enumerate(predictions):
            # for point in predictions:
                x, y, likelihood = point
                if likelihood > 0:
                    painter.setPen(QPen(QColor(255, 0, 0), 2))  # Red color, 2px width
                    painter.drawEllipse(int(x), int(y), 50, 50)
                    painter.setPen(QPen(QColor(255, 255, 255), 2))  # White color for text
                    painter.setFont(QFont("Arial", 50))  # Set font size to 12
                    painter.drawText(int(x) + 20, int(y) + 20, labels[i % len(labels)])  # Add label
                    # Draw lines between points
                    if prev_point is not None:
                        painter.drawLine(int(prev_point[0]), int(prev_point[1]), int(x), int(y))
                    prev_point = (x, y)

            painter.setPen(QPen(QColor(0, 255, 0), 2))  # Green color for text
            angle_text = f"Angle: {angle:.2f}°" if angle is not None else "Angle: N/A"
            painter.drawText(10, 30, angle_text)
            painter.end()

        pixmap = QPixmap.fromImage(q_img)
        scaled_pixmap = pixmap.scaled(label.size(), Qt.KeepAspectRatio)
        label.setPixmap(scaled_pixmap)

    def update_angle_plot(self):
        """Update the live plot of angles."""
        fig, ax = plt.subplots()
        ax.plot(self.angles, color="blue")
        ax.set_title("Live Angle Plot")
        ax.set_xlabel("Frame")
        ax.set_ylabel("Angle (°)")
        ax.set_ylim(self.y_min-5, self.y_max+5)  # Set y-limits based on the angles

        # Convert Matplotlib figure to QImage
        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        width, height = fig.get_size_inches() * fig.get_dpi()
        plot_image = QImage(canvas.buffer_rgba(), int(width), int(height), QImage.Format_ARGB32)

        # Display the plot image in the QLabel
        self.angle_plot_label.setPixmap(QPixmap.fromImage(plot_image))
        plt.close(fig)  # Close the figure to avoid memory leaks