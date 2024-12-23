import sys
import numpy as np
import threading
import queue
import time
from pathlib import Path
from improv.actor import Signal
from PyQt5.QtWidgets import (
    QApplication, QWidget, QGridLayout, QLabel, QPushButton, QHBoxLayout, QVBoxLayout, 
    QSizePolicy, QProgressBar, QDialog, QMessageBox
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

class ProgressDialog(QDialog):
    """
    A dialog window that displays progress bars for each camera's buffer conversion.
    """
    def __init__(self, total_buffers, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Buffer Conversion Progress")
        self.setMinimumWidth(400)
        
        self.layout = QVBoxLayout()
        self.progress_bars = []
        self.labels = []
        
        # Create a progress bar and label for each camera
        for i, total in enumerate(total_buffers):
            camera_label = QLabel(f"Camera {i+1}: 0/{total}")
            progress_bar = QProgressBar()
            progress_bar.setMaximum(total)
            progress_bar.setValue(0)
            
            self.labels.append(camera_label)
            self.progress_bars.append(progress_bar)
            
            # Layout for each camera's progress
            camera_layout = QHBoxLayout()
            camera_layout.addWidget(camera_label)
            camera_layout.addWidget(progress_bar)
            
            self.layout.addLayout(camera_layout)
        
        self.setLayout(self.layout)
    
    def update_progress(self, buffer_progress):
        """
        Update the progress bars based on the number of buffers completed.
        
        Args:
            buffer_progress (list): List containing the number of buffers completed for each camera.
        """
        for i, num_done in enumerate(buffer_progress):
            if i < len(self.progress_bars):
                self.progress_bars[i].setValue(num_done)
                total = self.progress_bars[i].maximum()
                self.labels[i].setText(f"Camera {i+1}: {num_done}/{total}")


class CameraStreamWidget(QWidget):
    """PyQt Widget for displaying multiple camera streams."""

    def __init__(self, visual, comm, q_sig):
        super().__init__()

        self.visual = visual
        self.comm = comm  # Link back to Nexus for transmitting signals
        self.q_sig = q_sig
        self.stop_program = False
        self.last_frame_ids = [None for _ in range(self.visual.num_cameras)]

        # Initialize ProgressDialog as None
        self.progress_dialog = None

        # Timer to update progress (assuming conversion happens asynchronously)
        self.progress_timer = QTimer()
        self.progress_timer.timeout.connect(self.conversion_buffer_progress)

        # get current folder path
        current_dir = Path(__file__).parent.parent
        
        # Dynamically set window size based on screen resolution
        screen = QScreen.availableGeometry(QApplication.primaryScreen())
        screen_width = screen.width()
        screen_height = screen.height()

        # Adjust window size to fit screen resolution
        self.setWindowTitle('Camera Streams')
        self.setGeometry(0, 0, int(screen_width), int(screen_height))

        # Layout to hold the camera labels
        layout = QGridLayout()
        layout.setSpacing(10)  # Add some padding between widgets

        # Calculate label size (half the screen width and height minus padding)
        label_width = int((screen_width) // 2 - 80)
        label_height = int((screen_height) // 2 - 60)

        # Create labels to show camera frames
        self.camera_labels = [QLabel(self) for _ in range(self.visual.num_cameras)]

        for label in self.camera_labels:
            label.setAlignment(Qt.AlignCenter)
            label.setStyleSheet("background-color: white; border: 1px solid black;")
            label.setScaledContents(True) # scale the image to fit the box
            label.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
            label.setFixedSize(label_width, label_height)

        # Buttons column layout
        buttons_layout = QHBoxLayout()
        buttons_layout.setSpacing(5)  # Add spacing between buttons

        # Run button
        self.run_btn = QPushButton('Start recording', self)
        self.run_btn.setIcon(QIcon(f'{current_dir}/assets/icons/start_recording.png'))  # Replace with your icon path
        self.run_btn.setStyleSheet(self.__get_button_style("#1e824c", "white"))
        self.run_btn.setIconSize(QSize(24, 24))  # Set icon size
        self.run_btn.clicked.connect(self.btn_run_action)

        # Stop button 
        self.stop_btn = QPushButton('Stop recording', self)
        self.stop_btn.setIcon(QIcon(f'{current_dir}/assets/icons/stop_recording.png'))  # Replace with your icon path
        self.stop_btn.setStyleSheet(self.__get_button_style("#d91e18", "white"))
        self.stop_btn.setIconSize(QSize(24, 24))  # Set icon size
        self.stop_btn.clicked.connect(self.btn_stop_action)
        self.stop_btn.setEnabled(False)  # Disable the stop button initially

        # Quit button
        self.quit_button = QPushButton('Quit', self)
        self.quit_button.setIcon(QIcon(f'{current_dir}/assets/icons/quit.png'))  # Replace with your icon path
        self.quit_button.setIconSize(QSize(24, 24))  # Set icon size
        self.quit_button.setStyleSheet(self.__get_button_style("#e4e9ed", "black"))
        self.quit_button.clicked.connect(self.btn_quit_action)

        ## adding the elements to the layout
        buttons_layout.addWidget(self.run_btn)
        buttons_layout.addWidget(self.stop_btn)
        buttons_layout.addWidget(self.quit_button)
        
        # Add the buttons layout to the grid layout in the first row, spanning two columns, centered
        layout.addLayout(buttons_layout, 0, 0, 1, 2, alignment=Qt.AlignCenter)

        # Add the three camera widgets
        layout.addWidget(self.camera_labels[0], 1, 0)  # Top-left
        layout.addWidget(self.camera_labels[1], 1, 1)  # Top-right
        layout.addWidget(self.camera_labels[2], 2, 0)  # Bottom-left

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
                frame = self.visual.get_last_frame(camera_id)
        
                self.display_frame(frame, self.camera_labels[camera_id])
            except Exception as e:
                logger.error(f"Error: {e}")

    def display_frame(self, frame, label):
        """Convert frame to QImage and display it in QLabel."""
        height, width, channel = frame.shape
        bytes_per_line = channel * width
        q_img = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)
        scaled_pixmap = pixmap.scaled(label.size(), Qt.KeepAspectRatio)  # Keep aspect ratio
        label.setPixmap(scaled_pixmap)

    def btn_run_action(self):
        """Action to execute when the run control button is clicked."""
        self.comm.put([Signal.run()])  # Starting the run of the cameras
        self.run_btn.setEnabled(False) # Disable the run button - only one run is supported
        self.stop_btn.setEnabled(True)  # Enable the stop button

    def start_conversion(self):
        """
        Initiates the buffer conversion process and displays the progress dialog.
        """
        logger.info("Handling buffer conversion click")
        self.visual.start_buffer_conversion()

        self.total_buffers = self.visual.get_number_buffer_conversion()

        # Initialize and show the ProgressDialog
        self.progress_dialog = ProgressDialog(self.total_buffers)
        self.progress_dialog.show()
        
        # Start a timer to periodically update the progress dialog
        self.progress_timer.start(500)  # Update every 1 second

    def conversion_buffer_progress(self):
        """
        Retrieves the current buffer conversion progress and updates the progress dialog.
        """
        buffer_progress = self.visual.check_buffer_conversion_progress()
        
        if self.progress_dialog:
            self.progress_dialog.update_progress(buffer_progress)
        
        # Check if all conversions are complete
        all_complete = all(
            done >= total for done, total in zip(buffer_progress, self.total_buffers)
        )
        
        if all_complete:
            self.progress_timer.stop()
            if self.progress_dialog:
                self.progress_dialog.close()
                self.progress_dialog = None

            # show a completion message
            self.show_completion_message()
    
    def show_completion_message(self):
        """ Displays a message indicating that the buffer conversion is complete. """
        QMessageBox.information(self, "Conversion Complete", "All buffers have been successfully converted.")

        self.__quit_application()

    def btn_stop_action(self):
        """Action to execute when the Stop control button is clicked."""
        self.comm.put([Signal.stop()])  # Stopping the run of the cameras
        self.stop_btn.setEnabled(False)  # Disable the stop button

        # Create and configure the message box
        msg_box = QMessageBox(self)
        msg_box.setIcon(QMessageBox.Question)
        msg_box.setWindowTitle("Convert Video")
        msg_box.setText("Do you want to convert the video now?")
        msg_box.setInformativeText("Video conversion can take some time for long recordings.")
        msg_box.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        msg_box.setDefaultButton(QMessageBox.No)

        # Execute the message box and capture the user's response
        reply = msg_box.exec_()

        if reply == QMessageBox.Yes:
            self.start_conversion()
        else:
            self.__quit_application()
        
    def btn_conversion_action(self):
        """Action to execute when the Start Conversion control button is clicked."""
        self.start_conversion() # Start the conversion process

    def btn_quit_action(self):
        """Action to execute when the quit control button is clicked."""
        self.__quit_application()

    def __quit_application(self):
        """Quit the application."""
        self.comm.put([Signal.quit()])

    def __get_button_style(self, back_color, font_color):
        """ Returns the CSS style for the buttons """
        return f"""
            QPushButton {{
                background-color: {back_color};
                color: {font_color};
                padding: 5px 5px;  /* top/bottom, left/right padding */
                text-align: left;   /* Align text to the left */
            }}
            QPushButton::icon {{
                margin-right: 10px; /* Space between icon and text */
            }}
        """