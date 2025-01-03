import sys
import logging
import time
from improv.actor import Signal
from .buffer_conversion_dialog import ProgressDialog
from PyQt5.QtWidgets import (
    QApplication, QWidget, QPushButton, QVBoxLayout, QHBoxLayout, 
    QFileDialog, QMessageBox, QLabel
)
from PyQt5.QtGui import QIcon, QScreen
from PyQt5.QtCore import QTimer, QSize, Qt
from pathlib import Path

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
log_file = "folder_selector.log"
file_handler = logging.FileHandler(log_file)
file_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

class OfflineConversionWidget(QWidget):
    def __init__(self, visual, comm, q_sig):
        super().__init__()

        self.visual = visual
        self.comm = comm  # Link back to Nexus for transmitting signals
        self.q_sig = q_sig

        self.selected_folder = ""

        self.init_ui()

        # Sending the signal for starting the setup of the cameras
        time.sleep(1) # Wait for the GUI to be ready
        self.comm.put([Signal.setup()])

    def init_ui(self):
        current_dir = Path(__file__).parent.parent

        # Initialize ProgressDialog as None
        self.progress_dialog = None

        # Timer to update progress (assuming conversion happens asynchronously)
        self.progress_timer = QTimer()
        self.progress_timer.timeout.connect(self.conversion_buffer_progress)

        # Create the main vertical layout
        layout = QVBoxLayout()

        # Label to display the selected folder path
        self.folder_path_label = QLabel("No folder selected.", self)
        self.folder_path_label.setAlignment(Qt.AlignCenter)
        self.folder_path_label.setStyleSheet("font-size: 14px;")
        layout.addWidget(self.folder_path_label)

        # Create a horizontal layout for the buttons
        buttons_layout = QHBoxLayout()
        buttons_layout.setAlignment(Qt.AlignCenter)  # Center the buttons horizontally
        buttons_layout.setSpacing(20)  # Space between buttons

        # Select Folder button
        self.select_folder_btn = QPushButton('Select Folder', self)
        self.select_folder_btn.setIcon(QIcon(f'{current_dir}/assets/icons/select_folder.png'))  # Replace with your icon path
        self.select_folder_btn.setIconSize(QSize(24, 24))  # Set icon size
        self.select_folder_btn.setStyleSheet(self.__get_button_style("#3498db", "white"))
        self.select_folder_btn.clicked.connect(self.select_folder)
        buttons_layout.addWidget(self.select_folder_btn)

        # Conversion button
        self.start_conversion_btn = QPushButton(' Start conversion', self)
        self.start_conversion_btn.setIcon(QIcon(f'{current_dir}/assets/icons/conversion.png'))  # Replace with your icon path
        self.start_conversion_btn.setIconSize(QSize(24, 24))  # Set icon size
        self.start_conversion_btn.setStyleSheet(self.__get_button_style("#197292", "white"))
        self.start_conversion_btn.clicked.connect(self.btn_start_conversion)
        self.start_conversion_btn.hide() # hide the start conversion button

        buttons_layout.addWidget(self.start_conversion_btn)

        # Quit button
        self.quit_btn = QPushButton(' Quit', self)
        self.quit_btn.setIcon(QIcon(f'{current_dir}/assets/icons/quit.png'))  # Replace with your icon path
        self.quit_btn.setIconSize(QSize(24, 24))  # Set icon size
        self.quit_btn.setStyleSheet(self.__get_button_style("#e4e9ed", "black"))
        self.quit_btn.clicked.connect(self.btn_quit_action)
        buttons_layout.addWidget(self.quit_btn)

        # Add the horizontal buttons layout to the main vertical layout
        layout.addLayout(buttons_layout)

        # Set the main layout for the widget
        self.setLayout(layout)
        self.setWindowTitle('Offline video converter')
        self.setGeometry(100, 100, 600, 200)  # Set window size and position as needed

    def select_folder(self):
        default_folder = self.visual.get_default_video_path() 

        options = QFileDialog.Options()
        options |= QFileDialog.ShowDirsOnly

        folder_path = QFileDialog.getExistingDirectory(
            self, 
            "Select Folder", 
            default_folder,  # Set the initial directory here
            options=options
        )

        if folder_path:
            self.selected_folder = folder_path
            
            logger.info(f"Selected folder: {self.selected_folder}")
            self.folder_path_label.setText(f"Selected folder:\n{self.selected_folder}")

            # Show the start conversion button
            self.start_conversion_btn.show()

    def btn_start_conversion(self):
        self.visual.start_buffer_conversion(self.selected_folder)
        self.total_buffers = self.visual.get_number_buffer_conversion()
        self.start_conversion_btn.setEnabled(False)

        logger.info(f"Total buffers received: {self.total_buffers}")

        if sum(self.total_buffers) == 0:
            QMessageBox.warning(self, "No buffer files found", "No buffer files to convert found in the selected folder.")
            self.__quit_application()
        else:
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

    def btn_quit_action(self):
        """Action to execute when the quit control button is clicked."""
        self.__quit_application()

    def __quit_application(self):
        """Quit the application."""
        self.comm.put([Signal.stop()])
        time.sleep(1)
        self.comm.put([Signal.quit()])
        QApplication.instance().quit()

    def __get_button_style(self, back_color, font_color):
        """Returns the CSS style for the buttons."""
        return f"""
            QPushButton {{
                background-color: {back_color};
                color: {font_color};
                padding: 10px 20px;  /* top/bottom, left/right padding */
                border: none;
                border-radius: 5px;
                font-size: 16px;
            }}
            QPushButton::icon {{
                margin-right: 10px; /* Space between icon and text */
            }}
            QPushButton:hover {{
                opacity: 0.8;
            }}
        """