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