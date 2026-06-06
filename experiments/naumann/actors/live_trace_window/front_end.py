import numpy as np
import pyqtgraph
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import QMessageBox
import traceback
from .data_manager import LiveTraceGUIDataManager

from improv.actor import Signal

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


from . import live_trace


class FrontEnd(QtWidgets.QMainWindow, live_trace.Ui_MainWindow):
    def __init__(self, visual: LiveTraceGUIDataManager, comm, parent=None):
        self.visual = visual
        self.comm = comm  # Link back to Nexus for transmitting signals

        pyqtgraph.setConfigOption("background", QColor(255, 255, 255))

        super(FrontEnd, self).__init__(parent)
        self.setupUi(self)
        pyqtgraph.setConfigOptions(leftButtonPan=True)

        self.plt = self.widget.getPlotItem()
        # Keep x/y units visually equal (matplotlib axis('equal') behavior).
        self.plt.setAspectLocked(lock=True, ratio=1)
        self.tail = pyqtgraph.PlotDataItem(pen=pyqtgraph.mkPen(color='black', width=2))
        self.scatter = pyqtgraph.ScatterPlotItem(
            size=4,
            brush=pyqtgraph.mkBrush(177, 177, 177),
            pen=pyqtgraph.mkPen(None),
        )
        self.plt.addItem(self.scatter)
        self.plt.addItem(self.tail)

        # Setup button
        self.pushButton.clicked.connect(self._setup)

        # Run button
        self.pushButton_2.clicked.connect(self._runProcess)

    def update(self):
        """Check if get data is successful, call plotting function and update GUI"""
        try:
            if self.visual.getData():
                self.plot()
        except Exception as e:
            logger.error('Front End Exception: {}'.format(e))
            logger.error(traceback.format_exc())
        QtCore.QTimer.singleShot(10, self.update)

    def plot(self):
        data_red = np.squeeze(self.visual.data)
        self.scatter.setData(pos=data_red[:,:2])
        self.tail.setData(data_red[-5:, :2])

    def _runProcess(self):
        logger.info("-------------------------   put run in comm")
        self.comm.put([Signal.run()])

    def _setup(self):
        logger.info("-------------------------   put setup in comm")
        self.comm.put([Signal.setup()])
        self.visual.setup()

    def closeEvent(self, event):
        """Clicked x/close on window
        Add confirmation for closing without saving
        """
        confirm = QMessageBox.question(
            self,
            "Message",
            "Quit without saving?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if confirm == QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()