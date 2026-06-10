import numpy as np
import pyqtgraph
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import QMessageBox

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

        self.stim_plot_items = []

        # Setup button
        self.pushButton.clicked.connect(self._setup)

        # Run button
        self.pushButton_2.clicked.connect(self._runProcess)

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.update)
        self.timer.start(50)

    def update(self):
        """Check if get data is successful, call plotting function and update GUI"""
        redraw_trace, redraw_stim = self.visual.getData()
        if redraw_trace:
            self.redraw_trace()
        if redraw_stim:
            self.redraw_stim()

    def redraw_trace(self):
        self.scatter.setData(pos=self.visual.data[:,:2])
        self.tail.setData(self.visual.data[-10:, :2])

    def redraw_stim(self):
        for item in self.stim_plot_items:
            self.plt.removeItem(item)
        self.stim_plot_items = []

        for event in self.visual.stim_events:
            if event.delivery_time >= self.visual.data.t.max():
                continue

            start_point = self.visual.data.slice_by_time(event.delivery_time)[:2]

            start_scatter = pyqtgraph.ScatterPlotItem(
                pos=np.array([start_point]),
                size=8,
                brush=pyqtgraph.mkBrush(255, 0, 0),
                pen=pyqtgraph.mkPen(None),
            )
            self.plt.addItem(start_scatter)
            self.stim_plot_items.append(start_scatter)

            # draw a red dot at start_point
            if not event.fufilled:
                pass # pass for now
            else:
                # draw a green line from start_point to event.used_prediction
                pred_point = event.used_prediction
                observed_point = event.used_prediction + np.squeeze(event.residual)


                pred_line = pyqtgraph.PlotDataItem(
                    x=[start_point[0], pred_point[0]],
                    y=[start_point[1], pred_point[1]],
                    pen=pyqtgraph.mkPen(color=(0, 180, 0), width=2),
                )
                self.plt.addItem(pred_line)
                self.stim_plot_items.append(pred_line)

                # Blue line: used_prediction -> used_prediction + residual
                residual_line = pyqtgraph.PlotDataItem(
                    x=[pred_point[0], observed_point[0]],
                    y=[pred_point[1], observed_point[1]],
                    pen=pyqtgraph.mkPen(color=(0, 100, 255), width=2),
                )
                self.plt.addItem(residual_line)
                self.stim_plot_items.append(residual_line)



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