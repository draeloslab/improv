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
    def __init__(self, visual: LiveTraceGUIDataManager, comm, parent=None, literal_stim_events=False):
        self.visual = visual
        self.comm = comm  # Link back to Nexus for transmitting signals
        self.literal_stim_events = literal_stim_events

        pyqtgraph.setConfigOption("background", QColor(255, 255, 255))

        super(FrontEnd, self).__init__(parent)
        self.setupUi(self)
        pyqtgraph.setConfigOptions(leftButtonPan=True)

        self.plt = self.widget.getPlotItem()
        # Keep x/y units visually equal (matplotlib axis('equal') behavior).
        self.plt.setAspectLocked(lock=True, ratio=1)
        self.tail = pyqtgraph.PlotDataItem(pen=pyqtgraph.mkPen(color='black', width=4))
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
            if max(event.delivery_time, event.difference_interval[0]) >= self.visual.data.t.max():
                continue

            delivery_point = self.visual.data.slice_by_time(event.delivery_time)[:2]
            difference_interval_start_point = self.visual.data.slice_by_time(event.difference_interval[0])[:2]

            start_scatter = pyqtgraph.ScatterPlotItem(
                pos=np.array([delivery_point]),
                size=8,
                brush=pyqtgraph.mkBrush(255, 0, 0),
                pen=pyqtgraph.mkPen(None),
            )
            self.plt.addItem(start_scatter)
            self.stim_plot_items.append(start_scatter)

            prediction = None
            if event.used_prediction is not None:
                prediction = event.used_prediction
            elif len(event.predictions) == 1:
                prediction = list(event.predictions.values())[0]


            if prediction is not None:
                pred_point = prediction
                if self.literal_stim_events:
                    greenline_start = difference_interval_start_point
                else:
                    greenline_start = delivery_point
                pred_line = pyqtgraph.PlotDataItem(
                    x=[greenline_start[0], pred_point[0]],
                    y=[greenline_start[1], pred_point[1]],
                    pen=pyqtgraph.mkPen(color=(0, 180, 0), width=2),
                )
                self.plt.addItem(pred_line)
                self.stim_plot_items.append(pred_line)


                pred_scatter = pyqtgraph.ScatterPlotItem(
                    pos=np.array([pred_point]),
                    size=8,
                    brush=pyqtgraph.mkBrush(0, 180, 0),
                    pen=pyqtgraph.mkPen(None),
                )
                self.plt.addItem(pred_scatter)
                self.stim_plot_items.append(pred_scatter)


                if event.used_prediction is not None:
                    observed_point = event.used_prediction + np.squeeze(event.residual)

                    # Blue line: used_prediction -> used_prediction + residual
                    residual_line = pyqtgraph.PlotDataItem(
                        x=[pred_point[0], observed_point[0]],
                        y=[pred_point[1], observed_point[1]],
                        pen=pyqtgraph.mkPen(color=(0, 100, 255), width=2),
                    )
                    self.plt.addItem(residual_line)
                    self.stim_plot_items.append(residual_line)


                    observed_scatter = pyqtgraph.ScatterPlotItem(
                        pos=np.array([observed_point]),
                        size=8,
                        brush=pyqtgraph.mkBrush(0,100,255),
                        pen=pyqtgraph.mkPen(None),
                    )
                    self.plt.addItem(observed_scatter)
                    self.stim_plot_items.append(observed_scatter)


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