import numpy as np
import time
import pyqtgraph
from pyqtgraph import EllipseROI, PolyLineROI, ColorMap
from PyQt5 import QtGui,QtCore,QtWidgets
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import QMessageBox, QApplication
from matplotlib.colors import ListedColormap
from math import atan2, floor
import traceback
from .data_manager import LiveTraceGUIDataManager

from improv.actor import Signal
from experiments.common_actors import video_2p

import logging; logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# class FrontEnd(QtWidgets.QMainWindow, video_2p.Ui_MainWindow):
#
#     COLOR = {0: ( 240, 122,  5),
#              1: (181, 240,  5),
#              2: (5, 240,  5),
#              3: (5,  240,  181),
#              4: (5,  122, 240),
#              5: (64,  5, 240),
#              6: ( 240,  5, 240),
#              7: ( 240, 5, 64),
#              8: ( 240, 240, 240)}
#
#     def __init__(self, visual, comm, parent=None):
#         ''' Setup GUI
#             Setup and start Nexus controls
#         '''
#         self.visual = visual #Visual class that provides plots and images
#         self.comm = comm #Link back to Nexus for transmitting signals
#
#         self.total_times = []
#         self.first = True
#         self.prev = 0
#
#         pyqtgraph.setConfigOption('background', QColor(100, 100, 100))
#         super(FrontEnd, self).__init__(parent)
#         self.setupUi(self)
#         pyqtgraph.setConfigOptions(leftButtonPan=False)
#
#         self.customizePlots()
#
#         self.pushButton_3.clicked.connect(_call(self._runProcess)) #Tell Nexus to start
#         self.pushButton_2.clicked.connect(_call(self._setup))
#
#         topLeftPoint = QApplication.desktop().availableGeometry().topLeft()
#         self.move(topLeftPoint)
#         self.red_circles = None
#
#     def update(self):
#         ''' Update visualization while running
#         '''
#         t = time.time()
#         self.visual.getData()
#         if self.draw:
#             try:
#                 self.updateLines()
#             except Exception as e:
#                 logger.info('update lines error {}'.format(e))
#                 import traceback
#                 print('---------------------Exception in update lines: ' , traceback.format_exc())
#             try:
#                 self.updateVideo()
#             except Exception as e:
#                 logger.error('Error in FrontEnd update Video:  {}'.format(e))
#                 import traceback
#                 print('---------------------Exception in update video: ' , traceback.format_exc())
#
#         if self.checkBox.isChecked():
#             self.draw = True
#         else:
#             self.draw = False
#         self.visual.draw = self.draw
#
#         QtCore.QTimer.singleShot(10, self.update)
#
#         self.total_times.append([self.visual.frame_num, time.time()-t])
#
#     def customizePlots(self):
#         self.checkBox.setChecked(True)
#         self.draw = True
#
#         #init line plot
#         self.flag = True
#         self.flagW = True
#         self.flagL = True
#         self.last_x = None
#         self.last_y = None
#         self.weightN = None
#         self.last_n = None
#
#         self.c1 = self.grplot.plot(clipToView=True)
#         self.c1_stim = [self.grplot.plot(clipToView=True) for _ in range(len(self.COLOR))]
#         self.c2 = self.grplot_2.plot()
#         grplot = [self.grplot, self.grplot_2]
#         for plt in grplot:
#             plt.getAxis('bottom').setTickSpacing(major=50, minor=50)
#         self.updateLines()
#         self.activePlot = 'r'
#
#         #videos
#         self.rawplot.ui.histogram.vb.setLimits(yMin=-0.1, yMax=200) #0-255 needed, saturated here for easy viewing
#
#     def _runProcess(self):
#         '''Run ImageProcessor in separate thread
#         '''
#         self.comm.put([Signal.run()])
#         logger.info('-------------------------   put run in comm')
#
#     def _setup(self):
#         self.comm.put([Signal.setup()])
#         self.visual.setup()
#
#     def updateVideo(self):
#         ''' TODO: Bug on clicking ROI --> trace and report to pyqtgraph
#         '''
#         raw, color = self.visual.getFrames()
#         if raw is not None:
#             raw = raw.T         ## necessary for plotting only, visuals same as on microscope computer
#             if np.unique(raw).size > 1:
#                 self.rawplot.setImage(raw) #, autoHistogramRange=False)
#                 self.rawplot.ui.histogram.vb.setLimits(yMin=80, yMax=200)
#         if color is not None:
#             color = color.T
#             self.rawplot_2.setImage(color)
#
#         if self.visual.last_stim_vector is not None:
#             currently_avialable_coords = len(self.visual.coords)
#
#             hit_neurons = np.nonzero(self.visual.last_stim_vector)[0]
#             xs = [self.visual.coords[i]['CoM'][1] for i in hit_neurons if i < currently_avialable_coords]
#             ys = [self.visual.coords[i]['CoM'][0] for i in hit_neurons if i < currently_avialable_coords]
#             self.draw_red_circles(xs, ys)
#
#     def updateLines(self):
#         ''' Helper function to plot the line traces
#             of the activity of the selected neurons.
#         '''
#         penW=pyqtgraph.mkPen(width=2, color='w')
#         penR=pyqtgraph.mkPen(width=2, color='r')
#
#         C = None
#         Cx = None
#         try:
#             (Cx, C, Cpop) = self.visual.getCurves()
#         except TypeError:
#             pass
#         except Exception as e:
#             logger.error('Output does not likely exist. Error: {}'.format(e))
#
#         if (C is not None and Cx is not None):
#             self.c1.setData(Cx, Cpop, pen=penW)
#             self.c2.setData(Cx, C, pen=penR)
#
#     # def mouseClick(self, event):
#     #     '''Clicked on processed image to select neurons
#     #     '''
#     #     event.accept()
#     #     mousePoint = event.pos()
#     #     self.selected = self.visual.selectNeurons(int(mousePoint.x()), int(mousePoint.y()))
#     #     selectedraw = np.zeros(2)
#     #     selectedraw[0] = int(mousePoint.x())
#     #     selectedraw[1] = int(mousePoint.y())
#     #     self._updateRedCirc()
#     #
#     #     # if self.last_n is None:
#     #     #     self.last_n = self.visual.selectedNeuron
#     #     # elif self.last_n == self.visual.selectedNeuron:
#     #     #     for i in range(18):
#     #     #         self.rawplot_2.getView().removeItem(self.lines[i])
#     #     #     self.flagW = True
#
#     def _draw_red_circle(self, x, y, pen, plots):
#         circles = []
#         for plot in plots:
#             c = CircleROI(pos=np.array([x, y]) - 5, size=10, movable=False, pen=pen)
#             plot.getView().addItem(c)
#             circles.append(c)
#         return circles
#
#     def draw_red_circles(self, xs, ys):
#         plots = [self.rawplot, self.rawplot_2]
#         pen=pyqtgraph.mkPen(width=1, color='r')
#
#         if self.red_circles is not None:
#             for row in self.red_circles:
#                 for plot, c in zip(plots, row):
#                     plot.getView().removeItem(c)
#         self.red_circles = []
#
#         for x, y in zip(xs, ys):
#             self.red_circles.append(self._draw_red_circle(x, y, pen, plots))
#
#     def closeEvent(self, event):
#         '''Clicked x/close on window
#             Add confirmation for closing without saving
#         '''
#         confirm = QMessageBox.question(self, 'Message', 'Stop the experiment?',
#                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
#         if confirm == QMessageBox.Yes:
#             self.comm.put(['stop'])
#             print('Visual got through ', self.visual.frame_num, ' frames')
#             np.savetxt('output/timing/visual_frame_time.txt', np.array(self.visual.total_times))
#             np.savetxt('output/timing/gui_frame_time.txt', np.array(self.total_times))
#             np.savetxt('output/timing/visual_timestamp.txt', np.array(self.visual.timestamp))
#             event.accept()
#         else: event.ignore()
#
# def _call(fnc, *args, **kwargs):
#     ''' Call handler for (external) events
#     '''
#     def _callback():
#         return fnc(*args, **kwargs)
#     return _callback
#
# class CircleROI(EllipseROI):
#     def __init__(self, pos, size, **args):
#         pyqtgraph.ROI.__init__(self, pos, size, **args)
#         self.path = None
#         self.aspectLocked = True
#
#
#
# if __name__=="__main__":
#     import sys
#     app = QtGui.QApplication(sys.argv)
#     rasp = FrontEnd(None,None)
#     rasp.show()
#     app.exec_()







from . import live_trace

class FrontEnd(QtWidgets.QMainWindow, live_trace.Ui_MainWindow):
    def __init__(self, visual: LiveTraceGUIDataManager, comm, parent=None):
        """Setup GUI
        Setup and start Nexus controls
        """
        logger.info("Setup and start Nexus controls")
        self.visual = visual
        self.comm = comm  # Link back to Nexus for transmitting signals

        pyqtgraph.setConfigOption("background", QColor(255, 255, 255))

        super(FrontEnd, self).__init__(parent)
        self.setupUi(self)
        pyqtgraph.setConfigOptions(leftButtonPan=True)

        self.plt = self.widget.getPlotItem()
        self.tail = pyqtgraph.PlotDataItem()
        self.scatter = pyqtgraph.ScatterPlotItem(
            size=1,
            brush=pyqtgraph.mkBrush(177, 177, 177),
            pen=pyqtgraph.mkPen(None),
        )

        # Setup button
        self.pushButton.clicked.connect(_call(self._setup))

        # Run button
        self.pushButton_2.clicked.connect(_call(self._runProcess))

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
        """Function for plotting dim reduced trajectories and bubbles"""
        self.plt.clear()
        # Dim reduced data plotting
        data_red = np.array(self.visual.data).reshape((-1,2))
        self.scatter.setData(pos=data_red)
        self.tail.setData(data_red[-5:])

        self.plt.addItem(self.scatter)
        self.plt.addItem(self.tail)

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
            # print('Visual broke, avg time per frame: ', np.mean(self.visual.total_times, axis=0))
            print("Visual got through ", self.visual.frame_num, " frames")
            # print('GUI avg time ', np.mean(self.total_times))
            event.accept()
        else:
            event.ignore()


def _call(fnc, *args, **kwargs):
    """Call handler for (external) events"""

    def _callback():
        return fnc(*args, **kwargs)

    return _callback
