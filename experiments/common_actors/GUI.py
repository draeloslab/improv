import numpy as np
import time
import pyqtgraph
from pyqtgraph import EllipseROI, PolyLineROI, ColorMap
from PyQt5 import QtGui,QtCore,QtWidgets
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import QMessageBox, QApplication
from matplotlib.colors import ListedColormap

from improv.actor import Signal
from . import video_photostim

import logging; logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class FrontEnd(QtWidgets.QMainWindow, video_photostim.Ui_MainWindow):

    COLOR = {0: ( 240, 122,  5),
             1: (181, 240,  5),
             2: (5, 240,  5),
             3: (5,  240,  181),
             4: (5,  122, 240),
             5: (64,  5, 240),
             6: ( 240,  5, 240),
             7: ( 240, 5, 64),
             8: ( 240, 240, 240)}

    def __init__(self, visual, comm, parent=None):
        ''' Setup GUI
            Setup and start Nexus controls
        '''
        self.visual = visual #Visual class that provides plots and images
        self.comm = comm #Link back to Nexus for transmitting signals

        self.total_times = []
        self.first = True
        self.prev = 0

        pyqtgraph.setConfigOption('background', QColor(100, 100, 100))
        super(FrontEnd, self).__init__(parent)
        self.setupUi(self)
        pyqtgraph.setConfigOptions(leftButtonPan=False)

        self.customizePlots()

        self.pushButton_3.clicked.connect(_call(self._runProcess)) #Tell Nexus to start
        self.pushButton_3.clicked.connect(_call(self.update)) #Update front-end graphics
        self.pushButton_2.clicked.connect(_call(self._setup))
        self.checkBox.stateChanged.connect(self.update) #Show live front-end updates

        topLeftPoint = QApplication.desktop().availableGeometry().topLeft()
        self.move(topLeftPoint)

    def update(self):
        ''' Update visualization while running
        '''
        t = time.time()
        self.visual.getData()
        if self.draw:
            try:
                self.updateLines()
            except Exception as e:
                logger.info('update lines error {}'.format(e))
                import traceback
                print('---------------------Exception in update lines: ' , traceback.format_exc())
            try:
                self.updateVideo()
            except Exception as e:
                logger.error('Error in FrontEnd update Video:  {}'.format(e))
                import traceback
                print('---------------------Exception in update video: ' , traceback.format_exc())

        if self.checkBox.isChecked():
            self.draw = True
        else:
            self.draw = False    
        self.visual.draw = self.draw
            
        QtCore.QTimer.singleShot(10, self.update)
        
        self.total_times.append([self.visual.frame_num, time.time()-t])

    def customizePlots(self):
        self.checkBox.setChecked(True)
        self.draw = True

        #init line plot
        self.flag = True
        self.flagW = True
        self.flagL = True
        self.last_x = None
        self.last_y = None
        self.weightN = None
        self.last_n = None

        self.c1 = self.grplot.plot(clipToView=True)
        self.c1_stim = [self.grplot.plot(clipToView=True) for _ in range(len(self.COLOR))]
        self.c2 = self.grplot_2.plot()
        grplot = [self.grplot, self.grplot_2]
        for plt in grplot:
            plt.getAxis('bottom').setTickSpacing(major=50, minor=50)
        self.updateLines()
        self.activePlot = 'r'

        #videos
        self.rawplot.ui.histogram.vb.setLimits(yMin=-0.1, yMax=200) #0-255 needed, saturated here for easy viewing

    def _runProcess(self):
        '''Run ImageProcessor in separate thread
        '''
        self.comm.put([Signal.run()])
        logger.info('-------------------------   put run in comm')

    def _setup(self):
        self.comm.put([Signal.setup()])
        self.visual.setup()
    
    def updateVideo(self):
        ''' TODO: Bug on clicking ROI --> trace and report to pyqtgraph
        '''
        raw, color = self.visual.getFrames()
        if raw is not None:
            raw = raw.T         ## necessary for plotting only, visuals same as on microscope computer
            if np.unique(raw).size > 1:
                self.rawplot.setImage(raw) #, autoHistogramRange=False)
                self.rawplot.ui.histogram.vb.setLimits(yMin=80, yMax=200)
        if color is not None:
            color = color.T
            self.rawplot_2.setImage(color)
        # if self.visual.selected_neuron is not None:
        #     self._updateRedCirc(self.visual.selected_neuron[1], self.visual.selected_neuron[2])

    def updateLines(self):
        ''' Helper function to plot the line traces
            of the activity of the selected neurons.
        '''
        penW=pyqtgraph.mkPen(width=2, color='w')
        penR=pyqtgraph.mkPen(width=2, color='r')

        C = None
        Cx = None
        try:
            (Cx, C, Cpop) = self.visual.getCurves()
        except TypeError:
            pass
        except Exception as e:
            logger.error('Output does not likely exist. Error: {}'.format(e))

        if (C is not None and Cx is not None):
            self.c1.setData(Cx, Cpop, pen=penW)

            for i, plot in enumerate(self.c1_stim):
                try:
                    if len(self.visual.allStims[i]) > 0:
                        d = []
                        for s in self.visual.allStims[i]:
                            d.extend(np.arange(s,s+10).tolist())
                        display = np.clip(d, np.min(Cx), np.max(Cx))
                        try:
                            plot.setData(display, [int(np.max(Cpop))+1] * len(display),
                                    symbol='s', symbolSize=6, antialias=False,
                                    pen=None, symbolPen=self.COLOR[i], symbolBrush=self.COLOR[i])
                        except:
                            print(display)
                    if i==8 and len(self.visual.stimTimes) > 0:
                        d = []
                        for s in self.visual.stimTimes:
                            d.extend(np.arange(s,s+10).tolist())
                        display = np.clip(d, np.min(Cx), np.max(Cx))
                        try:
                            plot.setData(display, [int(np.max(Cpop))+1] * len(display),
                                    symbol='s', symbolSize=6, antialias=False,
                                    pen=None, symbolPen=self.COLOR[8], symbolBrush=self.COLOR[8])
                        except:
                            print(display)
                except KeyError:
                    pass

            self.c2.setData(Cx, C, pen=penR)
        
    def mouseClick(self, event):
        '''Clicked on processed image to select neurons
        '''
        event.accept()
        mousePoint = event.pos()
        self.selected = self.visual.selectNeurons(int(mousePoint.x()), int(mousePoint.y()))
        selectedraw = np.zeros(2)
        selectedraw[0] = int(mousePoint.x())
        selectedraw[1] = int(mousePoint.y())
        self._updateRedCirc()

        # if self.last_n is None:
        #     self.last_n = self.visual.selectedNeuron
        # elif self.last_n == self.visual.selectedNeuron:
        #     for i in range(18):
        #         self.rawplot_2.getView().removeItem(self.lines[i])
        #     self.flagW = True

    def _updateRedCirc(self, x, y):
        ''' Circle neuron whose activity is in top (red) graph
            Default is neuron #0 from initialize
            #TODO: add arg instead of self.selected
        '''
        ROIpen1=pyqtgraph.mkPen(width=1, color='r')
        if self.flag:
            self.red_circ = CircleROI(pos = np.array([x, y])-5, size=10, movable=False, pen=ROIpen1)
            self.rawplot_2.getView().addItem(self.red_circ)
            self.red_circ2 = CircleROI(pos = np.array([x, y])-5, size=10, movable=False, pen=ROIpen1)
            self.rawplot.getView().addItem(self.red_circ2)
            self.flag = False
        else:
            self.rawplot_2.getView().removeItem(self.red_circ)
            self.rawplot.getView().removeItem(self.red_circ2)
            self.red_circ = CircleROI(pos = np.array([x, y])-5, size=10, movable=False, pen=ROIpen1)
            self.rawplot_2.getView().addItem(self.red_circ)
            self.red_circ2 = CircleROI(pos = np.array([x, y])-5, size=10, movable=False, pen=ROIpen1)
            self.rawplot.getView().addItem(self.red_circ2)

    def closeEvent(self, event):
        '''Clicked x/close on window
            Add confirmation for closing without saving
        '''
        confirm = QMessageBox.question(self, 'Message', 'Stop the experiment?',
                    QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if confirm == QMessageBox.Yes:
            self.comm.put(['stop'])
            print('Visual got through ', self.visual.frame_num, ' frames')
            np.savetxt('output/timing/visual_frame_time.txt', np.array(self.visual.total_times))
            np.savetxt('output/timing/gui_frame_time.txt', np.array(self.total_times))
            np.savetxt('output/timing/visual_timestamp.txt', np.array(self.visual.timestamp))
            event.accept()
        else: event.ignore()

def _call(fnc, *args, **kwargs):
    ''' Call handler for (external) events
    '''
    def _callback():
        return fnc(*args, **kwargs)
    return _callback

class CircleROI(EllipseROI):
    def __init__(self, pos, size, **args):
        pyqtgraph.ROI.__init__(self, pos, size, **args)
        self.aspectLocked = True

class PolyROI(PolyLineROI):
    def __init__(self, positions, pos, **args):
        closed = True
        print('got positions ', positions)
        pyqtgraph.ROI.__init__(self, positions, closed, pos, **args)

def cmapToColormap(cmap: ListedColormap) -> ColorMap:
    """ Converts matplotlib cmap to pyqtgraph ColorMap. """

    colordata = (np.array(cmap.colors) * 255).astype(np.uint8)
    indices = np.linspace(0., 1., len(colordata))
    return ColorMap(indices, colordata)


if __name__=="__main__":
    import sys
    app = QtGui.QApplication(sys.argv)
    rasp = FrontEnd(None,None)
    rasp.show()
    app.exec_()
