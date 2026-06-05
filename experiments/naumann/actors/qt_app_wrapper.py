import logging
from PyQt5 import QtWidgets

from improv.actor import Actor, Signal
from .video_window.front_end import FrontEnd as VideoWindowFrontEnd
from .live_trace_window.front_end import FrontEnd as LiveTraceWindowFrontEnd

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .video_window.data_manager import VideoGUIDataManager
    from .live_trace_window.data_manager import LiveTraceGUIDataManager

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class GUI_QTAppWrapper(Actor):
    ''' Class used to run a GUI + Visual as a single Actor 
    '''

    def run(self):
        logger.info('Loading FrontEnd')
        self.app = QtWidgets.QApplication([])
        self.video_window = VideoWindowFrontEnd(self.visual[0], self.q_comm)
        self.live_trace_window = LiveTraceWindowFrontEnd(self.visual[1], self.q_comm)
        self.video_window.show()
        self.live_trace_window.show()
        logger.info('GUI ready')
        self.q_comm.put([Signal.ready()])
        for v in self.visual:
            v.q_comm.put([Signal.ready()])
        self.video_window.update()
        self.live_trace_window.update()
        self.app.exec_()
        logger.info('Done running GUI')

    def setup(self, visual=None):
        logger.info('Running setup for ' + self.name)
        self.visual: tuple[VideoGUIDataManager, LiveTraceGUIDataManager] = visual
        for v in self.visual:
            v.setup()
