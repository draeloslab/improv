import traceback
from queue import Empty

from improv.actor import Actor, Signal
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
import numpy as np


class LiveTraceGUIDataManager(Actor):
    def __init__(self, *args):
        super().__init__(*args)

    def setup(self):
        self.data = []
        self.rng = np.random.default_rng(42)
        self.i = 0

    def run(self):
        pass  # NOTE: Special case here, tied to GUI

    def getData(self):
        """Load data from dim reduction and bubblewrap, returns false on timeout"""
        try:
            t = self.i * 2 * np.pi / 30
            self.data.append(np.array([np.cos(t), np.sin(t)]) * t)
            self.i += 1
        except Empty as e:
            return False
        except Exception as e:
            logger.error(f'Visual: Exception in get data: {e}')
            logger.error(traceback.format_exc())
        return True