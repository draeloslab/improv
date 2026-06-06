import traceback
from queue import Empty

from improv.actor import Actor
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
import numpy as np


class LiveTraceGUIDataManager(Actor):
    def __init__(self, *args):
        super().__init__(*args)

    def setup(self):
        self.data = []

    def run(self):
        pass  # NOTE: Special case here, tied to GUI

    def getData(self):
        try:
            id = self.links['latents_in'].get(timeout=.001)
        except Empty:
            return False
        else:
            data = self.client.get(id)
            self.data.append(data)
        return len(self.data) > 1