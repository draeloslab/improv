from improv.actor import Actor
import numpy as np
import logging
import pathlib

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class StreamIn(Actor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.C = np.load(pathlib.Path(__file__).parent.parent / 'C.npy')
        self.i = 0

    def setup(self):
        pass

    def runStep(self):
        a = np.array(self.C[self.i % self.C.shape[0]]).reshape(1,-1)
        self.links['q_out'].put(a)
        self.i += 1

    def stop(self):
        pass