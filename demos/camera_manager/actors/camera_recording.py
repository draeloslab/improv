from improv.actor import Actor
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class CameraReading(Actor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def setup(self):
        """ Prepare for recording video of each acquisition camera """

    def run(self):
        # TODO

    def stop(self):
        # TODO