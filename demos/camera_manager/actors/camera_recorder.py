from improv.actor import Actor
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class CameraRecorder(Actor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def setup(self):
        """ Prepare for recording video of each acquisition camera """
        # TODO
        logger.info("Camera recording setup")

    def runStep(self):
        # TODO
        pass

    def stop(self):
        # TODO
        pass