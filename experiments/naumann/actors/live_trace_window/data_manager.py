from queue import Empty

from improv.actor import Actor
import logging

from adaptive_latents.stim_regressor import StimEvent

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class LiveTraceGUIDataManager(Actor):
    def __init__(self, *args):
        super().__init__(*args)

    def setup(self):
        self.data = []
        self.stim_events: list[StimEvent] = []

    def run(self):
        pass  # NOTE: Special case here, tied to GUI

    def getData(self):
        new_data = False

        try:
            id = self.links['latents_in'].get(timeout=.001)
        except Empty:
            pass
        else:
            new_data = True
            data = self.client.get(id)
            self.data.append(data)

        try:
            id = self.links['stim_events_in'].get(timeout=.001)
        except Empty:
            pass
        else:
            new_data = True
            self.stim_events = self.client.get(id) # TODO: make this more efficient, don't send the whole list every time
            logger.info(self.stim_events)

        return new_data and len(self.data) > 1