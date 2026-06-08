from queue import Empty

from improv.actor import Actor
import logging

from adaptive_latents.stim_regressor import StimEvent
from adaptive_latents.timed_data_source import ArrayWithTime

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class LiveTraceGUIDataManager(Actor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def setup(self):
        self.data_list = []
        self.stim_events: list[StimEvent] = []
        self.data: ArrayWithTime = ArrayWithTime([], [])

    def run(self):
        pass  # NOTE: Special case here, tied to GUI

    def getData(self):
        if self.client is None:
            self._getStoreInterface()

        new_trace_data = False
        new_stim_data = False

        try:
            id = self.links['latents_in'].get(timeout=.001)
        except Empty:
            pass
        else:
            new_trace_data = True
            data = self.client.get(id)
            self.data_list.append(data)
            self.data = ArrayWithTime.from_list(self.data_list, squeeze_type='to_2d') # TODO: choose about preallocation/efficiency here

        try:
            id = self.links['stim_events_in'].get(timeout=.001)
        except Empty:
            pass
        else:
            new_stim_data = True
            self.stim_events = self.client.get(id) # TODO: make this more efficient, don't send the whole list every time

        redraw_trace = new_trace_data and len(self.data_list) > 1
        redraw_stim = new_stim_data and len(self.stim_events)
        return redraw_trace, redraw_stim