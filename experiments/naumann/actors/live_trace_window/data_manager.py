import traceback
from queue import Empty

from improv.actor import Actor, Signal
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class LiveTraceGUIDataManager(Actor):
    def __init__(self, *args):
        super().__init__(*args)

    def setup(self):
        self.data = None

    def run(self):
        pass  # NOTE: Special case here, tied to GUI

    def getData(self):
        """Load data from dim reduction and bubblewrap, returns false on timeout"""
        try:
            pass
            # bw_res = self.links['bw_in'].get(timeout=0.0005)
            # res = self.q_in.get(timeout=0.0005)
            # self.data = self.client.get(res[1])
            # self.bw_L = self.client.get(bw_res[1][1])
            # self.bw_mu = self.client.get(bw_res[1][2])
            # self.bw_n_obs = self.client.get(bw_res[1][3])
            # self.bw_dead_nodes = self.client.get(bw_res[1][6])
        except Empty as e:
            return False
        except Exception as e:
            logger.error(f'Visual: Exception in get data: {e}')
            logger.error(traceback.format_exc())
        return True