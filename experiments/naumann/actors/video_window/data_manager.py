import time
import numpy as np
from scipy.spatial.distance import cdist
from queue import Empty
from collections import deque
from PyQt5 import QtWidgets

from improv.actor import Actor, Signal
from improv.store import ObjectNotFoundError

import logging


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


class VideoGUIDataManager(Actor):
    ''' Class for displaying data from caiman processor
    '''

    def __init__(self, *args, stimuli=None, labels=None, **kwargs):
        super().__init__(*args)

        self.com1 = np.zeros(2)
        self.selectedNeuron = 0
        self.selectedTune = None
        self.frame_num = 0

        self.red_chan = None
        self.stimTimes = []

    def setup(self):
        self.Cx = None
        self.C = None
        self.tune = None
        self.raw = None
        self.color = None
        self.coords = None
        self.selected_neuron = None
        self.draw = True

        self.total_times = []
        self.timestamp = []

        self.last_stim_vector = None

        self.window = 150

        try:
            self.red_chan = np.load(self.red_chan_image, allow_pickle=True)
        except:
            pass

    def run(self):
        pass  # NOTE: Special case here, tied to GUI

    def getData(self):
        t = time.time()
        ids = None
        try:
            id = self.links['raw_frame_queue'].get(timeout=0.0001)
            self.raw_frame_number = list(id[0].keys())[0]
            # self.raw = self.client.getID(id[0][self.raw_frame_number])
            self.raw = self.client.get(id[0][self.raw_frame_number])
        except Empty as e:
            pass
        except Exception as e:
            logger.error('Visual: Exception in get data: {}'.format(e))

        try:
            id = self.links['stim_vector_in'].get(timeout=0.0001)
            self.last_stim_vector = self.client.get(id)
        except Empty as e:
            pass

        try:  # NOTE: try removing try block
            ids = self.q_in.get(timeout=0.0001)
            if ids is not None and ids[0] == 1:
                print('visual: missing frame')
                self.frame_num += 1
                self.total_times.append([time.time(), time.time() - t])
                raise Empty
            self.frame_num = ids[-1]
            if self.draw:
                # (self.Cx, self.C, self.Cpop, self.tune, self.color, self.coords, self.allStims, self.tc_list) = self.client.get(ids[:-1])
                self.Cx = self.client.get(ids[0])
                self.C = self.client.get(ids[1])
                self.Cpop = self.client.get(ids[2])
                self.tune = self.client.get(ids[3])
                self.color = self.client.get(ids[4])
                self.coords = self.client.get(ids[5])
                self.allStims = self.client.get(ids[6])
                self.tc_list = self.client.get(ids[7])
                self.total_times.append([time.time(), time.time() - t])
            self.timestamp.append([time.time(), self.frame_num])
        except Empty as e:
            pass
        except ObjectNotFoundError as e:
            logger.error('Object not found, continuing anyway...')
        except Exception as e:
            logger.error('Visual: Exception in get data: {}'.format(e))
        # logger.info('visual_viz_stim time: {}'.format(self.total_times))
        try:
            # stim_in = self.links['optim_in'].get(timeout=0.0001)
            # self.selected_neuron = stim_in
            self.selectedNeuron = 0
            # self.stimTimes.append(int(stim_in[3]))
        except Empty as e:
            pass
        except Exception as e:
            logger.error('Visual: Exception in get stim for visual: {}'.format(e))

    def getCurves(self):
        ''' Return the fluorescence traces and calculated tuning curves
            for the selected neuron as well as the population average
            Cx is the time (overall or window) as x axis
            C is indexed for selected neuron and Cpop is the population avg
            tune is a similar list to C
        '''
        if self.tune is not None:
            self.selectedTune = self.tune[0][self.selectedNeuron]
            self.tuned = [self.selectedTune, self.tune[1]]
        else:
            self.tuned = None

        if self.frame_num > self.window:
            self.C = self.C[:, -len(self.Cx):]
            self.Cpop = self.Cpop[-len(self.Cx):]

        return self.Cx, self.C[self.selectedNeuron, :], self.Cpop

    def getFrames(self):
        ''' Return the raw and colored frames for display
        '''
        return self.raw, self.color
