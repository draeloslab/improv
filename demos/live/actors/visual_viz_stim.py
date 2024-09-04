import time
import numpy as np
from scipy.spatial.distance import cdist
from queue import Empty
from collections import deque
from PyQt5 import QtWidgets

from improv.actor import Actor, Signal
from improv.store import ObjectNotFoundError
from .GUI import FrontEnd

import logging; logger = logging.getLogger(__name__)

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(message)s',
                    handlers=[logging.FileHandler("example1.log"),
                              logging.StreamHandler()])

class DisplayVisual(Actor):
    ''' Class used to run a GUI + Visual as a single Actor 
    '''
    def run(self):
        logger.info('Loading FrontEnd')
        self.app = QtWidgets.QApplication([])
        self.rasp = FrontEnd(self.visual, self.q_comm)
        self.rasp.show()
        logger.info('GUI ready')
        self.q_comm.put([Signal.ready()])
        self.visual.q_comm.put([Signal.ready()])
        self.app.exec_()
        logger.info('Done running GUI')

    def setup(self, visual=None):
        logger.info('Running setup for '+self.name)
        self.visual = visual
        self.visual.setup()

class CaimanVisualStim(Actor):
    ''' Class for displaying data from caiman processor
    '''
    def __init__(self, *args, stimuli=None, labels=None,  **kwargs):
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

        self.window=150

        try:
            self.red_chan = np.load(self.red_chan_image, allow_pickle=True)
        except:
            pass

    def run(self):
        pass #NOTE: Special case here, tied to GUI

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
            ids = self.q_in.get(timeout=0.0001)
            if ids is not None and ids[0]==1:
                print('visual: missing frame')
                self.frame_num += 1
                self.total_times.append([time.time(), time.time()-t])
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
                self.total_times.append([time.time(), time.time()-t])
            self.timestamp.append([time.time(), self.frame_num])
        except Empty as e:
            pass
        except ObjectNotFoundError as e:
            logger.error('Object not found, continuing anyway...')
        except Exception as e:
            logger.error('Visual: Exception in get data: {}'.format(e))
        logger.info('visual_viz_stim time: {}'.format(self.total_times))
        try:
            stim_in = self.links['optim_in'].get(timeout=0.0001)
            self.selected_neuron = stim_in
            self.selectedNeuron = int(stim_in[0])
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
        
        return self.Cx, self.C[self.selectedNeuron,:], self.Cpop

    def getFrames(self):
        ''' Return the raw and colored frames for display
        '''
        return self.raw, self.color

    # def selectNeurons(self, x, y):
    #     ''' x and y are coordinates
    #         identifies which neuron is closest to this point
    #         and updates plotEstimates to use that neuron
    #     '''
    #     neurons = [o['neuron_id']-1 for o in self.coords]
    #     com = np.array([o['CoM'] for o in self.coords])
    #     dist = cdist(com, [np.array([self.raw.shape[0]-x, self.raw.shape[1]-y])])
    #     if np.min(dist) < 50:
    #         selected = neurons[np.argmin(dist)]
    #         self.selectedNeuron = selected
    #         print('ID for selected neuron is :', selected)
    #         print(self.tune[0][self.selectedNeuron])
    #         self.com1 = [np.array([self.raw.shape[0]-com[selected][0], self.raw.shape[1]-com[selected][1]])]
    #     else:
    #         logger.error('No neurons nearby where you clicked')
    #         self.com1 = [com[0]]
    #     return self.com1

    # def getFirstSelect(self):
    #     first = None
    #     if self.coords:
    #         com = [o['CoM'] for o in self.coords]
    #         #first = [np.array([self.raw.shape[0]-com[0][1], com[0][0]])]
    #         first = [np.array([self.raw.shape[0]-com[0][0], self.raw.shape[1]-com[0][1]])]
    #     return first

    