from improv.actor import Actor, Signal
from improv.store import ObjectNotFoundError
from queue import Empty
import numpy as np
import time
import cv2

import logging; logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class VizStimAnalysis(Actor):

    def __init__(self, *args, stimuli = None, before_amount=2, after_amount=10, calc_color = True, **kwargs):
        super().__init__(*args, **kwargs)

        self.calibration_not_moving_dots = 0
        self.before_amount = before_amount
        self.after_amount = after_amount


    def setup(self, param_file=None):
        '''
        '''

        if self.client is None:
            self._getStoreInterface()

        np.seterr(divide='ignore')

        # TODO: same as behaviorAcquisition, need number of stimuli here. Make adaptive later
        self.frame = 0
        # self.curr_stim = 0 #start with zeroth stim unless signaled otherwise
        self.stim = {}
        self.stimStart = -1
        self.currentStim = None
        self.window = 150 #TODO: make user input, choose scrolling window for Visual
        self.C = None
        self.S = None
        self.Call = None
        self.Cx = None
        self.Cpop = None
        self.coords = None
        self.color = None
        self.tc_list = None  # TODO: not sure if we need this
        self.runMean = None
        self.runMeanOn = None
        self.runMeanOff = None
        self.lastOnOff = None
        self.recentStim = [0]*self.window
        self.currStimID = np.zeros((8, 1000000)) #FIXME
        self.currStim = -10
        self.allStims = {}
        self.estsAvg = None





    def runStep(self):
        ''' Take numpy estimates and frame_number
            Create X and Y for plotting
        '''
        t = time.time()
        ids = None
        
        try:
            ids = self.q_in.get(timeout=0.0001)
            if ids is not None and ids[0]==1:
                logger.info('analysis: missing frame')
                self.q_out.put([1])
                raise Empty
            self.frame = ids[-1]
            t_get = time.time()
            self.coordDict = self.client.get(ids[0])
            self.image = self.client.get(ids[1])
            self.S = self.client.get(ids[2])

            self.C = self.S
            self.C = np.where(np.isnan(self.C), 0, self.C)

            self.coords = [o['coordinates'] for o in self.coordDict]
            

            self.color, self.tc_list = self.plotColorFrame()
            
            if self.frame >= self.window:
                window = self.window
                self.Cx = np.arange(self.frame-window,self.frame)
            else:
                window = self.frame
                self.Cx = np.arange(0,self.frame)

            if self.C.shape[1]>0:
                self.Cpop = np.nanmean(self.C, axis=0)
                if np.isnan(self.Cpop).any():
                    logger.error('Nan in Cpop')
                self.Call = self.C #already a windowed version #[:,self.frame-window:self.frame]

            # trim C all and C pop for faster communication
            if self.Call is not None and len(self.Cx) > 0:
                self.Call = self.Call[:, -len(self.Cx):]
                self.Cpop = self.Cpop[-len(self.Cx):]
            self.putAnalysis()


        except ObjectNotFoundError:
            logger.error('Estimates unavailable from store, droppping')
        except Empty as e:
            pass
            # logger.error('Default queue in was empty')
        except Exception as e:
            logger.exception('Error in analysis: {}'.format(e))
    


    def putAnalysis(self):
        ''' Throw things to DS and put IDs in queue for Visual
        '''
        t = time.time()
        # logger.info(f"Cx shape {self.Cx.shape}, Call shape {self.Call.shape}, Cpop shape {self.Cpop.shape}, tune shape {len(self.tune)} with len {self.tune[0].shape}, color shape {self.color.shape}, coordDict len {len(self.coordDict)}, allStims len {len(self.allStims)}")
        
        ids = []
        ids.append(self.client.put(self.Cx))    #, 'Cx'+str(self.frame))) 
        ids.append(self.client.put(self.Call))  #, 'Call'+str(self.frame)))
        ids.append(self.client.put(self.Cpop))  #, 'Cpop'+str(self.frame)))
        ids.append(self.client.put(None))  #, 'tune'+str(self.frame))) (probably dont need)
        ids.append(self.client.put(self.color)) #, 'color'+str(self.frame))) (we should rename bc it's not colored (motion correction))
        ids.append(self.client.put(self.coordDict)) #, 'analys_coords'+str(self.frame)))
        ids.append(self.client.put(self.allStims))  #, 'stim'+str(self.frame)))
        ids.append(self.client.put(self.tc_list)) #, 'tc_list'))
        ids.append(self.frame)
        
        # logger.info('ids: {}'.format(ids))
        self.q_out.put(ids)


    def plotColorFrame(self):
        ''' Computes colored nicer background+components frame
        '''
        t = time.time()
        image = self.image
        
        tc_list = []
        color = image

        # TODO: not sure if this is ok
        if not tc_list: # tc_list is empty
            tc_list = None
        return color, tc_list