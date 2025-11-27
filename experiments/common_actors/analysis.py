from improv.actor import Actor, Signal
from improv.store import ObjectNotFoundError
from queue import Empty
import numpy as np
import time
import cv2
import colorsys
import scipy
import pickle

import logging; logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

from experiments.savier.gen_stim import StimulusSpace

class VizStimAnalysis(Actor):

    def __init__(self, *args, stimuli = None, before_amount=2, after_amount=10, calc_color = True, **kwargs):
        super().__init__(*args)

        # self.stimuli = np.load(stimuli, allow_pickle=True)
        self.stimuli_space = StimulusSpace()
        self.stim_space = self.stimuli_space.stim_space
        self.stim_space_dim = self.stimuli_space.param_space_size
        # self.stimuli = np.array([np.sort(stim) for stim in self.stim_space['stimuli']], dtype=object)
        self.stimuli = self.stim_space['stimuli']
        # logger.info('reading in stim: {}'.format(self.stimuli))


        self.before_amount = before_amount
        self.after_amount = after_amount
        self.calc_color = calc_color
        logger.info("from analysis viz this is calculating calc_color: {}".format(self.calc_color))
        

    def setup(self, param_file=None):
        '''
        '''
        np.seterr(divide='ignore')

        # TODO: same as behaviorAcquisition, need number of stimuli here. Make adaptive later
        self.num_stim = 12 
        self.frame = 0
        # self.curr_stim = 0 #start with zeroth stim unless signaled otherwise
        self.stim = {}
        self.stimStart = -1
        self.currentStim = None
        self.ests = np.zeros((1, self.num_stim, 2)) #number of neurons, number of stim, on and baseline
        self.counter = np.ones((self.num_stim,2))
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
        
        #TODO: need to rewrite xs and ys based on the size of the parameter space? 
        self.xs = np.zeros((self.stim_space_dim))
        self.ys = np.zeros((1, self.stim_space_dim, 2))
        
        #FIXME: hardcoded
        dims = [x.shape[0] for x in self.stimuli]
        # dims = [getattr(self, f'x_{label}').shape[0] for label in self.stim_space['labels']]
        self.all_y = np.zeros((500, *dims)) #NOTE: what is 500? 
        self.stim_count = np.zeros((dims))

        self.stimX = []
        self.stimY = []
        self.testNum = 0 
        self.nID = 0
        self.stimText = None

        self.total_times = []
        self.puttime = []
        self.colortime = []
        self.stimtime = []
        self.timestamp = []
        self.LL = []
        self.fit_times = []

    def stop(self):
        print('Analysis broke, avg time per frame: ', np.mean(self.total_times, axis=0))
        print('Analysis broke, avg time per put analysis: ', np.mean(self.puttime))
        print('Analysis broke, avg time per put analysis: ', np.mean(self.fit_times))
        print('Analysis broke, avg time per color frame: ', np.mean(self.colortime))
        print('Analysis broke, avg time per stim avg: ', np.mean(self.stimtime))
        print('Analysis got through ', self.frame, ' frames')

        np.savetxt('output/timing/analysis_frame_time.txt', np.array(self.total_times))
        np.savetxt('output/timing/analysis_timestamp.txt', np.array(self.timestamp))
        np.savetxt('output/analysis_estsAvg.txt', np.array(self.estsAvg))
        np.savetxt('output/analysis_proc_S.txt', np.array(self.S))

        with open("output/analysis_stimY.pkl", 'wb') as f:
            pickle.dump(self.stimY, f)
        with open("output/analysis_stimX.pkl", 'wb') as f:
            pickle.dump(self.stimX, f)
            
        stim = []
        for i in self.allStims.keys():
            stim.append(self.allStims[i])
        print('Stims ------------------------------')
        print(self.allStims)
        np.save('output/used_stims.npy', np.array(stim))
        # np.savetxt('output/used_stims.txt', self.currStimID)

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
                self.total_times.append(time.time()-t)
                self.q_out.put([1])
                raise Empty
            self.frame = ids[-1]
            self.coordDict = self.client.get(ids[0])
            self.image = self.client.get(ids[1])
            self.S = self.client.get(ids[2])

            self.C = self.S
            self.C = np.where(np.isnan(self.C), 0, self.C)

            self.coords = [o['coordinates'] for o in self.coordDict]
            
            # Compute tuning curves based on input stimulus
            # Just do overall average activity for now
            try: 
                sig = self.links['input_stim_queue'].get(timeout=0.0001) # sig: index pointing to a specific stimulus 
                self.updateStim_start(sig) #NOTE: do we even need a function for this? 
                logger.info('we called updatedStim_start')
                self.stimText = list(sig.values())
            except Empty as e:
                pass # no change in input stimulus
            except Exception as e:
                logger.error(f'an error occcured: {e}', exc_info=True)

            self.stimAvg_start()
            
            #NOTE: we don't need this (we're not plotting this) 
            self.globalAvg = np.mean(self.estsAvg[:,:8], axis=0)
            self.tune = [self.estsAvg[:,:8], self.globalAvg]


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

            self.putAnalysis()
            self.putStimulus()
            self.timestamp.append([time.time(), self.frame])
            self.total_times.append(time.time()-t)

        except ObjectNotFoundError:
            logger.error('Estimates unavailable from store, droppping')
        except Empty as e:
            pass
            # logger.error('Default queue in was empty')
        except Exception as e:
            logger.exception('Error in analysis: {}'.format(e))
    

    def updateStim_start(self, stim):
        pass

    def putAnalysis(self):
        ''' Throw things to DS and put IDs in queue for Visual
        '''
        t = time.time()
        ids = []
        ids.append(self.client.put(self.Cx))    #, 'Cx'+str(self.frame))) 
        ids.append(self.client.put(self.Call))  #, 'Call'+str(self.frame)))
        ids.append(self.client.put(self.Cpop))  #, 'Cpop'+str(self.frame)))
        ids.append(self.client.put(self.tune))  #, 'tune'+str(self.frame))) (probably dont need)
        ids.append(self.client.put(self.color)) #, 'color'+str(self.frame))) (we should rename bc it's not colored (motion correction))
        ids.append(self.client.put(self.coordDict)) #, 'analys_coords'+str(self.frame)))
        ids.append(self.client.put(self.allStims))  #, 'stim'+str(self.frame)))
        # ids.append(self.client.put(self.y_results, 'yres'+str(self.frame)))
        # ids.append(self.client.put(self.stimText, 'yres'+str(self.frame)))
        # ids.append(self.client.put(self.all_y, 'all_y'))
        ids.append(self.client.put(self.tc_list)) #, 'tc_list'))
        ids.append(self.frame)
        
        # logger.info('ids: {}'.format(ids))
        self.q_out.put(ids)
        self.puttime.append(time.time()-t)

    def putStimulus(self):
        ''' Throw things to DS and put IDS in queue for Optimizer
        '''
        ids = []
        ids.append(self.client.put(self.stimX))   #, 'stimX'+str(self.frame)))
        ids.append(self.client.put(self.stimY))   #, 'stimY'+str(self.frame)))
        ids.append(self.client.put(self.frame))
        ids.append(self.client.put(self.testNum)) #, 'stim_testNum'+str(self.frame)))
        ids.append(self.client.put(self.nID))     #, 'stim_nID'+str(self.frame)))
        self.links['stim_out'].put(ids)

    def stimAvg_start(self): #TODO: need to rewrite this section (since ys will no longer be a dict)
        t = time.time()

        ests = self.C
        
        if self.ests.shape[0]<ests.shape[0]:
            diff = ests.shape[0] - self.ests.shape[0]
            # added more neurons, grow the array
            self.ests = np.pad(self.ests, ((0,diff),(0,0),(0,0)), mode='constant')

        if self.currentStim is not None:
            if self.stimStart == self.frame:

                mean_val = np.mean(ests[:, self.frame-self.before_amount:self.frame], 1)

                self.ests[:, self.currentStim, 1] = (self.counter[self.currentStim, 1] * self.ests[:, self.currentStim, 1] + mean_val) / (self.counter[self.currentStim, 1] +1)
                self.counter[self.currentStim, 1] += self.before_amount
            
            elif self.frame in range(self.stimStart+1, self.stimStart+2):

                val = ests[:, self.frame-1]

                self.ests[:, self.currentStim, 1] = (self.counter[self.currentStim, 1] * self.ests[:, self.currentStim, 1] + val) / (self.counter[self.currentStim, 1] +1)
                self.counter[self.currentStim, 1] += 1
            
            elif self.frame in range(self.stimStart+2, self.stimStart+self.after_amount):

                val = ests[:, self.frame-1]

                self.ests[:, self.currentStim, 0] = (self.counter[self.currentStim, 0] * self.ests[:, self.currentStim, 0] + val) / (self.counter[self.currentStim, 0] +1)
                self.counter[self.currentStim, 0] += 1

            if self.frame == self.stimStart + self.after_amount:
                logger.info('appending to X')

                self.stimX.append(self.xs)
                self.stimY.append(np.mean(ests[:, self.frame-self.after_amount:self.frame], 1))
                logger.info('we have {} neurons right now'.format(ests.shape[0]))

                self.testNum += 1
                sc = self.stim_count[int(self.xs)]
                numN = self.ests.shape[0]
                idx = int(self.xs)
                self.all_y[(slice(0, numN), + idx)] = ((sc-1) * self.all_y[(slice(0, numN), ) + idx] + self.stimY[-1]) / sc
        
        self.estsAvg = np.squeeze(self.ests[:, :, 0] - self.ests[:, :, 1])
        self.estsAvg = np.where(np.isnan(self.estsAvg), 0, self.estsAvg)
        self.estsAvg[self.estsAvg == np.inf] = 0
        self.estsAvg[self.estsAvg < 0] = 0

        self.stimtime.append(time.time() - t)
