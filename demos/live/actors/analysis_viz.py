from improv.actor import Actor, Signal
from improv.store import ObjectNotFoundError
from queue import Empty
import numpy as np
import time
import cv2
import colorsys
import scipy

import logging; logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class VizStimAnalysis(Actor):

    def __init__(self, *args, stimuli = None, before_amount=2, after_amount=10, **kwargs):
        super().__init__(*args)

        self.stimuli = np.load(stimuli, allow_pickle=True)
        self.before_amount = before_amount
        self.after_amount = after_amount
        

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
        self.runMean = None
        self.runMeanOn = None
        self.runMeanOff = None
        self.lastOnOff = None
        self.recentStim = [0]*self.window
        self.currStimID = np.zeros((8, 1000000)) #FIXME
        self.currStim = -10
        self.allStims = {}
        self.estsAvg = None

        # self.x_angle = np.linspace(0,350,num=15)
        # self.x_vel = [0.02, 0.06, 0.1] #np.around(np.linspace(0.02, 0.1, num=4), decimals=2)
        # self.x_freq = np.array([20, 40, 60] ) #32 #[0, 32] #np.arange(5,80,5)
        # self.x_contrast = np.arange(5)

        self.x_angle = self.stimuli[0]
        self.x_vel = self.stimuli[1]
        # self.x_freq = self.stimuli[2]
        # self.x_contrast = self.stimuli[3]

        self.counters = {}
        self.counters['angle'] = np.ones((self.x_angle.shape[0],2))
        self.counters['vel'] =  np.ones((self.x_vel.shape[0],2))
        # self.counters['freq'] =  np.ones((self.x_freq.shape[0],2))
        # self.counters['contrast'] =  np.ones((self.x_contrast.shape[0],2))

        self.ys = {}
        self.ys['angle'] = np.zeros((1, self.x_angle.shape[0], 2))
        self.ys['vel'] = np.zeros((1, self.x_vel.shape[0], 2))
        # self.ys['freq'] = np.zeros((1, self.x_freq.shape[0], 2))
        # self.ys['contrast'] = np.zeros((1, self.x_contrast.shape[0], 2))

        self.y_results = {}
        self.y_results['angle'] = None
        self.y_results['vel'] = None
        # self.y_results['freq'] = None
        # self.y_results['contrast'] = None

        self.xs = {}
        self.xs['angle'] = 0
        self.xs['vel'] = 0
        # self.xs['freq'] = 0
        # self.xs['contrast'] = 0

        self.all_y = np.zeros((500, self.x_angle.shape[0],self.x_angle.shape[0]))
        self.stim_count = np.zeros((self.x_angle.shape[0],self.x_angle.shape[0]))

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
                print('analysis: missing frame')
                self.total_times.append(time.time()-t)
                self.q_out.put([1])
                raise Empty
            # t = time.time()
            self.frame = ids[-1]
            (self.coordDict, self.image, self.S) = self.client.getList(ids[:-1])
            self.C = self.S

            # why are there nans in C?
            self.C = np.where(np.isnan(self.C), 0, self.C)

            self.coords = [o['coordinates'] for o in self.coordDict]
            
            # Compute tuning curves based on input stimulus
            # Just do overall average activity for now
            # logger.info('Analysis.input_stim_queue: {}'.format(self.links['input_stim_queue']))
            # logger.info('Analysis.input_stim_queue GET: {}'.format(self.links['input_stim_queue'].get(timeout=0.0001)))
            # logger.info('ANALYSIS LINKS -------------------')
            # sig = self.links['input_stim_queue'].get(timeout=0.0001)
            # logger.info('analysis.input_stim_queue: {}'.format(sig))
            # logger.info('ANALYSIS LINKS (done) ------------')
            try: 
                ## stim format: stim, stimonOff, angle, vel, freq, contrast
                # logger.info('ANALYSIS - inside try block ------------')
                sig = self.links['input_stim_queue'].get(timeout=0.0001)
                self.updateStim_start(sig)
                logger.info('we called updatedStim_start')
                self.stimText = list(sig.values())
            except Empty as e:
                pass #no change in input stimulus
                # logger.error(f'an error occcured: {e}', exc_info=True)
            except Exception as e:
                logger.error(f'an error occcured: {e}', exc_info=True)

            self.stimAvg_start()
            
            self.globalAvg = np.mean(self.estsAvg[:,:8], axis=0)
            self.tune = [self.estsAvg[:,:8], self.globalAvg]

            # Compute coloring of neurons for processed frame
            # Also rotate and stack as needed for plotting
            # TODO: move to viz, but we don't need to compute this 30 times/sec
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

            # logger.info('BEFORE putAnalysis() -- Call = {}'.format(self.Call))
            # logger.info('BEFORE putAnalysis() -- Cx = {}'.format(self.Cx))

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
        ''' Rearrange the info about stimulus into
            cardinal directions and frame <--> stim correspondence.

            self.stimStart is the frame where the stimulus started.
        '''
        # get frame number and stimID
        frame = list(stim.keys())[0]
        whichStim = stim[frame][0]
        # convert stimID into 8 cardinal directions
        stimID = self.IDstim(int(whichStim))

        angle = stim[frame][0]
        vel = stim[frame][1]
        # freq = stim[frame][4]
        # contrast = stim[frame][5]

        self.xs['angle'] = np.argwhere(angle==self.x_angle)[0]
        # print(stim)
        # print(self.x_vel)
        self.xs['vel'] = np.argwhere(vel==self.x_vel)[0]
        # self.xs['freq'] = np.argwhere(freq==self.x_freq)[0]
        # self.xs['contrast'] = np.argwhere(contrast==self.x_contrast)[0]

        self.stim_count[int(self.xs['angle']), int(self.xs['vel'])] += 1

        # assuming we have one of those 8 stimuli
        # if stimID != -10:

        if stimID not in self.allStims.keys():
            self.allStims.update({stimID:[]})

                # # account for stimuli we haven't yet seen
                # if stimID not in self.stimStart.keys():
                #     self.stimStart.update({stimID:None})
        # determine if this is a new stimulus trial
        # if abs(stim[frame][1])>1 :
        curStim = 1 #on
        self.allStims[stimID].append(frame)
        # else:
        #     curStim = 0 #off
        # paradigm for these trials is for each stim: [off, on, off]
        if self.lastOnOff is None:
            self.lastOnOff = curStim
        elif curStim == 1: #self.lastOnOff == 0 and curStim == 1: #was off, now on
            # select this frame as the starting point of the new trial
            # and stimulus has started to be shown
            # All other computations will reference this point
            self.stimStart = frame 
            self.currentStim = stimID
            if stimID<8:
                if stimID >=0:
                    self.currStimID[stimID, frame] = 1
            # NOTE: this overwrites historical info on past trials
            logger.info('Stim {} started at {}'.format(stimID,frame))
        else:
            self.currStimID[:, frame] = np.zeros(8)
        # self.lastOnOff = curStim
        logger.info(f'Frame: {frame} On off : {self.lastOnOff}')
        logger.info(f'Current data frame is : {self.frame}')

    def putAnalysis(self):
        ''' Throw things to DS and put IDs in queue for Visual
        '''
        t = time.time()
        ids = []
        # stim = [self.lastOnOff, self.currStim]
        # self.stimText = None
        # logger.info('Cx: {}'.format(self.Cx))
        ids.append(self.client.put(self.Cx, 'Cx'+str(self.frame)))
        ids.append(self.client.put(self.Call, 'Call'+str(self.frame)))
        ids.append(self.client.put(self.Cpop, 'Cpop'+str(self.frame)))
        ids.append(self.client.put(self.tune, 'tune'+str(self.frame)))
        ids.append(self.client.put(self.color, 'color'+str(self.frame)))
        ids.append(self.client.put(self.coordDict, 'analys_coords'+str(self.frame)))
        ids.append(self.client.put(self.allStims, 'stim'+str(self.frame)))
        # ids.append(self.client.put(self.y_results, 'yres'+str(self.frame)))
        # ids.append(self.client.put(self.stimText, 'yres'+str(self.frame)))
        # ids.append(self.client.put(self.all_y, 'all_y'))
        ids.append(self.client.put(self.tc_list, 'tc_list'))
        ids.append(self.frame)
        
        # logger.info('ids: {}'.format(ids))
        self.q_out.put(ids)
        self.puttime.append(time.time()-t)

    def putStimulus(self):
        # pass
        logger.info('stimX: {}'.format(self.stimX))
        logger.info('stimY: {}'.format(self.stimY))
        ids = []
        ids.append(self.client.put(self.stimX, 'stimX'+str(self.frame)))
        ids.append(self.client.put(self.stimY, 'stimY'+str(self.frame)))
        ids.append(self.client.put(self.testNum, 'stim_testNum'+str(self.frame)))
        ids.append(self.client.put(self.nID, 'stim_nID'+str(self.frame)))
        self.links['stim_out'].put(ids)

    def stimAvg_start(self):
        t = time.time()

        ests = self.C
        
        if self.ests.shape[0]<ests.shape[0]:
            diff = ests.shape[0] - self.ests.shape[0]
            # added more neurons, grow the array
            self.ests = np.pad(self.ests, ((0,diff),(0,0),(0,0)), mode='constant')
            # print('------------------Grew:', self.ests.shape)
            for key in self.ys.keys():
                self.ys[key] = np.pad(self.ys[key], ((0,diff),(0,0),(0,0)), mode='constant')
            # self.y_angle = np.pad(self.y_angle, ((0,diff),(0,0),(0,0)), mode='constant')
            # self.y_vel = np.pad(self.y_vel, ((0,diff),(0,0),(0,0)), mode='constant')
            # self.y_freq = np.pad(self.y_freq, ((0,diff),(0,0),(0,0)), mode='constant')
            # self.y_contrast = np.pad(self.y_contrast, ((0,diff),(0,0),(0,0)), mode='constant')

        # self.before_amount = 5
        # self.after_amount = 20
        logger.info('currentStim is: {}'.format(self.currentStim))
        if self.currentStim is not None:
            # print('Computing for ', self.currentStim, ' starting at ', self.stimStart, ' but current  ', self.frame)

            if self.stimStart == self.frame:
                # print(' Starting compute for ', self.currentStim, ' at frame ', self.frame)
                # account for the baseline prior to stimulus onset
                
                mean_val = np.mean(ests[:,self.frame-self.before_amount:self.frame],1)

                self.ests[:,self.currentStim,1] = (self.counter[self.currentStim,1]*self.ests[:,self.currentStim,1] + mean_val)/(self.counter[self.currentStim,1]+1)
                self.counter[self.currentStim, 1] += self.before_amount

                for key in self.ys.keys():
                    ind = self.xs[key]
                    try:
                        self.ys[key][:,ind,1] = (self.counters[key][ind,1]*self.ys[key][:,ind,1] + mean_val[:,None])/(self.counters[key][ind,1]+1)
                    except:
                        print(key)
                        print(self.ys[key][:,ind,1].shape)
                        print(self.counters[key][ind,1].shape)
                        print(mean_val.shape)
                        print(((self.counters[key][ind,1]*self.ys[key][:,ind,1] + mean_val[:,None])/(self.counters[key][ind,1]+1)).shape)

                    self.counters[key][ind, 1] += self.before_amount
                    
                # self.y_angle[:,self.xa,1] = (self.counters['angle'][self.xa,1]*self.y_angle[:,self.xa,1] + mean_val)/(self.counter[self.xa,1]+1)
                # self.counters['angle'][self.xa, 1] += 10
                # self.y_vel[:,self.xv,1] = (self.counters['vel'][self.xv,1]*self.y_vel[:,self.xv,1] + mean_val)/(self.counter[self.xv,1]+1)
                # self.counters['vel'][self.xv, 1] += 10
                # self.y_freq[:,self.xf,1] = (self.counters['freq'][self.xf,1]*self.y_freq[:,self.xf,1] + mean_val)/(self.counter[self.xf,1]+1)
                # self.counters['freq'][self.xf, 1] += 10
                # self.y_contrast[:,self.xc,1] = (self.counters['contrast'][self.xc,1]*self.y_contrast[:,self.xc,1] + mean_val)/(self.counter[self.xc,1]+1)
                # self.counters['contrast'][self.xc, 1] += 10

            elif self.frame in range(self.stimStart+1, self.stimStart+2):
                val = ests[:,self.frame-1]
                self.ests[:,self.currentStim,1] = (self.counter[self.currentStim,1]*self.ests[:,self.currentStim,1] + val)/(self.counter[self.currentStim,1]+1)
                self.counter[self.currentStim, 1] += 1

                for key in self.ys.keys():
                    ind = self.xs[key]
                    try:
                        self.ys[key][:,ind,1] = (self.counters[key][ind,1]*self.ys[key][:,ind,1] + val[:,None])/(self.counters[key][ind,1]+1)
                    except:    
                        print(key)
                        print(self.ys[key][:,ind,1].shape)
                        print(self.counters[key][ind,1].shape)
                        print(mean_val.shape)
                        print(((self.counters[key][ind,1]*self.ys[key][:,ind,1] + mean_val)/(self.counters[key][ind,1]+1)).shape)
                    self.counters[key][ind, 1] += 1

                # self.y_angle[:,self.xa,1] = (self.counters['angle'][self.xa,1]*self.y_angle[:,self.xa,1] + val)/(self.counters['angle'][self.xa,1]+1)
                # self.counters['angle'][self.xa, 1] += 1
                # self.y_vel[:,self.xv,1] = (self.counters['vel'][self.xv,1]*self.y_vel[:,self.xv,1] + val)/(self.counters['vel'][self.xv,1]+1)
                # self.counters['vel'][self.xv, 1] += 1
                # self.y_freq[:,self.xf,1] = (self.counters['freq'][self.xf,1]*self.y_freq[:,self.xf,1] + val)/(self.counters['freq'][self.xf,1]+1)
                # self.counters['freq'][self.xf, 1] += 1
                # self.y_contrast[:,self.xc,1] = (self.counters['contrast'][self.xc,1]*self.y_contrast[:,self.xc,1] + val)/(self.counters['contrast'][self.xc,1]+1)
                # self.counters['contrast'][self.xc, 1] += 1

            elif self.frame in range(self.stimStart+2, self.stimStart+self.after_amount):
                val = ests[:,self.frame-1]
                self.ests[:,self.currentStim,0] = (self.counter[self.currentStim,0]*self.ests[:,self.currentStim,0] + val)/(self.counter[self.currentStim,0]+1)
                self.counter[self.currentStim, 0] += 1

                for key in self.ys.keys():
                    ind = self.xs[key]
                    self.ys[key][:,ind,0] = (self.counters[key][ind,0]*self.ys[key][:,ind,0] + val[:,None])/(self.counters[key][ind,0]+1)
                    self.counters[key][ind, 0] += 1

                    # self.y_angle[:,self.xa,0] = (self.counters['angle'][self.xa,0]*self.y_angle[:,self.xa,0] + val)/(self.counters['angle'][self.xa,0]+1)
                    # self.counters['angle'][self.xa, 0] += 1
                    # self.y_vel[:,self.xv,0] = (self.counters['vel'][self.xv,0]*self.y_vel[:,self.xv,0] + val)/(self.counters['vel'][self.xv,0]+1)
                    # self.counters['vel'][self.xv, 0] += 1
                    # self.y_freq[:,self.xf,0] = (self.counters['freq'][self.xf,0]*self.y_freq[:,self.xf,0] + val)/(self.counters['freq'][self.xf,0]+1)
                    # self.counters['freq'][self.xf, 0] += 1
                    # self.y_contrast[:,self.xc,0] = (self.counters['contrast'][self.xc,0]*self.y_contrast[:,self.xc,0] + val)/(self.counters['contrast'][self.xc,0]+1)
                    # self.counters['contrast'][self.xc, 0] += 1
            logger.info('frame_number: {}'.format(self.frame))
            logger.info('stimStart + self.after_amount: {}'.format((self.stimStart, self.after_amount)))
            if self.frame == self.stimStart + self.after_amount:
                logger.info('appending to X: {}'.format(list(self.xs.values())))
                self.stimX.append(list(self.xs.values()))
                self.stimY.append(np.mean(ests[:,self.frame-self.after_amount:self.frame],1))
                self.testNum += 1
                sc = self.stim_count[int(self.xs['angle']), int(self.xs['vel'])]
                numN = self.ests.shape[0]
                self.all_y[:numN, int(self.xs['angle']), int(self.xs['vel'])] = ((sc-1)*self.all_y[:numN, int(self.xs['angle']), int(self.xs['vel'])] + self.stimY[-1])/sc

        self.estsAvg = np.squeeze(self.ests[:,:,0] - self.ests[:,:,1])        
        self.estsAvg = np.where(np.isnan(self.estsAvg), 0, self.estsAvg)
        self.estsAvg[self.estsAvg == np.inf] = 0
        self.estsAvg[self.estsAvg<0] = 0

        for key in self.y_results.keys():
            self.y_results[key] = np.squeeze(self.ys[key][:,:,0] - self.ys[key][:,:,1])        
            self.y_results[key] = np.where(np.isnan(self.y_results[key]), 0, self.y_results[key])
            self.y_results[key][self.y_results[key] == np.inf] = 0
            self.y_results[key][self.y_results[key]<0] = 0

        # self.all_y[:, self.xs['angle'], self.xs['vel']] =  

        self.stimtime.append(time.time()-t)

    def plotColorFrame(self):
        ''' Computes colored nicer background+components frame
        '''
        t = time.time()
        image = self.image
        color = np.stack([image, image, image, image], axis=-1).astype(np.uint8).copy()
        color[...,3] = 255
        tc_list = []
            #TODO: don't stack image each time?
        if self.coords is not None:
            # activity = np.zeros((len(self.coords),self.C.shape[0]))
            for i,c in enumerate(self.coords):
                #c = np.array(c)
                pixels = c[~np.isnan(c).any(axis=1)].astype(int)
                #TODO: Compute all colors simultaneously! then index in...
                try:
                    tc = self._tuningColor(i, color[pixels[:,1], pixels[:,0]])
                    tc_list.append(tc)
                    cv2.fillConvexPoly(color, pixels, tc)
                except Exception as e:
                    # logger.error('Error in fill poly: {}'.format(e)).
                    pass
                
                
                # if pixels.size > 0:
                #     npx = np.unique(pixels, axis=0)
                #     act = self.C[:,npx[:,1],npx[:,0]]
                #     activity[i] = np.sum(act, axis=1)

        ## Note: try pixelwise C display

        # TODO: keep list of neural colors. Compute tuning colors and IF NEW, fill ConvexPoly. 

        self.colortime.append(time.time()-t)
        return color, tc_list

    def _tuningColor(self, ind, inten):
        ''' ind identifies the neuron by number
        '''
        ests = self.estsAvg
        #ests = self.tune_k[0] 
        if ests[ind] is not None: 
            try:
                return self.manual_Color_Sum(ests[ind])                
            except ValueError:
                return (255,255,255,0)
            except Exception:
                print('inten is ', inten)
                print('ests[i] is ', ests[ind])
        else:
            return (255,255,255,50)

    def manual_Color_Sum(self, x):
        ''' x should be length 12 array for coloring
            or, for k coloring, length 8
            Using specific coloring scheme from Naumann lab
        '''
        if x.shape[0] == 8:
            mat_weight = np.array([
            [1, 0.25, 0],
            [0.75, 1, 0],
            [0, 1, 0],
            [0, 0.75, 1],
            [0, 0.25, 1],
            [0.25, 0, 1.],
            [1, 0, 1],
            [1, 0, 0.25],
        ])
        elif x.shape[0] == 12:
            mat_weight = np.array([
                [1, 0.25, 0],
                [0.75, 1, 0],
                [0, 2, 0],
                [0, 0.75, 1],
                [0, 0.25, 1],
                [0.25, 0, 1.],
                [1, 0, 1],
                [1, 0, 0.25],
                [1, 0, 0],
                [0, 0, 1],
                [0, 0, 1],
                [1, 0, 0]
            ])
        else:
            print('Wrong shape for this coloring function')
            return (255, 255, 255, 10)

        color = x @ mat_weight

        blend = 0.8  
        thresh = 0.1   
        thresh_max = blend * np.max(color)

        color = np.clip(color, thresh, thresh_max)
        color -= thresh
        color /= thresh_max
        color = np.nan_to_num(color)

        if color.any() and np.linalg.norm(color-np.ones(3))>0.1: #0.35:
            color *=255
            return (color[0], color[1], color[2], 255)       
        else:
            return (255, 255, 255, 10)
    
    def IDstim(self, s):
        ''' Function to convert stim ID from Naumann lab experiment into
            the 8 cardinal directions they correspond to.
        ''' 
        stim = -10
        if s == 3:
            stim = 0
        elif s==10:
            stim = 1
        elif s==9:
            stim = 2
        elif s==16:
            stim = 3
        elif s==4:
            stim = 4
        elif s==14:
            stim = 5
        elif s==13:
            stim = 6
        elif s==12:
            stim = 7
        # add'l stim for specific coloring
        elif s==5:
            stim = 8
        elif s==6:
            stim = 9
        elif s==7:
            stim = 10
        elif s==8:
            stim = 11

        return stim
