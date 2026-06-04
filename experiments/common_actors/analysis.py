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

        
        self.stimuli_space = StimulusSpace()
        self.stim_space = self.stimuli_space.stim_space
        self.stim_space_dim = self.stimuli_space.param_space_size
        self.stimuli = self.stim_space['stimuli']
        self.d = self.stimuli.shape[0]
        self.param_space = self.stimuli_space.param_space
        self.param_space_size = self.stimuli_space.param_space_size 
        self.param_index_space = self.stimuli_space.param_index_space
        self.param_space_optim = self.stimuli_space.r1_params #param_space_optim

        self.calibration_not_moving_dots = 0
        self.before_amount = before_amount
        self.after_amount = after_amount
        self.calc_color = calc_color
        logger.info("from analysis viz this is calculating calc_color: {}".format(self.calc_color))
        

    def setup(self, param_file=None):
        '''
        '''
        np.seterr(divide='ignore')

        # TODO: same as behaviorAcquisition, need number of stimuli here. Make adaptive later
        self.num_stim = self.stim_space_dim
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


        self.xs = np.empty((0, self.d), dtype=int) #np.zeros((self.stim_space_dim))
        self.ys = np.zeros((1, self.stim_space_dim, 2))
        
        self.all_y = np.zeros((500, self.param_space_size)) #NOTE: what is 500? 
        self.stim_count = np.zeros((self.param_space_size, ), dtype=int) #np.zeros((dims), dtype=int)
        self.total_stim_counts = 0 #None
        self.old_stim_num = 0 

        self.stimX = []
        self.stimY = []
        self.testNum = 0 
        self.nID = 0
        self.stimText = None

        self.total_times = []
        self.puttime = []
        # self.colortime = []
        self.stimtime = []
        self.putstimtime = []
        self.getCtime = []
        self.timestamp = []
        self.LL = []
        self.fit_times = []
        self.ana_q_in_ts = []
        self.ana_input_stim_queue_ts = []
        self.ana_stim_out_ts = []
        self.ana_q_in_qsize = []
        self.ana_input_stim_queue_qsize = []

    def stop(self):
        print('Analysis broke, avg time per frame: ', np.mean(self.total_times, axis=0))
        print('Analysis broke, avg time per put analysis: ', np.mean(self.puttime))
        print('Analysis broke, avg time per put analysis: ', np.mean(self.fit_times))
        # print('Analysis broke, avg time per color frame: ', np.mean(self.colortime))
        print('Analysis broke, avg time per stim avg: ', np.mean(self.stimtime))
        print('Analysis got through ', self.frame, ' frames')

        np.savetxt('output/timing/analysis_frame_time.txt', np.array(self.total_times))
        np.savetxt('output/timing/analysis_timestamp.txt', np.array(self.timestamp))
        np.savetxt('output/analysis_estsAvg.txt', np.array(self.estsAvg))
        np.savetxt('output/analysis_proc_S.txt', np.array(self.S))
        np.savetxt('output/timing/analysis_puttime.txt', np.array(self.puttime))
        # np.savetxt('output/timing/analysis_colortime.txt', np.array(self.colortime))
        np.savetxt('output/timing/analysis_stimtime.txt', np.array(self.stimtime))
        # np.savetxt('output/timing/analysis_putstimtime.txt', np.array(self.putstimtime))
        np.savetxt('output/timing/analysis_getCtime.txt', np.array(self.getCtime))
        np.savetxt('output/timing/analysis_q_in_ts.txt', np.array(self.ana_q_in_ts))
        np.savetxt('output/timing/analysis_input_stim_queue_ts.txt', np.array(self.ana_input_stim_queue_ts))
        np.savetxt('output/timing/analysis_stim_out_ts.txt', np.array(self.ana_stim_out_ts))
        np.savetxt('output/timing/analysis_q_in_qsize.txt', np.array(self.ana_q_in_qsize))
        np.savetxt('output/timing/analysis_input_stim_queue_qsize.txt', np.array(self.ana_input_stim_queue_qsize))

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
            self.ana_q_in_qsize.append([self.frame, self.q_in.qsize()])
            if ids is not None and ids[0]==1:
                logger.info('analysis: missing frame')
                self.total_times.append(time.time()-t)
                self.q_out.put([1])
                raise Empty
            self.frame = ids[-1]
            self.ana_q_in_ts.append([self.frame, time.time()])
            t_get = time.time()
            self.coordDict = self.client.get(ids[0])
            # if self.calc_color:
            #     self.coordDict = self.client.get(ids[0])
            #     self.coords = [o['coordinates'] for o in self.coordDict]
            # else:
            #     self.coordDict = None
            #     self.coords = None
            self.image = self.client.get(ids[1])
            self.S = self.client.get(ids[2])
            self.getCtime.append(time.time()-t_get)

            self.C = self.S
            self.C = np.where(np.isnan(self.C), 0, self.C)

            self.coords = [o['coordinates'] for o in self.coordDict]
            
            # Compute tuning curves based on input stimulus
            # Just do overall average activity for now
            try: 
                sig = self.links['input_stim_queue'].get(timeout=0.0001) # sig: index pointing to a specific stimulus 
                self.ana_input_stim_queue_qsize.append([self.frame, self.links['input_stim_queue'].qsize()])
                self.ana_input_stim_queue_ts.append([self.frame, time.time()])
                self.updateStim_start(sig) #NOTE: do we even need a function for this? 
                logger.info('we called updatedStim_start')
                self.stimText = list(sig.values())
                self.total_stim_counts = self.stimText[-1]
                logger.info('total_stim_counts: {}'.format(self.total_stim_counts)) # add a logger: when send to optimizer
                self.should_send_frame_num = self.frame + self.after_amount
            except Empty as e:
                pass # no change in input stimulus
            except Exception as e:
                logger.error(f'an error occcured: {e}', exc_info=True)

            self.stimAvg_start()
            
            #NOTE: we don't need this (we're not plotting this) 
            self.globalAvg = np.mean(self.estsAvg[:,:8], axis=0)
            self.tune = [self.estsAvg[:,:8], self.globalAvg]

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
            self.putStimulus()
            if self.total_stim_counts > self.old_stim_num: 
                logger.info('sent stimX and stimY to optimizer at frame: {}'.format(self.frame))
                self.old_stim_num = self.total_stim_counts
            

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

        # frame = list(stim.keys())[0]
        # whichStim = int(stim[frame])
        

        frame = stim["frame"]
        whichStim = stim["indices"]
        tag = stim["tag"]
        if tag == "calibration":
            self.calibration_not_moving_dots += 1
            logger.info(f"current frame {frame}, whichstim {whichStim}, tag {tag}, counting {self.calibration_not_moving_dots}")
        # logger.info('sig {}, frame {},  whichStim {}, tag {}'.format(stim, frame, whichStim, tag))

        self.current_stim = whichStim

        if tag == 'optim' or tag == 'initial' or tag == 'calibration_initial' or tag == 'random' or tag == 'grid':
            params = self.param_space_optim[whichStim]
            multi_idx = self.stimuli_space.param_to_idx(params, tag=tag)

        else:
            multi_idx = self.param_index_space[whichStim]
        multi_idx = np.asarray(multi_idx, dtype=int)
        logger.info('multi_idx: {}'.format(multi_idx))

        self.xs = multi_idx #np.vstack([self.xs, multi_idx])
        logger.info('xs: {}'.format(self.xs))
        
        self.stim_count[whichStim] += 1
        # logger.info('stim_count: {}'.format(self.stim_count))

        curStim = 1
        
        self.allStims[frame] = {frame:whichStim} #stim
        if self.lastOnOff is None:
            self.lastOnOff = curStim
        # elif curStim == 1:
        self.stimStart = frame
        self.currentStim = whichStim

        logger.info('Stim {} started at frame {}'.format(self.currentStim, self.stimStart))
        logger.info('Frame: {} On off: {}'.format(self.frame, self.lastOnOff))
        logger.info('Current data frame is : {}'.format(self.frame))

    def putAnalysis(self):
        ''' Throw things to DS and put IDs in queue for Visual
        '''
        t = time.time()
        # logger.info(f"Cx shape {self.Cx.shape}, Call shape {self.Call.shape}, Cpop shape {self.Cpop.shape}, tune shape {len(self.tune)} with len {self.tune[0].shape}, color shape {self.color.shape}, coordDict len {len(self.coordDict)}, allStims len {len(self.allStims)}")
        
        ids = []
        ids.append(self.client.put(self.Cx))    #, 'Cx'+str(self.frame))) 
        ids.append(self.client.put(self.Call))  #, 'Call'+str(self.frame)))
        ids.append(self.client.put(self.Cpop))  #, 'Cpop'+str(self.frame)))
        ids.append(self.client.put(self.tune))  #, 'tune'+str(self.frame))) (probably dont need)
        ids.append(self.client.put(self.color)) #, 'color'+str(self.frame))) (we should rename bc it's not colored (motion correction))
        ids.append(self.client.put(self.coordDict)) #, 'analys_coords'+str(self.frame)))
        ids.append(self.client.put(self.allStims))  #, 'stim'+str(self.frame)))
        ids.append(self.client.put(self.tc_list)) #, 'tc_list'))
        ids.append(self.frame)
        
        # logger.info('ids: {}'.format(ids))
        self.q_out.put(ids)
        self.puttime.append(time.time()-t)

    def putStimulus(self):
        ''' Throw things to DS and put IDS in queue for Optimizer
        '''
        # t = time.time()
        ids = []
        ids.append(self.client.put(self.stimX))   #, 'stimX'+str(self.frame)))
        ids.append(self.client.put(self.stimY))   #, 'stimY'+str(self.frame)))
        ids.append(self.client.put(self.frame))
        ids.append(self.client.put(self.testNum)) #, 'stim_testNum'+str(self.frame)))
        ids.append(self.client.put(self.nID))     #, 'stim_nID'+str(self.frame)))
        ids.append(self.client.put(self.calibration_not_moving_dots))
        ids.append(self.client.put(self.total_stim_counts))
        self.links['stim_out'].put(ids)
        self.ana_stim_out_ts.append([self.frame, time.time()])
        # self.putstimtime.append(time.time()-t)

    def stimAvg_start(self): #TODO: need to rewrite this section (since ys will no longer be a dict)
        t = time.time()

        ests = self.C
        buffer_len = ests.shape[1]
        buffer_start_idx = max(0, self.frame - (buffer_len - 1))

        if self.ests.shape[0]<ests.shape[0]:
            diff = ests.shape[0] - self.ests.shape[0]
            # added more neurons, grow the array
            self.ests = np.pad(self.ests, ((0,diff),(0,0),(0,0)), mode='constant')

        if self.currentStim is not None:
            s_idx = int(self.currentStim)
            local_idx_now = self.frame - buffer_start_idx
            if self.stimStart == self.frame:
                local_end = local_idx_now + 1
                local_start = max(0, local_end - self.before_amount)
               
                # mean_val = np.mean(ests[:, -self.before_amount:], 1)  # relative indexing; need to be replaced
                mean_val = np.mean(ests[:, local_start:local_end], 1)
                self.ests[:, self.currentStim, 1] = (self.counter[self.currentStim, 1] * self.ests[:, self.currentStim, 1] + mean_val) / (self.counter[self.currentStim, 1] +1)
                self.counter[self.currentStim, 1] += self.before_amount
            
            elif self.frame in range(self.stimStart+1, self.stimStart+2):
                target_frame = self.frame - 1
                local_idx = target_frame - buffer_start_idx

                if 0 <= local_idx < buffer_len:
                    val = ests[:, local_idx]
                    self.ests[:, self.currentStim, 1] = (self.counter[self.currentStim, 1] * self.ests[:, self.currentStim, 1] + val) / (self.counter[self.currentStim, 1] + 1)
                    self.counter[self.currentStim, 1] += 1
                else:
                    logger.warning(f"Frame {target_frame} missing from 500-frame buffer.")

            elif self.frame in range(self.stimStart+2, self.stimStart+self.after_amount):
                target_frame = self.frame - 1
                local_idx = target_frame - buffer_start_idx
                
                if 0 <= local_idx < buffer_len:
                    val = ests[:, local_idx]
                    self.ests[:, self.currentStim, 0] = (self.counter[self.currentStim, 0] * self.ests[:, self.currentStim, 0] + val) / (self.counter[self.currentStim, 0] + 1)
                    self.counter[self.currentStim, 0] += 1
                else:
                    logger.warning(f"Frame {target_frame} missing from 500-frame buffer.")


            if self.frame == self.stimStart + self.after_amount:
                logger.info('appending to X: {}'.format(self.xs))

                self.stimX.append(self.xs)
                local_end = local_idx_now + 1
                local_start = max(0, local_end - self.after_amount)
                self.stimY.append(np.mean(ests[:, local_start:local_end], 1))

                # self.stimY.append(np.mean(ests[:, -self.after_amount:], 1))  # relative indexing
                logger.info(f"done appending at frame {self.frame}. before the estimated frame num {self.should_send_frame_num}? {self.frame <= self.should_send_frame_num}")
                logger.info('at frame {} we have {} neurons right now'.format(self.frame, ests.shape[0]))
                self.testNum += 1
                numN = self.ests.shape[0]
                
                sc = self.stim_count[self.currentStim]
                idx = int(self.currentStim)
                # logger.info(f"this is s_idx {s_idx}, this is idx {idx}, same? {s_idx == idx}")
                self.all_y[:numN, idx] = ((sc-1) * self.all_y[:numN, idx] + self.stimY[-1]) / sc

        self.estsAvg = np.squeeze(self.ests[:, :, 0] - self.ests[:, :, 1])
        self.estsAvg = np.where(np.isnan(self.estsAvg), 0, self.estsAvg)
        self.estsAvg[self.estsAvg == np.inf] = 0
        self.estsAvg[self.estsAvg < 0] = 0

        self.stimtime.append(time.time() - t)

    def plotColorFrame(self):
        ''' Computes colored nicer background+components frame
        '''
        # t = time.time()
        image = self.image
        
        tc_list = []
            #TODO: don't stack image each time?
        if self.calc_color:
            color = np.stack([image, image, image, image], axis=-1).astype(np.uint8).copy()
            color[...,3] = 255
            if self.coords is not None:
                # activity = np.zeros((len(self.coords),self.C.shape[0]))
                for i,c in enumerate(self.coords):
                    #c = np.array(c)
                    try:
                        pixels = c[~np.isnan(c).any(axis=1)].astype(int)
                        #TODO: Compute all colors simultaneously! then index in...
                        tc = self._tuningColor(i, color[pixels[:,1], pixels[:,0]])
                        tc_list.append(tc)
                        cv2.fillConvexPoly(color, pixels, tc)
                    except Exception as e:
                        logger.error('Error in fill poly: {}'.format(e))
                        pass
        
                    
                    # if pixels.size > 0:
                    #     npx = np.unique(pixels, axis=0)
                    #     act = self.C[:,npx[:,1],npx[:,0]]
                    #     activity[i] = np.sum(act, axis=1)

        ## Note: try pixelwise C display

        # TODO: keep list of neural colors. Compute tuning colors and IF NEW, fill ConvexPoly. 
        else:
            color = image

        # self.colortime.append(time.time()-t)
        # TODO: not sure if this is ok
        if not tc_list: # tc_list is empty
            tc_list = None
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