import os
import time
import h5py
import pickle 
import cv2
import numpy as np
from improv.actor import Actor, RunManager
from queue import Empty
import logging; logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class VideoSaver(Actor):

    def __init__(self, *args, init_frame=None, init_filename=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.frame_number = 0
        self.save_ind = 0
        self.initial_frame_num = init_frame     # Number of frames for initialization
        self.init_filename = init_filename
        self.dropped_frames_vid = []
        
    
    def setup(self):
        self.saveArray = []
        if not os.path.exists(self.init_filename):
            ## Save initial set of frames to output/initialization.h5
            self.kill_flag = False
            while self.frame_number < self.initial_frame_num:
                self.runStep()

            self.imgs = np.array(self.saveArray)
            f = h5py.File(self.init_filename, 'w', libver='earliest')
            f.create_dataset("default", data=self.imgs)
            f.close()
    
    def stop(self):
        logger.info('Video Saver stopping procedure --')
        self.imgs = np.array(self.saveArray)
        logger.info('VIDSAVER Trying to save 1')
        f = h5py.File('output/sample_stream_end.h5', 'w', libver='earliest')
        logger.info('VIDSAVER Trying to save 2')
        f.create_dataset("default", data=self.imgs)
        logger.info('VIDSAVER Trying to save 3')
        f.close()
        logger.info('VIDSAVER Trying to save 4')
    
    def runStep(self):
        frame = self._obtain_new_img()
        if frame is not None:
            try:
                self.frame = self.client.get(frame[0][str(self.frame_number)])
                self.saveArray.append(self.frame)
            except ObjectNotFoundError:
                logger.error("VidSaver: Frame {} unavailable from store, droppping"
                             .format(self.frame_number))
                self.dropped_frames_vid.append(self.frame_number)
            except KeyError as e:
                logger.error("VidSaver: Key error... {0}".format(e))
                # Proceed at all costs
                self.dropped_frames_vid.append(self.frame_number)
            except Exception as e:
                logger.error("VidSaver error: {}: {} during frame number {}"
                             .format(type(e).__name__, e, self.frame_number))
                print(traceback.format_exc())
                self.droppedropped_frames_vidd_frames.append(self.frame_number)
            self.frame_number += 1

        if len(self.saveArray) >= 1000:
            self.imgs = np.array(self.saveArray)
            logger.info(f"saving savearray {self.imgs.shape}")
            f = h5py.File('output/sample_stream'+str(self.save_ind)+'.h5', 'w', libver='earliest')
            f.create_dataset("default", data=self.imgs)
            f.close()
            self.save_ind += 1
            del self.saveArray
            self.saveArray = []
            logger.info('after saving internal')

    def _obtain_new_img(self):
        try:
            res = self.q_in.get(timeout=0.0005)
            return res
        # TODO: add'l error handling
        except Empty:
            return None