from improv.actor import Actor
import numpy as np
import logging
import cv2
import time
from pathlib import Path
import yaml



logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Generator(Actor):
    """Sample actor to generate data to pass into a sample processor.

    Intended for use along with sample_processor.py.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def setup(self):
        logger.info(f"Beginning setup for {self.name}")

         # load the configuration file
        source_folder = Path(__file__).resolve().parent.parent

        with open(f'{source_folder}/config.yaml', 'r') as file:
            config = yaml.safe_load(file)

        
        if '0' in self.name:
            self.video_path = config['video_path_0']
        elif '2' in self.name:
            self.video_path = config['video_path_2']
        else:
            self.video_path = config['video_path']

        self.cap = None
        self.frame_interval = 1.0 / config['fps']
        self.resize = config['resize']
        # self.name = "Generator"
        self.frame_num = 1
        self.gen_times = []
        self.full_times = []
        self.start = []

        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            logger.error("Error opening video file")
            return 
        total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))


        date = time.strftime("%Y%m%d")
        timestamp = time.strftime("%Y%m%d-%H%M")
        string = config['output_path']
        self.out_folder = Path(f"{string}/{date}/{timestamp}")
        self.out_folder.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output folder set to {self.out_folder}")
        logger.info(f'Total frames: {total_frames}')
        logger.info("Completed setup for Generator")

    def stop(self):

        logger.info("Generator stopping")
        if self.cap:
            self.cap.release()
        
        
        np.save(self.out_folder / "genstarts.npy", self.start)
        np.save(self.out_folder / "gen_latencies.npy", self.gen_times)
        np.save(self.out_folder / "full_latencies.npy", self.full_times)
        logger.info(f"Generator latencies saved to {self.out_folder}")
        return 0

    def runStep(self):

        if self.cap and self.cap.isOpened():
            self.start.append(time.time())
            self.start_perf = time.perf_counter()
            ret, self.frame = self.cap.read()
            if not ret:
                logger.info("End of video")
                self.stop()
                return
            def resize_frame(frame, resize):
                return cv2.resize(frame, (int(frame.shape[1] * resize), int(frame.shape[0] * resize)))
            # self.frame = resize_frame(self.frame, self.resize)
            # logger.info(f'Frame : {(self.frame.shape)}')
            # logger.info(f'Client: {self.client}')

            data_id = self.client.put(self.frame)
            # logger.info('Put data in store')
            try:
                self.q_out.put(data_id)
                # logger.info("Sent message on")

            except Exception as e:
                logger.error(f"--------------------------------Generator Exception: {e}")
            self.frame_num += 1
            self.gen_times.append(time.perf_counter() - self.start_perf)

            time.sleep(self.frame_interval)
            self.full_times.append(time.perf_counter() - self.start_perf)