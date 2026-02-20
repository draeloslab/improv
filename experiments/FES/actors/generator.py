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
        self.frame_num = 0
        
        # --- Timing logs ---
        # Wall-clock timestamp when each frame is produced (time.time())
        self.timestamps = []
        # Duration of actual work per frame: read + cvtColor + store put + q_out put (perf_counter)
        self.work_latencies = []
        # Duration including the sleep: work + time.sleep (perf_counter)
        self.full_latencies = []
        # Per-step breakdown (perf_counter durations in seconds)
        self.read_latencies = []    # cv2 read
        self.cvt_latencies = []     # cvtColor
        self.store_put_latencies = []  # client.put
        self.queue_put_latencies = []  # q_out.put

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
        self.done = False
        logger.info("Completed setup for Generator")

    def stop(self):

        logger.info("Generator stopping")
        if self.cap:
            self.cap.release()
        
        np.save(self.out_folder / "gen_timestamps.npy", self.timestamps)
        np.save(self.out_folder / "gen_work_latencies.npy", self.work_latencies)
        np.save(self.out_folder / "gen_full_latencies.npy", self.full_latencies)
        np.save(self.out_folder / "gen_read_latencies.npy", self.read_latencies)
        np.save(self.out_folder / "gen_cvt_latencies.npy", self.cvt_latencies)
        np.save(self.out_folder / "gen_store_put_latencies.npy", self.store_put_latencies)
        np.save(self.out_folder / "gen_queue_put_latencies.npy", self.queue_put_latencies)

        # Also save legacy names for backward compatibility
        np.save(self.out_folder / "genstarts.npy", self.timestamps)
        np.save(self.out_folder / "gen_latencies.npy", self.work_latencies)
        np.save(self.out_folder / "full_latencies.npy", self.full_latencies)

        logger.info(f"Generator latencies saved to {self.out_folder}")
        return 0

    def runStep(self):

        if self.done:
            return

        if self.cap and self.cap.isOpened():
            frame_start = time.time()
            perf_start = time.perf_counter()

            # --- Step 1: Read frame ---
            t0 = time.perf_counter()
            ret, self.frame = self.cap.read()
            if not ret:
                logger.info("End of video")
                self.done = True
                return
            self.read_latencies.append(time.perf_counter() - t0)

            # --- Step 2: Color convert ---
            t0 = time.perf_counter()
            self.frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
            self.cvt_latencies.append(time.perf_counter() - t0)

            # --- Step 3: Store put ---
            t0 = time.perf_counter()
            data_id = self.client.put(self.frame)
            self.store_put_latencies.append(time.perf_counter() - t0)

            # --- Step 4: Queue put ---
            # Pass frame_num so downstream actors can correlate events
            t0 = time.perf_counter()
            try:
                self.q_out.put([data_id, frame_start, self.frame_num])
            except Exception as e:
                logger.error(f"Generator Exception: {e}")
            self.queue_put_latencies.append(time.perf_counter() - t0)

            self.timestamps.append(frame_start)
            self.frame_num += 1
            self.work_latencies.append(time.perf_counter() - perf_start)

            time.sleep(self.frame_interval)
            self.full_latencies.append(time.perf_counter() - perf_start)