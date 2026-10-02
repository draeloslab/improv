"""Generator: plays a recorded video into the store as if it were a live camera (replay graphs, no hardware)."""
import time
from pathlib import Path

import cv2
import numpy as np
import yaml
from improv.actor import Actor

from . import cpu_affinity
from .run_paths import get_logger, run_folder

logger = get_logger(__name__, "generator.log")


class Generator(Actor):
    """Reads a video at its own frame rate (config `fps` if it has none), converts BGR -> RGB and sends [store id, time, frame_num] on q_out.

    Graph kwargs: camera_num (plays config.yaml video_paths[camera_num]) or video_path (any file). Without either
    it plays config.yaml video_path.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.camera_num = kwargs.get('camera_num')
        self.video_path_kw = kwargs.get('video_path')     # graph-level override of config.yaml's video_paths

    def setup(self):
        """Pick the video, open it and set up the timing logs."""
        with open(Path(__file__).resolve().parent.parent / 'config' / 'config.yaml') as f:
            config = yaml.safe_load(f)
        if self.video_path_kw:
            self.video_path = self.video_path_kw
        elif self.camera_num is not None:
            self.video_path = config['video_paths'][self.camera_num]
        else:
            self.video_path = config['video_path']

        # Replay actor: keep it off the processor's P-cores (and off the excluded cores).
        cpu_affinity.pin_actor(cpu_affinity.BACKGROUND, label=f"Generator {self.name}")
        self.cap = None
        self.frame_interval = 1.0 / config['fps']     # replaced by the video's own rate below when it has one
        self.next_due = None
        self.frame_num = 0

        # Timing logs (seconds)
        self.timestamps = []        # time.time() when each frame was produced
        self.work_latencies = []    # read + cvtColor + store put + q_out put
        self.full_latencies = []    # the same plus the pacing sleep
        self.read_latencies = []    # cv2 read
        self.cvt_latencies = []     # cvtColor
        self.store_put_latencies = []  # client.put
        self.queue_put_latencies = []  # q_out.put

        self.done = True    # stays True unless the video opens below
        self.out_folder = run_folder()
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            logger.error(f"Error opening video file: {self.video_path}")
            return
        video_fps = self.cap.get(cv2.CAP_PROP_FPS)
        if video_fps and 1 <= video_fps <= 500:
            self.frame_interval = 1.0 / video_fps     # play at the recording's speed, whatever config `fps` says
        logger.info(f"{self.name}: {self.video_path}, {int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))} frames "
                    f"at {1 / self.frame_interval:.1f} fps")
        self.done = False

    def stop(self):
        """Release the video and save the timing logs (gen_*.npy)."""
        if self.cap:
            self.cap.release()
        
        np.save(self.out_folder / "gen_timestamps.npy", self.timestamps)
        np.save(self.out_folder / "gen_work_latencies.npy", self.work_latencies)
        np.save(self.out_folder / "gen_full_latencies.npy", self.full_latencies)
        np.save(self.out_folder / "gen_read_latencies.npy", self.read_latencies)
        np.save(self.out_folder / "gen_cvt_latencies.npy", self.cvt_latencies)
        np.save(self.out_folder / "gen_store_put_latencies.npy", self.store_put_latencies)
        np.save(self.out_folder / "gen_queue_put_latencies.npy", self.queue_put_latencies)
        return 0

    def runStep(self):
        """Send the next frame, then sleep until it is due (paced against a deadline, not a fixed sleep)."""
        if self.done:
            return

        if self.cap and self.cap.isOpened():
            frame_start = time.time()
            perf_start = time.perf_counter()

            # 1. read
            t0 = time.perf_counter()
            ret, self.frame = self.cap.read()
            if not ret:
                logger.info("End of video")
                self.done = True
                return
            self.read_latencies.append(time.perf_counter() - t0)

            # 2. BGR -> RGB (the store holds RGB, like the cameras)
            t0 = time.perf_counter()
            self.frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
            self.cvt_latencies.append(time.perf_counter() - t0)

            # 3. store put
            t0 = time.perf_counter()
            data_id = self.client.put(self.frame)
            self.store_put_latencies.append(time.perf_counter() - t0)

            # 4. announce it (frame_num lets align_frames match the same instant across cameras)
            t0 = time.perf_counter()
            try:
                self.q_out.put([data_id, frame_start, self.frame_num])
            except Exception as e:
                logger.error(f"Generator Exception: {e}")
            self.queue_put_latencies.append(time.perf_counter() - t0)

            self.timestamps.append(frame_start)
            self.frame_num += 1
            self.work_latencies.append(time.perf_counter() - perf_start)

            # A fixed sleep on top of the work played at 27.5 fps instead of 30.
            now = time.perf_counter()
            self.next_due = (now if self.next_due is None else self.next_due) + self.frame_interval
            if self.next_due > now:
                time.sleep(self.next_due - now)
            else:
                self.next_due = now          # fell behind: don't try to catch up in a burst
            self.full_latencies.append(time.perf_counter() - perf_start)