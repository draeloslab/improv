import os
import time
import threading
import yaml
import numpy as np
from improv.actor import ManagedActor
from . import cpu_affinity
from pathlib import Path
from queue import Queue, Empty
from .video_converter import VideoConverter

from .run_paths import get_logger, run_folder, video_session_folder

logger = get_logger(__name__, "camera_video_saver.log")

#: A camera silent this long is reported once (and again when it comes back): a USB camera that drops out
#: mid-run otherwise just stops producing frames without any error anywhere.
SILENCE_WARNING_S = 2.0


class VideoSaver(ManagedActor):
    """Streams one camera's frames to buffer_NNNNN.bin files (buffer_length s each, length-prefixed raw
    frames) for VideoConverter to turn into an mp4 afterwards.

    Each frame is fetched from the store as soon as its id arrives and handed to a writer thread. The saver
    used to hold buffer_length s of ids and then fetch them all at once, which every 5 s hit Redis with
    ~150 x 1.5 MB reads per camera -- Redis is single-threaded, so the cameras' writes queued behind them
    (the frame-acquisition spikes every 5 s in run 20260917-1557).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.camera_num = kwargs['camera_num']

    # ------------------------------------------------------------------ threads
    def read_frames_process(self):
        """q_in -> store get -> writer queue, one frame at a time."""
        last_frame, silent = time.time(), False
        while not self.stop_program:
            try:
                msg = self.q_in.get(timeout=0.5)
            except Empty:
                if not silent and time.time() - last_frame > SILENCE_WARNING_S:
                    silent = True
                    logger.warning(f"[Camera {self.camera_name}] no frames for {SILENCE_WARNING_S:.0f} s "
                                   f"(camera disconnected or stalled?)")
                continue
            except Exception as e:
                logger.error(f"[Camera {self.camera_name}] reading q_in: {e}")
                continue
            self.start_times.append(time.time())
            t0 = time.perf_counter()
            if silent:
                logger.warning(f"[Camera {self.camera_name}] frames again after {time.time() - last_frame:.1f} s")
                silent = False
            last_frame = time.time()
            frame_id = msg[0] if msg else None
            if frame_id is None:
                continue
            try:
                frame = self.client.get(frame_id)
            except Exception as e:      # one lost frame (e.g. expired from the store) must not end the recording
                self.frames_lost += 1
                logger.error(f"[Camera {self.camera_name}] frame {self.total_frames} not saved: {e}")
                continue
            self.write_queue.put(frame)
            self.total_frames += 1
            self.frame_count += 1
            if self.frame_count % 900 == 0:
                logger.info(f"[Camera {self.camera_name}] writer FPS: "
                            f"{round(self.frame_count / (time.perf_counter() - self.time_start), 2)}")
                self.frame_count, self.time_start = 0, time.perf_counter()
            self.latencies.append(time.perf_counter() - t0)

    def write_frames_process(self):
        """Writer queue -> buffer_NNNNN.bin, a new file every buffer_length s of frames. None ends it."""
        num_buf_frames = self.fps * self.buffer_length
        f, n_in_file, num_buffer = None, 0, 0
        try:
            while True:
                frame = self.write_queue.get()
                if frame is None:
                    break
                if f is None:
                    f = open(os.path.join(self.out_folder_buffer, f'buffer_{num_buffer:05d}.bin'), 'wb')
                data = frame.tobytes()
                f.write(len(data).to_bytes(4, byteorder='little'))
                f.write(data)
                n_in_file += 1
                if n_in_file == num_buf_frames:
                    f.close()
                    f, n_in_file, num_buffer = None, 0, num_buffer + 1
        except Exception as e:
            logger.error(f"[Camera {self.camera_name}] writing buffer {num_buffer}: {e}")
        finally:
            if f is not None:
                f.close()

    # ------------------------------------------------------------------ lifecycle
    def setup(self):
        # Disk writing is throughput work, not latency work -- keeping it (and its ffmpeg children, which
        # inherit this affinity) on the E-cores leaves the P-cores for the processor and camera readers.
        cpu_affinity.pin_actor(cpu_affinity.BACKGROUND, label=f"VideoSaver cam{self.camera_num}")

        self._getStoreInterface()

        source_folder = Path(__file__).resolve().parent.parent
        with open(f'{source_folder}/config/camera_config.yaml', 'r') as file:
            camera_config = yaml.safe_load(file)

        camera_params = camera_config['camera_params']
        # Frames in the store are at stream_resolution (TIS scales in GStreamer); falls back to `resolution`.
        stream_res = camera_params.get('stream_resolution', camera_params['resolution'])
        self.frame_w = stream_res['width']
        self.frame_h = stream_res['height']
        self.fps = int(camera_params['fps'].split('/')[0])

        camera_config = camera_config['active_cameras'][self.camera_num]['camera']
        self.camera_name = camera_config['name']

        with open(f'{source_folder}/config/video_config.yaml', 'r') as file:
            video_config = yaml.safe_load(file)
        self.buffer_length = video_config['buffer_length']
        self.compression_quality = video_config['compression_quality']

        # One session folder for all of this run's savers, from the shared run id (see run_paths).
        session_folder = video_session_folder(video_config['raw_chunks_path'])
        self.out_folder_buffer = f"{session_folder}/camera_{self.camera_num}/"
        self.out_folder_video = f"{session_folder}/"
        Path(self.out_folder_buffer).mkdir(parents=True, exist_ok=True)
        timestamp_hhmm = session_folder.name[:4]
        self.output_video = os.path.join(self.out_folder_video, f"camera_video_{self.camera_num+1}_{timestamp_hhmm}.mp4")

        self.latencies = []           # per frame: store get + hand-off to the writer (s)
        self.start_times = []         # per frame: wall time it was taken off q_in
        self.out_folder = run_folder()
        logger.info(f"Latency Output folder set to {self.out_folder}")

        self.video_converter = VideoConverter(
            compression_quality=self.compression_quality,
            video_params={'frame_w': self.frame_w, 'frame_h': self.frame_h, 'fps': self.fps},
            output_video=self.output_video,
            out_folder_buffer=self.out_folder_buffer,
            log_progress=True)

        self.stop_program = False
        self.start_program = False
        self.conversion_started = False
        self.total_frames = self.frame_count = self.frames_lost = 0
        self.time_start = time.perf_counter()

        self.write_queue = Queue()
        self.writer_thread = threading.Thread(target=self.write_frames_process, daemon=True)
        self.reader_thread = threading.Thread(target=self.read_frames_process)
        self.convert_saved_frames_proc = threading.Thread(target=self.convert_saved_frames)
        logger.info(f"[Camera {self.camera_name}] saver setup completed")

    def runStep(self):
        if not self.start_program:
            while not self.q_in.empty():          # drop frames queued before the run started
                self.q_in.get_nowait()
            self.start_program = True
            self.frame_count, self.time_start = 0, time.perf_counter()     # writer FPS counts from here, not from setup
            self.writer_thread.start()
            self.reader_thread.start()
            logger.info(f"[Camera {self.camera_name}] recording started")

    def stop(self):
        self.stop_program = True
        logger.info(f"[Camera {self.camera_name}] waiting for the saver threads to finish")
        if self.reader_thread.is_alive():
            self.reader_thread.join()
        self.write_queue.put(None)
        if self.writer_thread.is_alive():
            self.writer_thread.join()

        logger.info(f"[Camera {self.camera_name}] total frames saved: {self.total_frames}"
                    + (f", {self.frames_lost} lost (not in the store any more)" if self.frames_lost else ""))
        np.save(self.out_folder / f"saverlatencies_cam_{self.camera_num}.npy", self.latencies)
        np.save(self.out_folder / f"saverstarts_cam_{self.camera_num}.npy", self.start_times)
        logger.info(f"[Camera {self.camera_name}] Latencies saved to {self.out_folder}")

        self.wait_conversion_process()
        if self.conversion_started:
            self.convert_saved_frames_proc.join()
            logger.info(f"[Camera {self.camera_name}] video conversion completed")

    def wait_conversion_process(self):
        """Wait for the GUI's video_conversion message on msg_in. Graphs that don't wire msg_in (the 3D
        graphs) convert offline, so return at once -- this used to loop forever there and every quit had to
        kill the saver."""
        link = self.links.get("msg_in")
        if link is None:
            logger.info(f"[Camera {self.camera_name}] no msg_in link: buffers left for offline conversion")
            return
        while not self.conversion_started:
            try:
                msg = link.get(timeout=0.5)
            except KeyboardInterrupt:
                return
            except Exception:
                msg = None
            if msg is not None:
                logger.info(f"[Camera {self.camera_name}] received message: {msg}")
                if msg.get('type') == 'video_conversion':
                    if not msg.get('value'):
                        return
                    self.conversion_started = True
                    logger.info(f"[Camera {self.camera_name}] video conversion started")
                    self.convert_saved_frames_proc.start()
            time.sleep(0.5)

    def convert_saved_frames(self):
        self.video_converter.convert_saved_frames()
