import os
# Limit numpy/BLAS threading to avoid contention with PyTorch in multiprocessing
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import numpy as np
import logging
import yaml
import time
import traceback
import cv2
import torch
from pathlib import Path
from improv.actor import Actor
from collections import deque
from .jarvis_infer import build_cfg, load_predictor, run_inference
from .kalmanfilter import KalmanFilterPredictor
from . import cpu_affinity
from improv.store import ObjectNotFoundError

from .run_paths import get_logger, run_folder

logger = get_logger(__name__, "processor_jarvis.log")

# Bodypart order the fisk_freebie fine-tune was trained on (see
# 02_jarvis_finetune.ipynb / convert_dlc_to_jarvis.py, dataset
# dataset_fisk_freebie, run finetune_monkeyhand). This is also the channel
# order of the fine-tuned KeypointDetect heatmap head, so it must not be
# reordered without retraining.
#
# Sanity-checked directly against the saved checkpoint
# (KeypointDetect/finetune_monkeyhand/EfficientTrack-medium_final.pth):
# final_conv1.weight is [11, 88, 3, 3] and deconv1.weight is [88, 11, 4, 4] --
# 11 output channels, not JARVIS's native 23-joint MonkeyHand skeleton. The
# fine-tune only had 11-joint DLC labels to train against (see
# load_pose_pretrain in efficienttrack.py, which drops the pretrained
# 23-channel head and reinits it at 11 whenever NUM_JOINTS doesn't match), so
# this model physically cannot emit the other 12 MonkeyHand joints -- there is
# no "also output all 23" available from this checkpoint. Front-end plotting
# is sized for exactly these 11 points; see front_end5.py.
BODYPARTS = ["Wrist", "Thumb_Tip", "Thumb_IP", "Index_Tip", "Index_MCP",
             "Middle_Tip", "Middle_MCP", "Ring_Tip", "Ring_MCP", "Small_Tip", "Small_MCP"]

# Joint used for the same "single scalar proxy for finger flexion" trick
# processor.py does (`angle = smoothed_prediction[1][1]`, the PIP y-value, in
# the old 4-point DIP/PIP/MCP/Wrist model). There's no PIP in this 11-point
# skeleton, so this picks Index_MCP's y-coordinate as the closest equivalent
# (same joint *type* the old code used -- MCP -- just on a named finger now
# that the model tracks the whole hand instead of one digit). Point this at a
# different BODYPARTS entry if index flexion isn't the signal FES control
# actually wants.
ANGLE_PROXY_INDEX = BODYPARTS.index("Index_MCP")

# Sentinel written into self.prediction's x/y/score when CenterDetect finds no
# hand this frame (JARVIS's equivalent of DLC's -1/-2 "assembly failed"
# codes). Deliberately <= -1.5 so it's caught by both the existing
# `angle > -1.5` guard below (excluded from the smoothing window) and
# front_end5's `x < -1.5 or y < -1.5` skip-drawing check -- same convention
# the DLC path already relies on, not a new one.
NO_DETECTION_SENTINEL = -2.0


class ProcessorJarvis(Actor):
    """Applying a fine-tuned JARVIS-MoCap EfficientTrack model (CenterDetect +
    KeypointDetect) to each video frame, in place of DLC.

    Alternate version of processor.py: swaps DLC's single-stage pose model for
    JARVIS's 2-stage EfficientTrack pipeline, fine-tuned on the same 11
    fisk_freebie hand keypoints (see 02_jarvis_finetune.ipynb). Everything
    downstream (Kalman filter, angle proxy, queue message shape, GUI) is
    unchanged from processor.py other than the bodypart count/order and the
    inference call itself.

    Environment note: this needs the `jarvis-mocap` package importable in
    whatever conda env runs this actor (see jarvis_infer.py's docstring) --
    it is not installed in the same env as processor.py's DLC path today.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.pred_active = kwargs['pred_active']
        self.camera_num = kwargs.get('camera_num', 0)  # Default to camera 0 if not specified
        self.out_folder = None  # only set once setup() finishes; guards stop() if setup() raises

    def setup(self):
        """Initializes all class variables."""
        # Claim a dedicated physical P-core BEFORE anything else -- see
        # processor.py's setup() for why this has to happen ahead of model
        # loading (CUDA driver helper threads inherit affinity from whichever
        # thread spins them up).
        cpu_affinity.pin_actor(cpu_affinity.COMPUTE, slot=self.camera_num,
                               label=f"ProcessorJarvis cam{self.camera_num}")

        self._getStoreInterface()

        if self.pred_active:
            logger.info("Beginning setup for ProcessorJarvis")

            # Limit PyTorch threading to avoid contention with numpy in multiprocessing
            torch.set_num_threads(1)

            # load the configuration file
            source_folder = Path(__file__).resolve().parent.parent

            with open(f'{source_folder}/config.yaml', 'r') as file:
                config = yaml.safe_load(file)

            self.resize = config['resize']

            # Select fine-tuned JARVIS weights based on camera number. Unlike
            # the DLC path's per-camera single-finger models, this is the same
            # whole-hand model for every camera -- config.yaml just repeats
            # the same two paths per camera_num for symmetry with the DLC keys.
            center_weights = Path(config[f'jarvis_center_weights_{self.camera_num}'])
            keypoint_weights = Path(config[f'jarvis_keypoint_weights_{self.camera_num}'])

            model_size = config.get('jarvis_model_size', 'medium')
            bbox_size = config.get('jarvis_bbox_size', 512)
            self.center_threshold = config.get('jarvis_center_threshold', 40)

            self.cfg = build_cfg(model_size=model_size, bbox_size=bbox_size,
                                  num_joints=len(BODYPARTS))

            logger.info(f"Camera {self.camera_num}: Using JARVIS CenterDetect weights {center_weights}")
            logger.info(f"Camera {self.camera_num}: Using JARVIS KeypointDetect weights {keypoint_weights}")

            self.centerDetect, self.keypointDetect = load_predictor(
                self.cfg, str(center_weights), str(keypoint_weights))

            # Initializing Kalman Filter with smoother parameters
            self.kalman_filter = KalmanFilterPredictor(
                adapt=True,
                forward=0.002,
                fps=30,
                nderiv=2,
                priors=[1, 1],  #[1e5, 1e5]
                initial_var=10,
                process_var=1,
                dlc_var=10,
                lik_thresh=0.6
            )
            logger.info(f'Kalman filter initialized for camera {self.camera_num}')

            self.resize = config['resize']
            # See processor.py's setup() -- same rule: only cv2.resize when the
            # frame source hasn't already downscaled it (TIS via GStreamer).
            self.camera_prescaled = config.get('camera_prescaled', False)
            self.name = "ProcessorJarvis"
            self.frame = None
            self.predictions = []
            self.raw_predictions = []
            self.frame_num = 0
            self.frame_sentTime = 0
            self.frames_log = 200 # num frames after which to log
            self.angle_queue = deque(maxlen=15)  # to store last 10 angles for smoothing
            self.alpha = config['alpha']
            self.interp_thresh = config['threshold']
            self.prev_angle = None
            self.smoothed_prediction = None

            # --- Timing logs ---
            # Same names/files as processor.py so the existing latency
            # notebooks (notebooks/*latenc*.ipynb) keep working unmodified --
            # `dlc_latencies` / `proc_dlc_cam*.npy` now time JARVIS's 2-stage
            # inference call instead of DLC's single-stage one, not a rename.
            self.timestamps = []
            self.frame_nums_received = []
            self.generator_timestamps = []
            self.queue_wait_latencies = []   # time spent waiting on q_in.get
            self.store_get_latencies = []    # client.get
            self.resize_latencies = []       # cv2.resize
            self.dlc_latencies = []          # JARVIS inference only (name kept for compatibility)
            self.kalman_latencies = []       # Kalman filter
            self.postprocess_latencies = []  # angle calc + smoothing
            self.queue_put_latencies = []    # q_out.put
            self.total_latencies = []        # queue get → q_out put (full runStep work)
            self.angles_sent = []
            self.frames_dropped = []         # number of stale frames skipped per runStep

            self.time_start = time.perf_counter()


            self.out_folder = run_folder()
            logger.info(f"Output folder set to {self.out_folder}")
            logger.info(f"Using alpha: {self.alpha} and interp_thresh: {self.interp_thresh} and resize: {self.resize}")
            logger.info("Completed setup for ProcessorJarvis")

    def stop(self):
        """Stop function for saving results and cleaning up."""
        if self.pred_active:
            self.done = True
            cam = self.camera_num

            if self.out_folder is None:
                # setup() raised before finishing (e.g. model weights failed to
                # load) -- nothing was initialized, so there's nothing to save.
                logger.error(f"ProcessorJarvis cam{cam}: stop() called but setup() "
                              "never completed (out_folder unset) -- skipping save")
                logger.info(f"Processor {self.name} stopped")
                return

            # Predictions and raw data
            np.save(self.out_folder / f"predictions_cam{cam}.npy", self.predictions)
            np.save(self.out_folder / f"raw_predictions_cam{cam}.npy", self.raw_predictions)
            np.save(self.out_folder / f"angles_sent_cam{cam}.npy", self.angles_sent)

            # Timestamps for cross-actor correlation
            np.save(self.out_folder / f"proc_timestamps_cam{cam}.npy", self.timestamps)
            np.save(self.out_folder / f"proc_frame_nums_cam{cam}.npy", self.frame_nums_received)
            np.save(self.out_folder / f"proc_gen_timestamps_cam{cam}.npy", self.generator_timestamps)

            # Per-step latency breakdowns
            np.save(self.out_folder / f"proc_queue_wait_cam{cam}.npy", self.queue_wait_latencies)
            np.save(self.out_folder / f"proc_store_get_cam{cam}.npy", self.store_get_latencies)
            np.save(self.out_folder / f"proc_resize_cam{cam}.npy", self.resize_latencies)
            np.save(self.out_folder / f"proc_dlc_cam{cam}.npy", self.dlc_latencies)
            np.save(self.out_folder / f"proc_kalman_cam{cam}.npy", self.kalman_latencies)
            np.save(self.out_folder / f"proc_postprocess_cam{cam}.npy", self.postprocess_latencies)
            np.save(self.out_folder / f"proc_queue_put_cam{cam}.npy", self.queue_put_latencies)
            np.save(self.out_folder / f"proc_total_cam{cam}.npy", self.total_latencies)

            # Queue drain tracking
            np.save(self.out_folder / f"proc_frames_dropped_cam{cam}.npy", self.frames_dropped)

            # Legacy file names for backward compatibility
            np.save(self.out_folder / f"latencies_cam{cam}.npy", self.total_latencies)
            np.save(self.out_folder / f"startLatencies_cam{cam}.npy", self.timestamps)
            np.save(self.out_folder / f"dlcLatencies_cam{cam}.npy", self.dlc_latencies)

            logger.info(f"ProcessorJarvis cam{cam}: Predictions and latencies saved to {self.out_folder}")
        logger.info(f"Processor {self.name} stopped")

    def runStep(self):
        frame_id = None
        self.prediction = None
        angle = None
        smoothed_prediction = None
        smoothed_angle = None  # Initialize smoothed_angle

        if self.pred_active:
            # --- Step 1: Queue drain — skip to the NEWEST frame ---
            t_queue_wait = time.perf_counter()
            try:
                msg = self.q_in.get(timeout=0.01)
                # ----------------------------
                # Drain any additional frames — keep only the newest
                dropped = 0
                while True:
                    try:
                        newer_msg = self.q_in.get_nowait()
                        dropped += 1
                        msg = newer_msg
                    except Exception:
                        break  # queue is empty, msg holds the newest frame

                if dropped > 0:
                    logger.debug(f"Cam {self.camera_num}: Skipped {dropped} stale frame(s) to reduce latency")
                #------------------------------

                # Support both old [data_id, timestamp] and new [data_id, timestamp, frame_num] formats
                if len(msg) == 3:
                    frame_id, camera_start, gen_frame_num = msg
                else:
                    frame_id, camera_start = msg
                    gen_frame_num = -1
                t_got = time.perf_counter()
                self.queue_wait_latencies.append(t_got - t_queue_wait)
                self.frames_dropped.append(dropped)
            except Exception as e:
                # No frame available, nothing to process
                return

            if frame_id is not None:
                self.done = False
                step_start = time.perf_counter()
                now = time.time()
                self.timestamps.append(now)
                self.frame_nums_received.append(gen_frame_num)
                self.generator_timestamps.append(camera_start)

                try:
                    # --- Step 2: Store get ---
                    t0 = time.perf_counter()
                    frame = self.client.get(frame_id)
                    if isinstance(frame, np.ndarray) and len(frame.shape) == 3:
                        pass
                    else:
                        # uncompressing the frame
                        frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)
                    self.store_get_latencies.append(time.perf_counter() - t0)

                    self.frame_num += 1

                    # --- Step 3: Resize ---
                    # Skipped when the frame source already delivers pre-scaled
                    # frames (TIS + camera_prescaled: true in config.yaml).
                    t0 = time.perf_counter()
                    if not self.camera_prescaled:
                        frame = cv2.resize(frame, (int(frame.shape[1] * self.resize), int(frame.shape[0] * self.resize)))
                    self.resize_latencies.append(time.perf_counter() - t0)

                    # --- Step 4: JARVIS inference (CenterDetect + KeypointDetect) ---
                    t0 = time.perf_counter()
                    try:
                        pts, conf, _center = run_inference(
                            self.centerDetect, self.keypointDetect, self.cfg, frame,
                            center_threshold=self.center_threshold)
                    except Exception as e:
                        logger.error(f"JARVIS inference error: {e}")
                        logger.error(traceback.format_exc())
                        # Use previous prediction as fallback if available
                        if self.smoothed_prediction is not None:
                            smoothed_prediction = self.smoothed_prediction
                            smoothed_angle = self.prev_angle
                        self.dlc_latencies.append(time.perf_counter() - t0)
                        self.total_latencies.append(time.perf_counter() - step_start)
                        self.angles_sent.append(smoothed_angle if smoothed_angle is not None else np.nan)
                        # camera_num is appended so aggregating downstream actors (Sender,
                        # VideoScreen) that fan multiple Processor instances into one
                        # actor can identify which physical camera a message came from.
                        self.q_out.put([smoothed_prediction, smoothed_angle, camera_start, gen_frame_num, self.camera_num])
                        logger.info(f"ProcessorJarvis camera {self.camera_num}: Sent fallback prediction and angle")
                        return
                    self.dlc_latencies.append(time.perf_counter() - t0)

                    if pts is None:
                        # CenterDetect found no hand this frame -- JARVIS's
                        # equivalent of DLC's assembly failure. Same sentinel
                        # convention as the DLC path (see NO_DETECTION_SENTINEL).
                        self.prediction = np.full((len(BODYPARTS), 3), NO_DETECTION_SENTINEL)
                    else:
                        self.prediction = np.concatenate([pts, conf[:, None]], axis=1)  # (11, 3): x, y, score
                    logger.debug(f"Camera {self.camera_num}: Prediction shape after extraction: {self.prediction}")

                    self.raw_predictions.append(self.prediction.copy())

                    # --- Step 5: Kalman filter ---
                    t0 = time.perf_counter()
                    try:
                        assert False
                        smoothed_prediction = self.kalman_filter.process(self.prediction, frame_time=camera_start)
                    except Exception as e:
                        smoothed_prediction = self.prediction  # fallback to raw prediction on error
                    self.kalman_latencies.append(time.perf_counter() - t0)

                    # Save prediction for analysis
                    self.predictions.append(smoothed_prediction)
                    self.smoothed_prediction = smoothed_prediction
                    logger.debug(f"Camera {self.camera_num}: Smoothed prediction shape: {smoothed_prediction.shape}")
                    # --- Step 6: Post-processing (angle calculation + smoothing) ---
                    t0 = time.perf_counter()
                    # Same "one scalar proxy, not a real geometric angle" trick
                    # as processor.py -- see ANGLE_PROXY_INDEX above for why
                    # Index_MCP's y-value is the stand-in here.
                    angle = smoothed_prediction[ANGLE_PROXY_INDEX][1]

                    # Only feed real numbers into the smoothing window -- see
                    # NO_DETECTION_SENTINEL, which is deliberately <= -1.5 so
                    # it's excluded here exactly like DLC's -1/-2 sentinels.
                    if np.isfinite(angle) and angle > -1.5:
                        self.angle_queue.append(angle)

                    if len(self.angle_queue) > 0:
                        smoothed_angle = float(np.mean(self.angle_queue))
                    else:
                        # Nothing valid seen yet this run -- hold the last good angle
                        # rather than emitting NaN downstream to the sender/GUI.
                        smoothed_angle = self.prev_angle

                    # Apply sudden jump detection on the smoothed angle
                    if (smoothed_angle is not None and self.prev_angle is not None
                            and np.abs(smoothed_angle - self.prev_angle) > 500):
                        smoothed_angle = self.prev_angle  # ignore sudden large jumps
                    self.prev_angle = smoothed_angle
                    self.postprocess_latencies.append(time.perf_counter() - t0)

                    if self.frame_num % self.frames_log == 0:
                        dlc_end = time.perf_counter()
                        total_time = dlc_end - self.time_start
                        logger.info(f"Frame number: {self.frame_num}")
                        logger.info(f"Overall Average FPS: {round(self.frames_log / total_time,2)}")
                        self.time_start = time.perf_counter()

                except ObjectNotFoundError:
                    logger.error("ProcessorJarvis: Frame unavailable from store, dropping")
                    return
                except Exception as e:
                    logger.error(f"Processing error: {e}")
                    logger.error(traceback.format_exc())
                    return

                # --- Step 7: Queue put (send to downstream) ---
                t0 = time.perf_counter()
                try:
                    logger.debug(f"Camera {self.camera_num}: Sending smoothed prediction and angle to downstream")
                    if smoothed_angle is not None:
                        smoothed_angle = smoothed_angle/self.resize
                    self.q_out.put([smoothed_prediction, smoothed_angle, camera_start, gen_frame_num, self.camera_num])
                except Exception as e:
                    logger.error(f"ProcessorJarvis Exception: {e}")
                    logger.error(traceback.format_exc())
                self.queue_put_latencies.append(time.perf_counter() - t0)

                self.total_latencies.append(time.perf_counter() - step_start)
                self.angles_sent.append(smoothed_angle if smoothed_angle is not None else np.nan)

        else:
            pass

    def calculateAngle(self, prediction, finger="Index"):
        """Geometric flexion angle at <finger>_MCP, using Wrist-MCP-Tip. Not
        called from runStep (processor.py's equivalent was already dead code
        behind a commented-out call, kept here updated to the 11-point
        skeleton in case it's wired back in later)."""
        tip = BODYPARTS.index(f"{finger}_Tip")
        mcp = BODYPARTS.index(f"{finger}_MCP")
        wrist = BODYPARTS.index("Wrist")

        p_tip, p_mcp, p_wrist = prediction[tip, :2], prediction[mcp, :2], prediction[wrist, :2]
        v_mcp_to_tip = p_tip - p_mcp
        v_mcp_to_wrist = p_wrist - p_mcp

        dot_product = np.dot(v_mcp_to_tip, v_mcp_to_wrist)
        determinant = v_mcp_to_tip[0] * v_mcp_to_wrist[1] - v_mcp_to_tip[1] * v_mcp_to_wrist[0]

        angle = np.degrees(np.arctan2(determinant, dot_product)) % 360
        return angle
