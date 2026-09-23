"""Minimal 2-camera batched processor: both cameras' frames go through ONE
model in a SINGLE batched forward pass (batch_size=2), instead of two
independent actors.processor.Processor actors each submitting to the GPU on
their own (see processor_batch3d.py's docstring for why that contention is
bad). No triangulation, no 3D -- each camera keeps its own 2D keypoints and
angle and is put out on its own named link (q_out0/q_out1) in the exact same
[prediction, angle, camera_start, frame_num, camera_num] shape
actors.processor.Processor uses, so VideoScreen/Sender need no changes.

Uses the model configured for camera 0 (model_path_0/model_snapshot_0) for
both cameras -- a placeholder until a 2-camera-specific model exists.
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import time
import traceback
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml

from improv.actor import Actor
from improv.store import ObjectNotFoundError

from deeplabcut.pose_estimation_pytorch.config import read_config_as_dict
from deeplabcut.pose_estimation_pytorch.apis.utils import get_inference_runners

from .kalmanfilter import KalmanFilterPredictor
from . import cpu_affinity
from .run_paths import get_logger, run_folder

logger = get_logger(__name__, "processor_batch2.log")


class ProcessorBatch2(Actor):
    """Two cameras, one model, one batched forward pass, no triangulation."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pred_active = kwargs.get('pred_active', True)
        self.camera_nums = kwargs.get('camera_nums', [0, 1])

    def setup(self):
        cpu_affinity.pin_actor(cpu_affinity.COMPUTE, slot=0, label="ProcessorBatch2")
        self._getStoreInterface()

        if not self.pred_active:
            logger.info("ProcessorBatch2: pred_active=False, running as a no-op")
            return

        source_folder = Path(__file__).resolve().parent.parent
        with open(f'{source_folder}/config/config.yaml', 'r') as file:
            config = yaml.safe_load(file)

        self.resize = config['resize']
        self.camera_prescaled = config.get('camera_prescaled', False)

        # Single model for both cameras -- camera 0's, per the task.
        train_dir = Path(config['model_path_0'])
        snapshot_path = train_dir / config['model_snapshot_0']
        model_cfg = read_config_as_dict(train_dir / "pytorch_config.yaml")

        # Same PAF -> HeatmapPredictor override as processor.py -- these are
        # single-animal models, PAF's assembly is what blanks keypoints to NaN.
        if config.get('single_animal_predictor', True):
            try:
                head_cfg = model_cfg['model']['heads']['bodypart']
                old_pred = head_cfg.get('predictor', {})
                if old_pred.get('type') != 'HeatmapPredictor':
                    locref_std = old_pred.get('locref_stdev', old_pred.get('locref_std', 7.2801))
                    head_cfg['predictor'] = {
                        'type': 'HeatmapPredictor',
                        'apply_sigmoid': old_pred.get('apply_sigmoid', True),
                        'clip_scores': old_pred.get('clip_scores', False),
                        'location_refinement': True,
                        'locref_std': locref_std,
                    }
            except (KeyError, TypeError) as e:
                logger.warning(f"could not override predictor ({e}); keeping the model's own")

        torch.set_num_threads(1)
        # batch_size=2: exactly the point -- one forward pass carries both
        # cameras' frames instead of two separate calls.
        self.pose_runner, _ = get_inference_runners(
            model_config=model_cfg,
            snapshot_path=snapshot_path,
            max_individuals=1,
            batch_size=2,
            detector_batch_size=1,
            detector_path=None,
        )
        logger.info(f"ProcessorBatch2: model {train_dir.name} / {config['model_snapshot_0']} "
                    f"for cameras {self.camera_nums}, batch_size=2")

        self.kalman_filters = {
            cam: KalmanFilterPredictor(adapt=True, forward=0.002, fps=30, nderiv=2,
                                        priors=[1, 1], initial_var=10, process_var=1,
                                        dlc_var=10, lik_thresh=0.6)
            for cam in self.camera_nums
        }
        self.prev_angle = {cam: None for cam in self.camera_nums}

        self.frame_num = 0
        self.frames_log = 200
        self.time_start = time.perf_counter()
        self.predictions_log = {cam: [] for cam in self.camera_nums}
        self.angles_log = {cam: [] for cam in self.camera_nums}
        self.timestamps = []
        self.inference_latencies = []

        self.out_folder = run_folder()
        logger.info(f"Output folder set to {self.out_folder}")
        logger.info("Completed setup for ProcessorBatch2")

    def _gather_frames(self):
        """Newest frame id per camera slot (0/1), draining stale ones."""
        frame_ids = [None, None]
        starts = [None, None]
        fnums = [-1, -1]
        for slot in range(2):
            link = self.links.get(f"frames{slot}_in")
            if link is None:
                continue
            msg = None
            try:
                msg = link.get(timeout=0.005)
            except Exception:
                continue
            while True:
                try:
                    msg = link.get_nowait()
                except Exception:
                    break
            if len(msg) == 3:
                frame_ids[slot], starts[slot], fnums[slot] = msg
            elif len(msg) == 2:
                frame_ids[slot], starts[slot] = msg
        return frame_ids, starts, fnums

    def runStep(self):
        if not self.pred_active:
            return

        frame_ids, starts, fnums = self._gather_frames()
        present = [s for s in range(2) if frame_ids[s] is not None]
        if not present:
            return

        frames = {}
        for slot in present:
            try:
                frame = self.client.get(frame_ids[slot])
                if not (isinstance(frame, np.ndarray) and frame.ndim == 3):
                    frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)
                if not self.camera_prescaled:
                    frame = cv2.resize(
                        frame, (int(frame.shape[1] * self.resize), int(frame.shape[0] * self.resize)))
                frames[slot] = frame
            except ObjectNotFoundError:
                logger.debug(f"slot {slot}: frame gone from store, skipping")
            except Exception as e:
                logger.error(f"slot {slot}: store get failed: {e}")

        if not frames:
            return

        batch_slots = sorted(frames)
        self.frame_num += 1
        self.timestamps.append(time.time())

        # --- one batched forward pass through the one model, both cameras ---
        t0 = time.perf_counter()
        try:
            raw = self.pose_runner.inference([frames[s] for s in batch_slots])
        except Exception as e:
            logger.error(f"inference failed: {e}")
            logger.error(traceback.format_exc())
            return
        self.inference_latencies.append(time.perf_counter() - t0)

        for out_i, slot in enumerate(batch_slots):
            cam = self.camera_nums[slot]
            try:
                pred = np.asarray(raw[out_i]['bodyparts'][0], dtype=float)
            except Exception as e:
                logger.error(f"slot {slot} (cam {cam}): could not read prediction: {e}")
                continue
            # DLCRNet emits offset columns 3/4; heatmap-only models emit x,y,score.
            if pred.shape[1] >= 5:
                pred[:, 0] += pred[:, 3]
                pred[:, 1] += pred[:, 4]
            pred = pred[:, :3]

            try:
                smoothed_pred = self.kalman_filters[cam].process(pred, frame_time=starts[slot])
            except Exception:
                smoothed_pred = pred

            if len(smoothed_pred) >= 3:
                angle = smoothed_pred[1][1]
            else:
                angle = smoothed_pred[0][0]

            if np.isfinite(angle) and angle > -1.5:
                smoothed_angle = angle
            else:
                smoothed_angle = self.prev_angle[cam]
            self.prev_angle[cam] = smoothed_angle
            if smoothed_angle is not None:
                smoothed_angle = smoothed_angle / self.resize

            self.predictions_log[cam].append(smoothed_pred)
            self.angles_log[cam].append(smoothed_angle if smoothed_angle is not None else np.nan)

            out_link = self.links.get(f"q_out{slot}")
            if out_link is not None:
                out_link.put([smoothed_pred, smoothed_angle, starts[slot], fnums[slot], cam])

        if self.frame_num % self.frames_log == 0:
            elapsed = time.perf_counter() - self.time_start
            logger.info(f"frame {self.frame_num}: {round(self.frames_log / elapsed, 2)} fps, "
                        f"batched infer {np.median(self.inference_latencies[-self.frames_log:]) * 1000:.1f} ms")
            self.time_start = time.perf_counter()

    def stop(self):
        if not self.pred_active:
            logger.info("ProcessorBatch2 stopped (inactive)")
            return
        try:
            for cam in self.camera_nums:
                np.save(self.out_folder / f"predictions_cam{cam}.npy", np.asarray(self.predictions_log[cam]))
                np.save(self.out_folder / f"angles_sent_cam{cam}.npy", np.asarray(self.angles_log[cam]))
            np.save(self.out_folder / "batch2_timestamps.npy", np.asarray(self.timestamps))
            np.save(self.out_folder / "batch2_lat_inference.npy", np.asarray(self.inference_latencies))
            logger.info(f"ProcessorBatch2: saved {self.frame_num} frames to {self.out_folder}")
        except Exception:
            logger.error(f"could not save logs: {traceback.format_exc()}")
        logger.info("ProcessorBatch2 stopped")
