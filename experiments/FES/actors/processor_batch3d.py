"""Single batched processor: N cameras -> one pose-estimation step -> 3D -> joint angles.

Replaces the one-Processor-per-camera design (actors/processor.py). The
motivation is measured, not theoretical: four independent processes submitting
to the same GPU pile into a shared queue and each one's `inference()` call
blocks until the whole pile drains, so the earliest submitter reports the
*longest* time (corr(submission phase, measured inference) = -0.999 in run
20260804-1432). See MULTICAM_3D_PLAN.md Part 1. One process handling all N
cameras has no cross-process queue to wait on.

The pipeline per step is:

    frames{0..N-1}_in  ->  ONE pose-estimation step, all N cameras  (_infer)
                       ->  per-camera 2D keypoints + likelihood
                       ->  likelihood gate -> NaN
                       ->  aniposelib CameraGroup.triangulate  -> (K, 3) mm
                       ->  dlc2kinematics jointangle_calc      -> {joint: deg}
                       ->  q_out

`batch3d_backend` in config.yaml picks how that pose-estimation step works
(see _setup_mediapipe/_infer_mediapipe and _setup_dlc/_infer_dlc):

  - "mediapipe" (default): N HandLandmarker instances run concurrently on a
    thread pool. MediaPipe's Python Tasks API has no tensor-batch call, so this
    -- not a literal batched tensor -- is what "one step, N cameras" means for
    this backend; TFLite releases the GIL during inference so it still lands in
    one step's time budget (measured 32.6 ms for 4 cameras concurrently vs
    138 ms sequential).
  - "dlc": a real batched tensor forward pass through one PyTorch model
    (batch_size = num_cameras): measured 41.6 ms for 4 frames vs 75.6 ms for
    four sequential calls (1.8x).

Triangulation and joint angles both come from the libraries the rest of the lab
uses (aniposelib, dlc2kinematics) rather than reimplementations -- see
_triangulate() and _joint_angles() for the two places where their APIs needed
adapting to a per-frame realtime call.
"""

import os
# Limit numpy/BLAS threading to avoid contention with PyTorch in multiprocessing
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml

from improv.actor import Actor
from improv.store import ObjectNotFoundError

# Only needed for triangulation/joint angles. Tolerated as missing so a
# `triangulate: false` (2D-only) run works in an env without them; the actor
# raises in setup if 3D is requested and they are absent.
try:
    from aniposelib.cameras import CameraGroup
    from dlc2kinematics.utils import auxiliaryfunctions as d2k_aux
    _HAVE_3D_DEPS = True
except ImportError:
    CameraGroup = d2k_aux = None
    _HAVE_3D_DEPS = False

# torch/deeplabcut (the "dlc" backend) and mediapipe (the "mediapipe" backend)
# are imported lazily inside _setup_dlc/_setup_mediapipe -- a machine running
# only one backend doesn't need the other's dependencies installed.

from . import cpu_affinity
from .run_paths import get_logger, run_folder

logger = get_logger(__name__, "processor_batch3d.log")


# Joint angle definitions for the 21-keypoint MediaPipe hand layout
# (humanWristMP21). Each entry is (proximal, vertex, distal); the angle is
# measured AT the vertex. dlc2kinematics' convention is deviation from straight:
# a fully extended (collinear) joint reads 0 deg, a right angle reads 90 deg.
#
# Used only when the model's bodyparts match these names. Any other model falls
# back to _auto_joints(), so this actor is not hardcoded to one model.
MP21_JOINTS = {
    "THUMB_CMC":  ("WRIST", "THUMB_CMC", "THUMB_MCP"),
    "THUMB_MCP":  ("THUMB_CMC", "THUMB_MCP", "THUMB_IP"),
    "THUMB_IP":   ("THUMB_MCP", "THUMB_IP", "THUMB_TIP"),
    "INDEX_MCP":  ("WRIST", "INDEX_FINGER_MCP", "INDEX_FINGER_PIP"),
    "INDEX_PIP":  ("INDEX_FINGER_MCP", "INDEX_FINGER_PIP", "INDEX_FINGER_DIP"),
    "INDEX_DIP":  ("INDEX_FINGER_PIP", "INDEX_FINGER_DIP", "INDEX_FINGER_TIP"),
    "MIDDLE_MCP": ("WRIST", "MIDDLE_FINGER_MCP", "MIDDLE_FINGER_PIP"),
    "MIDDLE_PIP": ("MIDDLE_FINGER_MCP", "MIDDLE_FINGER_PIP", "MIDDLE_FINGER_DIP"),
    "MIDDLE_DIP": ("MIDDLE_FINGER_PIP", "MIDDLE_FINGER_DIP", "MIDDLE_FINGER_TIP"),
    "RING_MCP":   ("WRIST", "RING_FINGER_MCP", "RING_FINGER_PIP"),
    "RING_PIP":   ("RING_FINGER_MCP", "RING_FINGER_PIP", "RING_FINGER_DIP"),
    "RING_DIP":   ("RING_FINGER_PIP", "RING_FINGER_DIP", "RING_FINGER_TIP"),
    "PINKY_MCP":  ("WRIST", "PINKY_MCP", "PINKY_PIP"),
    "PINKY_PIP":  ("PINKY_MCP", "PINKY_PIP", "PINKY_DIP"),
    "PINKY_DIP":  ("PINKY_PIP", "PINKY_DIP", "PINKY_TIP"),
}

# Bones for the 3D skeleton the GUI draws. Same layout assumption as
# MP21_JOINTS; _auto_skeleton() covers everything else.
MP21_SKELETON = [
    ("WRIST", "THUMB_CMC"), ("THUMB_CMC", "THUMB_MCP"),
    ("THUMB_MCP", "THUMB_IP"), ("THUMB_IP", "THUMB_TIP"),
    ("WRIST", "INDEX_FINGER_MCP"), ("INDEX_FINGER_MCP", "INDEX_FINGER_PIP"),
    ("INDEX_FINGER_PIP", "INDEX_FINGER_DIP"), ("INDEX_FINGER_DIP", "INDEX_FINGER_TIP"),
    ("WRIST", "MIDDLE_FINGER_MCP"), ("MIDDLE_FINGER_MCP", "MIDDLE_FINGER_PIP"),
    ("MIDDLE_FINGER_PIP", "MIDDLE_FINGER_DIP"), ("MIDDLE_FINGER_DIP", "MIDDLE_FINGER_TIP"),
    ("WRIST", "RING_FINGER_MCP"), ("RING_FINGER_MCP", "RING_FINGER_PIP"),
    ("RING_FINGER_PIP", "RING_FINGER_DIP"), ("RING_FINGER_DIP", "RING_FINGER_TIP"),
    ("WRIST", "PINKY_MCP"), ("PINKY_MCP", "PINKY_PIP"),
    ("PINKY_PIP", "PINKY_DIP"), ("PINKY_DIP", "PINKY_TIP"),
    # palm arch, so the MCPs read as a hand rather than five loose chains
    ("INDEX_FINGER_MCP", "MIDDLE_FINGER_MCP"), ("MIDDLE_FINGER_MCP", "RING_FINGER_MCP"),
    ("RING_FINGER_MCP", "PINKY_MCP"),
]


def _auto_joints(bodyparts):
    """Fallback joint set for a model whose bodyparts aren't the MP21 layout.

    Chains consecutive bodyparts into (i-1, i, i+1) triplets. Crude, but it
    means a 4- or 5-keypoint single-finger model still produces angles instead
    of the actor refusing to start.
    """
    return {bodyparts[i]: (bodyparts[i - 1], bodyparts[i], bodyparts[i + 1])
            for i in range(1, len(bodyparts) - 1)}


def _auto_skeleton(bodyparts):
    """Fallback skeleton: chain consecutive bodyparts in list order."""
    return [(bodyparts[i], bodyparts[i + 1]) for i in range(len(bodyparts) - 1)]


# MediaPipe HandLandmarker's fixed output order (indices 0-20). This is also
# humanWristMP21's bodyparts list -- that DLC project's labels were bootstrapped
# from this exact model (see hand-tracking-notebooks-jake-2026-08-17/mediapipe/
# mediapipe_dlc_utils.py), so MP21_JOINTS/MP21_SKELETON above apply unchanged
# to either backend.
MEDIAPIPE_LANDMARK_NAMES = [
    "WRIST",
    "THUMB_CMC", "THUMB_MCP", "THUMB_IP", "THUMB_TIP",
    "INDEX_FINGER_MCP", "INDEX_FINGER_PIP", "INDEX_FINGER_DIP", "INDEX_FINGER_TIP",
    "MIDDLE_FINGER_MCP", "MIDDLE_FINGER_PIP", "MIDDLE_FINGER_DIP", "MIDDLE_FINGER_TIP",
    "RING_FINGER_MCP", "RING_FINGER_PIP", "RING_FINGER_DIP", "RING_FINGER_TIP",
    "PINKY_MCP", "PINKY_PIP", "PINKY_DIP", "PINKY_TIP",
]


def _prefixed_hand_names(prefix):
    """MP21 landmark name -> bodypart name for a lowercase `{prefix}_wrist`,
    `{prefix}_index_mcp`... layout (the HandTrackingVideo1 two-hand models)."""
    out = {}
    for n in MEDIAPIPE_LANDMARK_NAMES:
        out[n] = prefix + "_" + n.lower().replace("_finger", "")
    return out


def _hand_layouts(bodyparts):
    """Joint/skeleton definitions for every MP21-shaped hand found in bodyparts.

    Handles the bare uppercase layout (humanWristMP21) and one or two prefixed
    hands (right_*/left_*). Joint names are prefixed with the hand when there is
    more than one. Returns (joints, skeleton), or (None, None) if nothing fits.
    """
    have = set(bodyparts)
    maps = []
    if all(n in have for n in MEDIAPIPE_LANDMARK_NAMES):
        maps.append(("", {n: n for n in MEDIAPIPE_LANDMARK_NAMES}))
    for prefix in ("right", "left"):
        m = _prefixed_hand_names(prefix)
        if all(v in have for v in m.values()):
            maps.append((prefix, m))
    if not maps:
        return None, None
    joints, skeleton = {}, []
    for prefix, m in maps:
        for jn, trip in MP21_JOINTS.items():
            key = f"{prefix}_{jn}" if prefix else jn
            joints[key] = tuple(m[t] for t in trip)
        skeleton += [(m[a], m[b]) for a, b in MP21_SKELETON]
    return joints, skeleton


class ProcessorBatch3D(Actor):
    """One actor, N cameras, one batched forward pass, 3D keypoints + joint angles."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_cameras = kwargs.get('num_cameras', 4)
        # Physical camera_num for each frames{i}_in slot, in slot order. Slot
        # index is just wiring order in the yaml and is NOT a camera identity --
        # this list is what maps a slot to the camera it actually carries, and
        # therefore to the right calibration entry.
        self.camera_nums = kwargs.get('camera_nums', list(range(self.num_cameras)))
        self.pred_active = kwargs.get('pred_active', True)
        # A camera whose frame was captured more than this many ms before the newest
        # frame in the step is left out of that step's triangulation (its 2D keypoints
        # are still shown). None = never drop. Uses the cameras' driver capture
        # timestamps, so it does nothing for a camera that sends none (e.g. Generator).
        self.max_frame_skew_ms = kwargs.get('max_frame_skew_ms', None)
        # False = 2D-only: run pose estimation and hand the per-camera keypoints
        # to the GUI, but skip calibration, triangulation and joint angles. For
        # runs without a usable calibration.
        self.triangulate = kwargs.get('triangulate', True)

    # ------------------------------------------------------------------ setup

    def setup(self):
        source_folder = Path(__file__).resolve().parent.parent
        with open(f'{source_folder}/config/config.yaml', 'r') as file:
            config = yaml.safe_load(file)
        self.config = config

        # Claim P-core(s) before anything spins up CUDA/TFLite helper threads --
        # thread affinity is only inherited by threads created after this call.
        # The mediapipe backend runs num_cameras HandLandmarker instances
        # concurrently on a thread pool and needs more than one physical core
        # or that concurrency starves itself (measured 57 ms for a 4-camera
        # step pinned to 1 core vs 32.6 ms given the P-cores that used to be
        # split across 4 separate per-camera Processor actors). The dlc
        # backend's forward pass is GPU-bound and single-threaded on the CPU
        # side (torch.set_num_threads(1) in _setup_dlc), so it keeps the
        # original one-core pin.
        backend = config.get('batch3d_backend', 'mediapipe')
        n_slots = self.num_cameras if (self.pred_active and backend == 'mediapipe') else 1
        cpu_affinity.pin_actor(cpu_affinity.COMPUTE, slot=0, label="ProcessorBatch3D",
                               n_slots=n_slots)

        self._getStoreInterface()

        if not self.pred_active:
            logger.info("ProcessorBatch3D: pred_active=False, running as a no-op")
            return

        self.resize = config['resize']
        self.camera_prescaled = config.get('camera_prescaled', False)
        # Keypoints below this likelihood are dropped (set to NaN) before
        # triangulation, so a camera that cannot see a joint contributes nothing
        # to it rather than dragging the DLT solution toward a bad ray. This is
        # the knob the task asks for; aniposelib then needs >=2 surviving views
        # per keypoint and returns NaN for the rest. (With the mediapipe
        # backend this is a per-camera whole-hand gate, not per-keypoint --
        # see _setup_mediapipe / _infer_mediapipe.)
        # Temporal cleanup of the 3D output (both off by default): frames to hold
        # a lost keypoint, and EMA weight on the previous value (0 = no smoothing).
        self.hold_3d = int(config.get('triangulation_hold_frames', 0))
        self.smooth_3d = float(config.get('triangulation_smooth_alpha', 0.0))
        self._last_3d = None
        self.likelihood_threshold = float(config.get('triangulation_likelihood_threshold', 0.3))
        self.min_cameras = int(config.get('triangulation_min_cameras', 2))

        # ---------------- pose estimator ----------------
        self.backend = config.get('batch3d_backend', 'mediapipe')
        if self.backend == 'mediapipe':
            self._setup_mediapipe(config)
        elif self.backend == 'dlc':
            self._setup_dlc(config)
        else:
            raise ValueError(f"unknown batch3d_backend {self.backend!r}, expected "
                             f"'mediapipe' or 'dlc'")

        self.bp_index = {name: i for i, name in enumerate(self.bodyparts)}
        logger.info(f"backend={self.backend}, {self.n_keypoints} keypoints, "
                    f"bodyparts: {self.bodyparts}")

        # ---------------- calibration ----------------
        if self.triangulate:
            if not _HAVE_3D_DEPS:
                raise ImportError("aniposelib/dlc2kinematics are not installed in this env "
                                  "(use improvPytorchJarvis, or set triangulate: false)")
            self._setup_calibration(config, source_folder)
        else:
            self.cgroup = None
            logger.info("triangulate=False: 2D keypoints only, no 3D / joint angles")

        # ---------------- joint angles ----------------
        self.joints, self.skeleton = _hand_layouts(self.bodyparts)
        if self.joints:
            logger.info("using the MP21 hand joint/skeleton definitions")
        else:
            self.joints = _auto_joints(self.bodyparts)
            self.skeleton = _auto_skeleton(self.bodyparts)
            logger.warning(f"bodyparts are not the MP21 hand layout; falling back to "
                           f"a consecutive-triplet joint set ({len(self.joints)} joints)")
        self.joint_names = list(self.joints)
        # Resolve names to indices once -- doing it per frame would be a dict
        # lookup per joint per frame for no reason.
        self.joint_idx = [tuple(self.bp_index[b] for b in self.joints[j])
                          for j in self.joint_names]
        self.skeleton_idx = [(self.bp_index[a], self.bp_index[b])
                             for a, b in self.skeleton
                             if a in self.bp_index and b in self.bp_index]
        logger.info(f"{len(self.joint_names)} joint angles: {self.joint_names}")

        # ---------------- bookkeeping ----------------
        self.frame_num = 0
        self.frames_log = 200
        self.time_start = time.perf_counter()

        self.points_2d_log = []
        self.points_3d_log = []
        self.angles_log = []
        self.timestamps = []
        self.frame_nums_received = []
        self.frame_skew = []           # max-min frame_num across cameras, per step
        # Real time spread of the frames in each step (ms, from the cameras' capture
        # timestamps; NaN if <2 cameras stamped). Unlike frame_skew, this is not
        # affected by when each camera actor happened to start counting.
        self.time_skew_ms = []
        self.stale_cameras = []        # slots left out of triangulation by max_frame_skew_ms, per step
        self.cameras_present = []      # how many cameras contributed each step

        self.queue_wait_latencies = []
        self.store_get_latencies = []
        self.inference_latencies = []
        self.triangulate_latencies = []
        self.angle_latencies = []
        self.queue_put_latencies = []
        self.total_latencies = []

        self.out_folder = run_folder()
        logger.info(f"Output folder set to {self.out_folder}")
        logger.info(f"likelihood threshold {self.likelihood_threshold}, "
                    f"min cameras per keypoint {self.min_cameras}")
        logger.info("Completed setup for ProcessorBatch3D")

    def _setup_mediapipe(self, config):
        """One HandLandmarker instance per camera, run concurrently.

        MediaPipe's Python Tasks API has no batched/tensor-batch call -- a
        single HandLandmarker.detect() takes one image. The concurrency trick
        that gets this to real-time is that TFLite's C++ inference releases the
        GIL, so N landmarker instances called from N threads run genuinely in
        parallel on CPU: measured 32.6 ms for 4 cameras via a ThreadPoolExecutor
        vs 138 ms calling one instance 4 times sequentially. One instance per
        camera (rather than sharing one across threads) sidesteps needing
        HandLandmarker to be thread-safe for concurrent .detect() calls on the
        same object, which is not documented as supported.
        """
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision as mp_vision
        self._mp = mp

        model_path = config['mediapipe_model_path']
        num_hands = int(config.get('mediapipe_num_hands', 1))
        # Both hands: 42 keypoints, right_* rows then left_* rows (the same
        # layout as the two-hand DLC models, so joint/skeleton setup and the GUI
        # need nothing special). Each row block is filled from the detection
        # MediaPipe labels with that handedness.
        self.mp_both_hands = bool(config.get('mediapipe_both_hands', False))
        self.mp_swap_handedness = bool(config.get('mediapipe_swap_handedness', False))
        if self.mp_both_hands:
            num_hands = max(num_hands, 2)
        min_conf = float(config.get('mediapipe_min_hand_detection_confidence', 0.5))
        # VIDEO mode carries a tracker between frames, so a hand found once is
        # followed instead of re-detected from scratch every frame (IMAGE mode).
        # It needs strictly increasing timestamps per landmarker (see detect_one).
        self.mp_video_mode = str(config.get('mediapipe_running_mode', 'image')).lower() == 'video'
        self._mp_ts_ms = 0
        self._mp_step_ms = max(1, int(round(1000.0 / float(config.get('fps', 30)))))
        track_conf = float(config.get('mediapipe_min_tracking_confidence', 0.5))
        presence_conf = float(config.get('mediapipe_min_hand_presence_confidence', 0.5))
        # Keep a lost hand's last 2D keypoints for this many frames instead of
        # blanking them at once (0 = off).
        self.mp_hold_frames = int(config.get('mediapipe_hold_frames', 0))
        self._mp_held = {}   # (slot, block offset) -> [pred, age]
        delegate_name = str(config.get('mediapipe_delegate', 'CPU')).upper()
        delegate = getattr(mp_python.BaseOptions.Delegate, delegate_name)

        self.landmarkers = []
        for _ in range(self.num_cameras):
            base_options = mp_python.BaseOptions(model_asset_path=model_path, delegate=delegate)
            options = mp_vision.HandLandmarkerOptions(
                base_options=base_options,
                running_mode=(mp_vision.RunningMode.VIDEO if self.mp_video_mode
                              else mp_vision.RunningMode.IMAGE),
                num_hands=num_hands,
                min_hand_detection_confidence=min_conf,
                min_hand_presence_confidence=presence_conf,
                min_tracking_confidence=track_conf,
            )
            self.landmarkers.append(mp_vision.HandLandmarker.create_from_options(options))

        self.mp_executor = ThreadPoolExecutor(max_workers=self.num_cameras,
                                              thread_name_prefix="mediapipe")
        if self.mp_both_hands:
            self.bodyparts = ([_prefixed_hand_names("right")[n] for n in MEDIAPIPE_LANDMARK_NAMES]
                              + [_prefixed_hand_names("left")[n] for n in MEDIAPIPE_LANDMARK_NAMES])
        else:
            self.bodyparts = list(MEDIAPIPE_LANDMARK_NAMES)
        self.n_keypoints = len(self.bodyparts)
        logger.info(f"mediapipe: both_hands={self.mp_both_hands}, "
                    f"swap_handedness={self.mp_swap_handedness}, mediapipe: {model_path}, delegate={delegate_name}, "
                    f"{self.num_cameras} landmarker instances, min_hand_detection_confidence={min_conf}")

    def _setup_dlc(self, config):
        """One DLC model, batched across all cameras in a single forward pass.

        batch_size == num_cameras is the whole point: inference() fills a batch
        and only flushes when it is full, so anything smaller would split one
        step's frames across several forward passes.
        """
        import torch
        from deeplabcut.pose_estimation_pytorch.config import read_config_as_dict
        from deeplabcut.pose_estimation_pytorch.apis.utils import get_inference_runners
        torch.set_num_threads(1)

        train_dir = Path(config['batch3d_model_path'])
        snapshot_path = train_dir / config['batch3d_model_snapshot']
        # Top-down models (method: td) need their detector to find the crop the
        # pose net runs on; bottom-up models leave batch3d_detector_snapshot unset.
        det_snap = config.get('batch3d_detector_snapshot')
        detector_path = train_dir / det_snap if det_snap else None
        model_cfg = read_config_as_dict(train_dir / "pytorch_config.yaml")

        # PAF -> HeatmapPredictor override, same as processor.py. These models
        # are single-animal, so PAF's grouping step has no job to do and is the
        # sole source of blanked-out (NaN) keypoints; HeatmapPredictor argmaxes
        # per bodypart and always returns a coordinate + score. Critically for
        # 3D: a NaN here is indistinguishable from "below threshold" downstream,
        # so PAF would silently starve the triangulation.
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
                    logger.info(f"predictor {old_pred.get('type')} -> HeatmapPredictor")
            except (KeyError, TypeError) as e:
                logger.warning(f"could not override predictor ({e}); keeping the model's own")

        # Configs written by a newer DLC carry unset (null) detector options the
        # installed one does not know (SSDLite has no `variant` here) -- drop
        # them so the constructor falls back to its own defaults.
        det_model = (model_cfg.get('detector') or {}).get('model')
        if isinstance(det_model, dict):
            for k in [k for k, v in det_model.items() if v is None]:
                del det_model[k]

        self.bodyparts = list(model_cfg['metadata']['bodyparts'])
        self.n_keypoints = len(self.bodyparts)

        self.pose_runner, self.detector_runner = get_inference_runners(
            model_config=model_cfg,
            snapshot_path=snapshot_path,
            max_individuals=1,
            batch_size=self.num_cameras,
            detector_batch_size=self.num_cameras if detector_path else 1,
            detector_path=detector_path,
        )
        logger.info(f"dlc model {train_dir.name} / {config['batch3d_model_snapshot']}, "
                    f"batch_size={self.num_cameras}")

    def _setup_calibration(self, config, source_folder):
        """Load the anipose calibration and work out which row is which camera.

        The toml stores cameras by *name* (cam0, cam2, ...), and the order they
        appear in is the order aniposelib expects points in. A run can wire
        cameras the calibration does not contain (e.g. today's calibration only
        solved cam0+cam2) -- those slots are still processed for 2D but simply
        never contribute a ray, which is why self.calib_row can hold None.
        """
        self.cgroup = None
        self.calib_row = [None] * self.num_cameras
        # Set here, not only on the success path: every early return below would
        # otherwise leave the attribute undefined.
        self.calib_scale = None

        toml_path = config.get('calibration_toml')
        if not toml_path:
            logger.error("no `calibration_toml` in config.yaml -- 3D output will be all-NaN")
            return

        toml_path = Path(toml_path)
        if not toml_path.is_absolute():
            toml_path = source_folder / toml_path
        if not toml_path.exists():
            logger.error(f"calibration file {toml_path} not found -- 3D output will be all-NaN")
            return

        try:
            self.cgroup = CameraGroup.load(str(toml_path))
        except Exception as e:
            logger.error(f"could not load calibration {toml_path}: {e}")
            return

        calib_names = list(self.cgroup.get_names())
        # camera_num -> name in the toml. Defaults to "cam{N}", which is what
        # calibration_20260903 writes, but is overridable because a calibration
        # produced elsewhere may use the rig's own names (top, back_left, ...).
        name_map = config.get('calibration_camera_names') or {}
        for slot, cam_num in enumerate(self.camera_nums):
            name = str(name_map.get(cam_num, f"cam{cam_num}"))
            if name in calib_names:
                self.calib_row[slot] = calib_names.index(name)
            else:
                logger.warning(f"camera_num {cam_num} (slot {slot}) maps to calibration "
                               f"name {name!r}, which is not in {calib_names} -- this "
                               f"camera will not contribute to triangulation")

        n_linked = sum(r is not None for r in self.calib_row)
        logger.info(f"calibration {toml_path.name}: cameras {calib_names}; "
                    f"{n_linked}/{self.num_cameras} wired cameras are calibrated")
        if n_linked < 2:
            logger.error(f"only {n_linked} calibrated camera(s) among the wired ones -- "
                         f"triangulation needs at least 2 and will return all-NaN")

        # The calibration was solved at some frame size; live frames must be
        # expressed in those same pixel units. Both are 960x540 today, so this
        # is a no-op, but it stops a silent, hard-to-spot error if a calibration
        # made at 1920x1080 is ever dropped in.
        calib_size = config.get('calibration_frame_size')
        if calib_size:
            cw, ch = float(calib_size[0]), float(calib_size[1])
            live = self.cgroup.cameras[0].get_size()
            if live is not None and (abs(live[0] - cw) > 1 or abs(live[1] - ch) > 1):
                self.calib_scale = (cw / live[0], ch / live[1])
                logger.info(f"scaling live 2D points by {self.calib_scale} into calibration space")

    # ------------------------------------------------------------------- step

    def _gather_frames(self):
        """Take the newest frame from every camera slot.

        Each slot is drained to its most recent message: a stale frame is worse
        than a missing one here, because all N frames are triangulated as if
        they were simultaneous. Returns (frames, meta, present) where `present`
        marks the slots that actually produced a frame this step.
        """
        frame_ids = [None] * self.num_cameras
        starts = [None] * self.num_cameras
        fnums = [-1] * self.num_cameras
        caps = [None] * self.num_cameras     # wall-clock capture time (driver timestamp), if the camera sent one

        for slot in range(self.num_cameras):
            link = self.links.get(f"frames{slot}_in")
            if link is None:
                continue
            msg = None
            try:
                msg = link.get(timeout=0.005)
            except Exception:
                continue
            # Drain to the newest available frame for this camera.
            while True:
                try:
                    msg = link.get_nowait()
                except Exception:
                    break
            if len(msg) >= 3:
                frame_ids[slot], starts[slot], fnums[slot] = msg[:3]
                if len(msg) >= 4: caps[slot] = msg[3]
            elif len(msg) == 2:
                frame_ids[slot], starts[slot] = msg

        present = [i for i, fid in enumerate(frame_ids) if fid is not None]
        return frame_ids, starts, fnums, present, caps

    def runStep(self):
        if not self.pred_active:
            return

        step_start = time.perf_counter()

        # --- 1. one newest frame per camera ---
        t0 = time.perf_counter()
        frame_ids, starts, fnums, present, caps = self._gather_frames()
        self.queue_wait_latencies.append(time.perf_counter() - t0)

        if not present:
            return

        # --- 2. pull the frames out of the store ---
        t0 = time.perf_counter()
        frames = {}
        for slot in present:
            try:
                frame = self.client.get(frame_ids[slot])
                if not (isinstance(frame, np.ndarray) and frame.ndim == 3):
                    frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)
                if not self.camera_prescaled:
                    frame = cv2.resize(
                        frame,
                        (int(frame.shape[1] * self.resize), int(frame.shape[0] * self.resize)))
                frames[slot] = frame
            except ObjectNotFoundError:
                logger.debug(f"slot {slot}: frame gone from store, skipping this camera")
            except Exception as e:
                logger.error(f"slot {slot}: store get failed: {e}")
        self.store_get_latencies.append(time.perf_counter() - t0)

        if not frames:
            return

        batch_slots = sorted(frames)
        self.frame_num += 1
        self.timestamps.append(time.time())
        self.frame_nums_received.append([fnums[s] for s in batch_slots])
        self.cameras_present.append(len(batch_slots))

        # The cameras free-run, so "the same step" is not "the same instant".
        # Record the spread so a run can be judged after the fact -- at 30 fps a
        # skew of 1 frame is 33 ms, which is ~8 mm of 3D error on a fingertip
        # moving 0.25 m/s. This is the number that justifies hardware sync.
        valid_fnums = [fnums[s] for s in batch_slots if fnums[s] >= 0]
        self.frame_skew.append(max(valid_fnums) - min(valid_fnums) if len(valid_fnums) > 1 else 0)

        stamped = {s: caps[s] for s in batch_slots if caps[s] is not None and np.isfinite(caps[s])}
        self.time_skew_ms.append((max(stamped.values()) - min(stamped.values())) * 1e3
                                 if len(stamped) > 1 else float('nan'))
        stale = set()
        if self.max_frame_skew_ms is not None and len(stamped) > 1:
            newest = max(stamped.values())
            stale = {s for s, t in stamped.items() if (newest - t) * 1e3 > self.max_frame_skew_ms}
        self.stale_cameras.append(sorted(stale))

        # --- 3. pose estimation: N cameras through ONE model, ONE step ---
        # DLC: a literal batched tensor forward pass. mediapipe: N landmarker
        # instances run concurrently on a thread pool (see _infer_mediapipe) --
        # its Python API has no tensor-batch call, so this is the closest
        # equivalent. Either way, exactly one q_in-to-q_out step per output.
        t0 = time.perf_counter()
        try:
            raw_2d = self._infer(frames, batch_slots)
        except Exception as e:
            logger.error(f"inference failed: {e}")
            logger.error(traceback.format_exc())
            self.inference_latencies.append(time.perf_counter() - t0)
            return
        self.inference_latencies.append(time.perf_counter() - t0)

        if not self.triangulate:
            self.points_2d_log.append(raw_2d)
            self._emit(None, {}, starts, fnums, batch_slots, raw_2d, step_start)
            return

        # --- 4. per-camera 2D keypoints, in calibration camera order ---
        # points_2d rows are indexed by the CALIBRATION's camera order, not by
        # wiring slot; anything the calibration doesn't know about stays NaN and
        # is thus ignored by triangulate().
        n_calib = len(self.cgroup.cameras) if self.cgroup is not None else 0
        points_2d = np.full((max(n_calib, 1), self.n_keypoints, 2), np.nan)

        for slot in batch_slots:
            row = self.calib_row[slot]
            if row is None or self.cgroup is None or slot in stale:
                continue

            pred = raw_2d[slot]
            xy = pred[:, :2].copy()
            lik = pred[:, 2]
            # The likelihood gate. -1/-2 are DLC's "no detection" sentinels and
            # are not coordinates; NaN is already unusable. Everything else is
            # kept only if the model is confident enough about it. (mediapipe:
            # `lik` is the same handedness score repeated across all 21 rows --
            # see _infer_mediapipe -- so this gate is whole-hand there, not
            # per-keypoint.)
            bad = (~np.isfinite(xy).all(axis=1)) | (lik < self.likelihood_threshold) \
                  | (xy[:, 0] < -1.5) | (xy[:, 1] < -1.5)
            xy[bad] = np.nan
            if self.calib_scale is not None:
                xy[:, 0] *= self.calib_scale[0]
                xy[:, 1] *= self.calib_scale[1]
            points_2d[row] = xy

        # --- 5. triangulate ---
        t0 = time.perf_counter()
        points_3d = self._smooth_3d(self._triangulate(points_2d))
        self.triangulate_latencies.append(time.perf_counter() - t0)

        # --- 6. joint angles ---
        t0 = time.perf_counter()
        angles = self._joint_angles(points_3d)
        self.angle_latencies.append(time.perf_counter() - t0)

        self.points_2d_log.append(raw_2d)
        self.points_3d_log.append(points_3d)
        self.angles_log.append([angles[j] for j in self.joint_names])

        self._emit(points_3d, angles, starts, fnums, batch_slots, raw_2d, step_start)

        if self.frame_num % self.frames_log == 0:
            elapsed = time.perf_counter() - self.time_start
            n3d = int(np.sum(np.isfinite(points_3d[:, 0]))) if points_3d is not None else 0
            logger.info(f"frame {self.frame_num}: {round(self.frames_log / elapsed, 2)} fps, "
                        f"{len(batch_slots)} cams, {n3d}/{self.n_keypoints} keypoints in 3D, "
                        f"frame skew {self.frame_skew[-1]}, time skew {np.nanmedian(self.time_skew_ms[-self.frames_log:]):.1f} ms, "
                        f"infer {np.median(self.inference_latencies[-self.frames_log:])*1000:.1f} ms, "
                        f"tri {np.median(self.triangulate_latencies[-self.frames_log:])*1000:.2f} ms, "
                        f"ang {np.median(self.angle_latencies[-self.frames_log:])*1000:.2f} ms")
            self.time_start = time.perf_counter()

    def _emit(self, points_3d, angles, starts, fnums, batch_slots, raw_2d, step_start):
        """Step 7: publish one message covering every camera (points_3d None in
        2D-only mode)."""
        valid_fnums = [fnums[s] for s in batch_slots if fnums[s] >= 0]
        # camera_start is taken from the earliest contributing camera so the
        # end-to-end latency the sender computes is the worst case over the
        # batch, not a flattering pick.
        valid_starts = [starts[s] for s in batch_slots if starts[s] is not None]
        camera_start = min(valid_starts) if valid_starts else time.time()
        gen_frame_num = max(valid_fnums) if valid_fnums else self.frame_num

        t0 = time.perf_counter()
        try:
            msg = {
                'joint_angles': angles,
                'joint_names': self.joint_names,
                'points_3d': points_3d,
                'points_2d': raw_2d,
                'skeleton_idx': self.skeleton_idx,
                'bodyparts': self.bodyparts,
                'camera_start': camera_start,
                'frame_num': gen_frame_num,
                'camera_slots': batch_slots,
                'camera_nums': [self.camera_nums[s] for s in batch_slots],
            }
            self.q_out.put(msg)
        except Exception as e:
            logger.error(f"q_out.put failed: {e}")
            logger.error(traceback.format_exc())
        self.queue_put_latencies.append(time.perf_counter() - t0)

        self.total_latencies.append(time.perf_counter() - step_start)


    # -------------------------------------------------------------- inference

    def _infer(self, frames, batch_slots):
        """Dispatch to the configured backend. Returns (num_cameras, K, 3) --
        x, y, likelihood per camera slot, NaN for any slot not in batch_slots."""
        if self.backend == 'mediapipe':
            return self._infer_mediapipe(frames, batch_slots)
        return self._infer_dlc(frames, batch_slots)

    def _infer_dlc(self, frames, batch_slots):
        batch = [frames[s] for s in batch_slots]
        if self.detector_runner is not None:
            # Top-down: detector finds the box(es), pose net runs on each crop.
            ctx = self.detector_runner.inference(batch)
            batch = list(zip(batch, ctx))
        raw = self.pose_runner.inference(batch)
        raw_2d = np.full((self.num_cameras, self.n_keypoints, 3), np.nan)
        for out_i, slot in enumerate(batch_slots):
            try:
                pred = np.asarray(raw[out_i]['bodyparts'][0], dtype=float)
            except Exception as e:
                logger.error(f"slot {slot}: could not read prediction: {e}")
                continue
            # DLCRNet emits offset columns 3/4; heatmap-only models emit x,y,score.
            if pred.shape[1] >= 5:
                pred[:, 0] += pred[:, 3]
                pred[:, 1] += pred[:, 4]
            raw_2d[slot] = pred[:, :3]
        return raw_2d

    def _infer_mediapipe(self, frames, batch_slots):
        """Run every camera's landmarker concurrently; assemble into the same
        (num_cameras, K, 3) shape the DLC path produces.

        MediaPipe gives no per-keypoint confidence (visibility/presence are
        always 0 for HandLandmarker), only a per-HAND handedness score. That
        score is broadcast across all 21 rows' likelihood column, so the
        existing per-keypoint likelihood gate in runStep becomes, for this
        backend, a whole-hand accept/reject per camera -- which is the correct
        adaptation: mediapipe either finds the whole hand or it doesn't.
        """
        mp = self._mp

        def detect_one(slot):
            # Frames arriving from the store are already RGB -- both the real
            # camera path (TIS.py opens the device with SinkFormats.RGB) and the
            # synthetic Generator (which converts BGR->RGB before client.put)
            # guarantee this, and nothing else in this codebase re-converts
            # them (front_end_3d.py renders them straight into
            # QImage.Format_RGB888). No cv2.cvtColor here -- swapping channels
            # on an already-RGB frame would hand mediapipe a blue-tinted image.
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frames[slot])
            if self.mp_video_mode:
                return slot, self.landmarkers[slot].detect_for_video(mp_image, ts_ms)
            return slot, self.landmarkers[slot].detect(mp_image)

        self._mp_ts_ms += self._mp_step_ms
        ts_ms = self._mp_ts_ms

        raw_2d = np.full((self.num_cameras, self.n_keypoints, 3), np.nan)
        for slot, result in self.mp_executor.map(detect_one, batch_slots):
            if not result.hand_landmarks:
                continue
            h, w = frames[slot].shape[:2]

            def hand_rows(i):
                hand = result.hand_landmarks[i]
                score = (result.handedness[i][0].score if result.handedness else 1.0)
                pred = np.empty((len(hand), 3), dtype=float)
                pred[:, 0] = [lm.x * w for lm in hand]
                pred[:, 1] = [lm.y * h for lm in hand]
                pred[:, 2] = score
                return pred, score

            if not self.mp_both_hands:
                raw_2d[slot] = hand_rows(0)[0]
                continue

            # Route each detection into its handedness block; if two detections
            # claim the same side, the more confident one wins.
            best = {}
            for i in range(len(result.hand_landmarks)):
                label = result.handedness[i][0].category_name.lower() if result.handedness else "right"
                if self.mp_swap_handedness:
                    label = "left" if label == "right" else "right"
                pred, score = hand_rows(i)
                if label not in best or score > best[label][1]:
                    best[label] = (pred, score)
            for label, (pred, _) in best.items():
                off = 0 if label == "right" else 21
                raw_2d[slot, off:off + 21] = pred
        if self.mp_hold_frames > 0:
            self._hold_2d(raw_2d, batch_slots)
        return raw_2d

    def _hold_2d(self, raw_2d, batch_slots):
        """Fill a hand block that vanished for a few frames with its last value."""
        for slot in batch_slots:
            for off in range(0, self.n_keypoints, 21):
                block = raw_2d[slot, off:off + 21]
                key = (slot, off)
                if np.isfinite(block[:, 0]).any():
                    self._mp_held[key] = [block.copy(), 0]
                elif key in self._mp_held:
                    held = self._mp_held[key]
                    held[1] += 1
                    if held[1] <= self.mp_hold_frames:
                        raw_2d[slot, off:off + 21] = held[0]
                    else:
                        del self._mp_held[key]

    # ------------------------------------------------------------------- math

    def _smooth_3d(self, pts):
        """Optional temporal cleanup of the triangulated keypoints: hold the last
        good value through short dropouts, then EMA-blend into the new one."""
        if self.hold_3d <= 0 and self.smooth_3d <= 0:
            return pts
        if self._last_3d is None:
            self._last_3d = np.full_like(pts, np.nan)
            self._age_3d = np.zeros(len(pts), dtype=int)
        fresh = np.isfinite(pts).all(axis=1)
        out = pts.copy()
        self._age_3d = np.where(fresh, 0, self._age_3d + 1)
        prev_ok = np.isfinite(self._last_3d).all(axis=1)
        blend = fresh & prev_ok
        if self.smooth_3d > 0:
            a = self.smooth_3d
            out[blend] = a * self._last_3d[blend] + (1 - a) * pts[blend]
        hold = ~fresh & prev_ok & (self._age_3d <= self.hold_3d)
        out[hold] = self._last_3d[hold]
        self._last_3d = out.copy()
        return out

    def _triangulate(self, points_2d):
        """aniposelib DLT over every camera that kept a given keypoint.

        `CameraGroup.triangulate` already does the right thing with NaN: a
        keypoint is solved from whichever cameras still have it and comes back
        NaN if fewer than two do, which is exactly the likelihood-gate semantics
        we want. `triangulate_ransac` would be more robust to a confidently
        wrong keypoint but measured 387 ms per call here against 0.55 ms for
        this one, so it is not an option in a 33 ms budget.
        """
        if self.cgroup is None:
            return np.full((self.n_keypoints, 3), np.nan)

        # Enforce min_cameras before the solve when it is stricter than
        # aniposelib's built-in >=2, so the knob means what it says.
        if self.min_cameras > 2:
            seen = np.isfinite(points_2d[:, :, 0]).sum(axis=0)
            points_2d = points_2d.copy()
            points_2d[:, seen < self.min_cameras, :] = np.nan

        try:
            return np.asarray(self.cgroup.triangulate(points_2d, progress=False), dtype=float)
        except Exception as e:
            logger.error(f"triangulation failed: {e}")
            return np.full((self.n_keypoints, 3), np.nan)

    def _joint_angles(self, points_3d):
        """Joint angles via dlc2kinematics' own jointangle_calc.

        dlc2kinematics' batch entry point (`compute_joint_angles`) takes a whole
        DLC DataFrame and, with save=True, will silently read back a stale .h5
        instead of computing -- neither is usable per frame. `jointangle_calc`
        is the primitive underneath it and is what we call directly. It insists
        on a pandas object (`pos.values` in jointquat_calc, despite the
        docstring claiming ndarray is fine), hence the Series wrapper.

        Returns degrees, deviation-from-straight: 0 = fully extended.
        """
        angles = {}
        for name, (a, b, c) in zip(self.joint_names, self.joint_idx):
            trip = points_3d[[a, b, c]]
            if not np.isfinite(trip).all():
                angles[name] = float('nan')
                continue
            try:
                angles[name] = float(d2k_aux.jointangle_calc(pd.Series(trip.ravel())))
            except Exception:
                angles[name] = float('nan')
        return angles

    # ------------------------------------------------------------------- stop

    def stop(self):
        if not self.pred_active:
            logger.info("ProcessorBatch3D stopped (inactive)")
            return

        if getattr(self, 'backend', None) == 'mediapipe':
            try:
                self.mp_executor.shutdown(wait=False, cancel_futures=True)
                for lm in self.landmarkers:
                    lm.close()
            except Exception:
                logger.error(f"error closing mediapipe landmarkers: {traceback.format_exc()}")

        try:
            np.save(self.out_folder / "batch3d_points_2d.npy", np.asarray(self.points_2d_log))
            np.save(self.out_folder / "batch3d_points_3d.npy", np.asarray(self.points_3d_log))
            np.save(self.out_folder / "batch3d_joint_angles.npy", np.asarray(self.angles_log))
            np.save(self.out_folder / "batch3d_joint_names.npy", np.asarray(self.joint_names))
            np.save(self.out_folder / "batch3d_bodyparts.npy", np.asarray(self.bodyparts))
            np.save(self.out_folder / "batch3d_timestamps.npy", np.asarray(self.timestamps))
            np.save(self.out_folder / "batch3d_frame_skew.npy", np.asarray(self.frame_skew))
            np.save(self.out_folder / "batch3d_time_skew_ms.npy", np.asarray(self.time_skew_ms))
            np.save(self.out_folder / "batch3d_stale_cameras.npy", np.asarray(self.stale_cameras, dtype=object), allow_pickle=True)
            np.save(self.out_folder / "batch3d_cameras_present.npy", np.asarray(self.cameras_present))

            np.save(self.out_folder / "batch3d_lat_queue_wait.npy", np.asarray(self.queue_wait_latencies))
            np.save(self.out_folder / "batch3d_lat_store_get.npy", np.asarray(self.store_get_latencies))
            np.save(self.out_folder / "batch3d_lat_inference.npy", np.asarray(self.inference_latencies))
            np.save(self.out_folder / "batch3d_lat_triangulate.npy", np.asarray(self.triangulate_latencies))
            np.save(self.out_folder / "batch3d_lat_angles.npy", np.asarray(self.angle_latencies))
            np.save(self.out_folder / "batch3d_lat_queue_put.npy", np.asarray(self.queue_put_latencies))
            np.save(self.out_folder / "batch3d_lat_total.npy", np.asarray(self.total_latencies))

            if self.inference_latencies:
                logger.info(f"median batched inference: {np.median(self.inference_latencies)*1000:.1f} ms "
                            f"over {len(self.inference_latencies)} steps")
            if self.frame_skew:
                logger.info(f"frame skew across cameras: median {np.median(self.frame_skew):.1f}, "
                            f"max {np.max(self.frame_skew)} frames")
            logger.info(f"ProcessorBatch3D: saved {self.frame_num} frames to {self.out_folder}")
        except Exception:
            logger.error(f"could not save logs: {traceback.format_exc()}")
        logger.info("ProcessorBatch3D stopped")
