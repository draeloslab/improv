"""
The whole current DLC step (processor_batch3d._infer_dlc: SSDLite detector runner + pose runner, 2 cameras) on recorded frames,
split into detector / pose, to compare with bench_dlc.py's network-only times.  improvDLC3 env, run from ~.
"""
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from deeplabcut.pose_estimation_pytorch.apis.utils import get_inference_runners
from deeplabcut.pose_estimation_pytorch.config import read_config_as_dict

torch.set_num_threads(1)
cfg = yaml.safe_load(open(Path.home() / 'improv/experiments/FES/config/config.yaml'))
train = Path(cfg['batch3d_model_path'])
mc = read_config_as_dict(train / 'pytorch_config.yaml')
dm = (mc.get('detector') or {}).get('model')
for k in [k for k, v in (dm or {}).items() if v is None]:
    del dm[k]
pose, det = get_inference_runners(model_config=mc, snapshot_path=train / cfg['batch3d_model_snapshot'], max_individuals=1, batch_size=2,
                                  detector_batch_size=2, detector_path=train / cfg['batch3d_detector_snapshot'])
V = Path.home() / 'camera_video/2026-09-25/130900'
caps = [cv2.VideoCapture(str(V / f'camera_video_{c}_0925_1309.mp4')) for c in (0, 1)]
for c in caps:
    c.set(cv2.CAP_PROP_POS_FRAMES, 100)
td, tp = [], []
for i in range(80):
    frames = [cv2.cvtColor(c.read()[1], cv2.COLOR_BGR2RGB) for c in caps]
    t0 = time.perf_counter(); ctx = det.inference(frames); t1 = time.perf_counter(); pose.inference(list(zip(frames, ctx))); t2 = time.perf_counter()
    td.append((t1 - t0) * 1e3); tp.append((t2 - t1) * 1e3)
print(f'current DLC step, 2 cameras: detector {np.median(td[15:]):.1f} ms + pose {np.median(tp[15:]):.1f} ms = {np.median(np.array(td[15:]) + np.array(tp[15:])):.1f} ms')
