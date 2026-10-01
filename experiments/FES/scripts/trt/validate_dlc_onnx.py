"""
actors/dlc_onnx.py vs DLC's own runners on recorded 2-camera frames: keypoint disagreement (px) and ms per step. improvDLC3 env, run from ~.
"""
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from deeplabcut.pose_estimation_pytorch.apis.utils import get_inference_runners
from deeplabcut.pose_estimation_pytorch.config import read_config_as_dict

FES = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(FES))
from actors.dlc_onnx import DlcOnnxTopDown

torch.set_num_threads(1)
cfg = yaml.safe_load(open(FES / 'config/config.yaml'))
train = Path(cfg['batch3d_model_path'])
mc = read_config_as_dict(train / 'pytorch_config.yaml')
dm = (mc.get('detector') or {}).get('model')
for k in [k for k, v in (dm or {}).items() if v is None]:
    del dm[k]
mc['model']['heads']['bodypart']['predictor'] = {'type': 'HeatmapPredictor', 'apply_sigmoid': False, 'clip_scores': True,
                                                 'location_refinement': True, 'locref_std': 7.2801}
pose, det = get_inference_runners(model_config=mc, snapshot_path=train / cfg['batch3d_model_snapshot'], max_individuals=1, batch_size=2,
                                  detector_batch_size=2, detector_path=train / cfg['batch3d_detector_snapshot'])
import onnxruntime as ort
ort.preload_dlls()
trt = [('TensorrtExecutionProvider', {'trt_fp16_enable': True}), 'CUDAExecutionProvider', 'CPUExecutionProvider']   # one cached engine per batch size seen
nets = {'ONNX CUDA': DlcOnnxTopDown(FES / 'models/dlc_trt'),
        'ONNX det CUDA + pose TensorRT': DlcOnnxTopDown(FES / 'models/dlc_trt', pose_providers=trt)}
V = Path.home() / 'camera_video/2026-09-25/130900'
caps = [cv2.VideoCapture(str(V / f'camera_video_{c}_0925_1309.mp4')) for c in (0, 1)]
for c in caps:
    c.set(cv2.CAP_PROP_POS_FRAMES, 100)
errs = {k: [] for k in nets}; t_dlc = []; t_net = {k: [] for k in nets}; miss = {k: 0 for k in nets}
for i in range(120):
    frames = [cv2.cvtColor(c.read()[1], cv2.COLOR_BGR2RGB) for c in caps]
    t0 = time.perf_counter(); ctx = det.inference(frames); ref = pose.inference(list(zip(frames, ctx))); t_dlc.append((time.perf_counter() - t0) * 1e3)
    ref = [np.asarray(r['bodyparts'][0], float)[:, :3] for r in ref]
    for k, n in nets.items():
        t0 = time.perf_counter(); got = n.inference(frames); t_net[k].append((time.perf_counter() - t0) * 1e3)
        for g, r in zip(got, ref):
            if g is None:
                miss[k] += 1
            else:
                errs[k].append(np.linalg.norm(g[:, :2] - r[:, :2], axis=1))
print(f'DLC runners (PyTorch), 2 cameras: median {np.median(t_dlc[20:]):.1f} ms')
for k in nets:
    e = np.concatenate(errs[k]) if errs[k] else np.array([np.nan])
    print(f'{k}: median {np.median(t_net[k][20:]):.1f} ms  keypoint diff vs DLC median {np.median(e):.2f} px p95 {np.percentile(e, 95):.2f} px  no box: {miss[k]}')
