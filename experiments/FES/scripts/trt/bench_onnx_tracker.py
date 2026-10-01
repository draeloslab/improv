"""
Apples-to-apples with bench_mediapipe.py: the ONNX tracker (actors/hand_onnx.py) on two recorded cameras in parallel threads,
same settings, ms per step. ~/envs_trt:   python bench_onnx_tracker.py <video A> <video B> [--frames 600]
"""
import argparse, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'actors'))
from hand_onnx import OnnxHandTracker

ap = argparse.ArgumentParser(); ap.add_argument('videos', nargs=2); ap.add_argument('--frames', type=int, default=600)
ap.add_argument('--providers', default='cpu', choices=['cpu', 'cuda', 'trt'])
a = ap.parse_args()
frames = []
for v in a.videos:
    cap, fs = cv2.VideoCapture(v), []
    while len(fs) < a.frames:
        ok, f = cap.read()
        if not ok:
            break
        fs.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    frames.append(fs)
n = min(map(len, frames))
models = Path(__file__).resolve().parents[2] / 'models' / 'hand_trt'
trt = [('TensorrtExecutionProvider', {'trt_fp16_enable': True}), 'CUDAExecutionProvider', 'CPUExecutionProvider']
prov = {'cpu': None, 'cuda': ['CUDAExecutionProvider', 'CPUExecutionProvider'], 'trt': trt}[a.providers]
pool = ThreadPoolExecutor(2)
for hands, conf in [(2, 0.02), (1, 0.02), (2, 0.5), (1, 0.5)]:
    trk = [OnnxHandTracker(models, num_hands=hands, det_conf=conf, presence_conf=conf, providers=prov) for _ in range(2)]
    t, found = [], 0
    for i in range(n):
        t0 = time.perf_counter()
        res = list(pool.map(lambda k: trk[k].step(frames[k][i]), range(2)))
        t.append((time.perf_counter() - t0) * 1e3); found += all(bool(r) for r in res)
    t = np.array(t[30:])
    print(f'num_hands={hands} conf={conf}: median {np.median(t):.1f} ms  p95 {np.percentile(t, 95):.1f} ms  hand in both cams {found / n * 100:.0f}% of steps')
