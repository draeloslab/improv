"""
Check actors/hand_onnx.py against MediaPipe on recorded video, and time it. ~/envs_trt:

    python validate_onnx.py <video.mp4> <reference.npz from make_reference.py> [--frames 600] [--palm-range 0 1] [--providers cpu|cuda|trt]

Reports: fraction of frames where each finds a hand, landmark pixel error (median/p95) on frames where both do, the handedness
sign agreement, and ms per frame (palm / landmark / total) of the ONNX tracker.
"""
import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'actors'))
from hand_onnx import OnnxHandTracker

ap = argparse.ArgumentParser()
ap.add_argument('video'); ap.add_argument('ref')
ap.add_argument('--frames', type=int, default=600)
ap.add_argument('--palm-range', type=float, nargs=2, default=[0.0, 1.0])
ap.add_argument('--conf', type=float, default=0.5)
ap.add_argument('--hands', type=int, default=2); ap.add_argument('--always-detect', action='store_true')
a = ap.parse_args()
ref = np.load(a.ref)
trk = OnnxHandTracker(Path(__file__).resolve().parents[2] / 'models' / 'hand_trt', num_hands=a.hands, det_conf=a.conf,
                      presence_conf=a.conf, palm_range=tuple(a.palm_range), always_detect=a.always_detect)
cap = cv2.VideoCapture(a.video)
errs, pres, agree_sign, n_both, n_onnx, n_mp, ms, palm_ms, lm_ms = [], [], [], 0, 0, 0, [], [], []
for i in range(min(a.frames, len(ref['n_hands']))):
    ok, f = cap.read()
    if not ok:
        break
    rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
    t0 = time.perf_counter()
    hands = trk.step(rgb)
    ms.append((time.perf_counter() - t0) * 1e3); palm_ms.append(trk.timing['palm'] * 1e3); lm_ms.append(trk.timing['landmark'] * 1e3)
    mp_h = [(ref['landmarks_px'][i, k], ref['handedness'][i, k]) for k in range(ref['landmarks_px'].shape[1]) if not np.isnan(ref['landmarks_px'][i, k, 0, 0])]
    n_onnx += bool(hands); n_mp += bool(mp_h)
    if hands and mp_h:
        n_both += 1
        for xy, lab in mp_h:                       # each MediaPipe hand vs the closest ONNX hand
            d = [np.linalg.norm(h['xy'] - xy, axis=1).mean() for h in hands]
            j = int(np.argmin(d)); errs.append(d[j]); pres.append(hands[j]['score'])
            agree_sign.append((hands[j]['right'] > 0.5) == (lab > 0))
n = len(ms)
print(f'{n} frames: hand found  onnx {n_onnx / n * 100:.0f}%  mediapipe {n_mp / n * 100:.0f}%  both {n_both / n * 100:.0f}%')
if errs:
    print(f'landmark error vs MediaPipe (mean px over 21 points, frame is {ref["size"][0]}x{ref["size"][1]}): median {np.median(errs):.2f}  p95 {np.percentile(errs, 95):.2f}  handedness agrees {np.mean(agree_sign) * 100:.0f}%')
errs, pres = np.array(errs), np.array(pres)
for lo in (0.0, 0.5, 0.9):
    m = pres >= lo
    if m.any():
        print(f'  presence >= {lo}: {m.sum()} hands, error median {np.median(errs[m]):.2f} px  p95 {np.percentile(errs[m], 95):.2f}')
ms = np.array(ms[20:])
print(f'onnx tracker per frame: median {np.median(ms):.2f} ms  p95 {np.percentile(ms, 95):.2f}  (palm {np.mean(palm_ms[20:]):.2f} avg, landmark {np.mean(lm_ms[20:]):.2f} avg)')
