"""
Baseline for the TensorRT work: time of the CURRENT pose step (one MediaPipe HandLandmarker per camera, run on a
thread pool, VIDEO mode, CPU) on recorded camera video, for a few settings. Run in the improvDLC3 env:

    python bench_mediapipe.py <video cam A> <video cam B> [--frames 600]

Prints median / p95 ms per step (both cameras in parallel, like processor_batch3d._infer_mediapipe) and how often a
hand was returned. Compare with the live log line "infer N ms" and batch3d_lat_inference.npy.
"""
import argparse
import time
from concurrent.futures import ThreadPoolExecutor

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

MODEL = '/home/chesteklab/Desktop/hand-tracking-notebooks-jake-2026-08-17/mediapipe/hand_landmarker.task'


def load(path, n):
    cap, out = cv2.VideoCapture(path), []
    while len(out) < n:
        ok, f = cap.read()
        if not ok:
            break
        out.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    return out


def bench(videos, num_hands, conf, delegate='CPU'):
    lms = [mp_vision.HandLandmarker.create_from_options(mp_vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=MODEL, delegate=getattr(mp_python.BaseOptions.Delegate, delegate)),
        running_mode=mp_vision.RunningMode.VIDEO, num_hands=num_hands, min_hand_detection_confidence=conf,
        min_hand_presence_confidence=conf, min_tracking_confidence=conf)) for _ in videos]
    pool, times, found = ThreadPoolExecutor(len(videos)), [], 0
    for i in range(len(videos[0])):
        def one(k):
            r = lms[k].detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=videos[k][i]), i * 33)
            return bool(r.hand_landmarks)
        t0 = time.perf_counter()
        res = list(pool.map(one, range(len(videos))))
        times.append((time.perf_counter() - t0) * 1e3)
        found += all(res)
    t = np.array(times[30:])           # drop warm-up
    return np.median(t), np.percentile(t, 95), found / len(times)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('videos', nargs='+')
    ap.add_argument('--frames', type=int, default=600)
    a = ap.parse_args()
    vids = [load(v, a.frames) for v in a.videos]
    n = min(map(len, vids))
    vids = [v[:n] for v in vids]
    print(f'{n} frames x {len(vids)} cameras, {vids[0][0].shape}')
    for hands, conf in [(2, 0.02), (1, 0.02), (2, 0.5), (1, 0.5)]:
        med, p95, found = bench(vids, hands, conf)
        print(f'num_hands={hands} conf={conf}: median {med:.1f} ms  p95 {p95:.1f} ms  hand in both cams {found * 100:.0f}% of steps')
