"""
Record MediaPipe's own output on a recorded video, to validate the ONNX port (validate_onnx.py) against. improvDLC3 env:

    python make_reference.py <video.mp4> <out.npz> [--frames 600] [--hands 2] [--conf 0.02]

Saves per frame: landmarks_px (frames, hands, 21, 2), handedness ('Left'/'Right' score), n_hands, in MediaPipe's own order.
"""
import argparse

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

MODEL = '/home/chesteklab/Desktop/hand-tracking-notebooks-jake-2026-08-17/mediapipe/hand_landmarker.task'
ap = argparse.ArgumentParser()
ap.add_argument('video'); ap.add_argument('out')
ap.add_argument('--frames', type=int, default=600); ap.add_argument('--image-mode', action='store_true', help='no tracker/smoothing: detect every frame'); ap.add_argument('--hands', type=int, default=2); ap.add_argument('--conf', type=float, default=0.02)
a = ap.parse_args()
lm = mp_vision.HandLandmarker.create_from_options(mp_vision.HandLandmarkerOptions(
    base_options=mp_python.BaseOptions(model_asset_path=MODEL), running_mode=mp_vision.RunningMode.IMAGE if a.image_mode else mp_vision.RunningMode.VIDEO, num_hands=a.hands,
    min_hand_detection_confidence=a.conf, min_hand_presence_confidence=a.conf, min_tracking_confidence=a.conf))
cap = cv2.VideoCapture(a.video)
pts, label, n = [], [], []
i = 0
while i < a.frames:
    ok, f = cap.read()
    if not ok:
        break
    h, w = f.shape[:2]
    im = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    r = lm.detect(im) if a.image_mode else lm.detect_for_video(im, i * 33)
    p = np.full((a.hands, 21, 2), np.nan); lab = np.full(a.hands, np.nan)
    for k, hand in enumerate(r.hand_landmarks[:a.hands]):
        p[k] = [[l.x * w, l.y * h] for l in hand]
        c = r.handedness[k][0]
        lab[k] = c.score if c.category_name == 'Right' else -c.score
    pts.append(p); label.append(lab); n.append(len(r.hand_landmarks)); i += 1
np.savez(a.out, landmarks_px=np.array(pts), handedness=np.array(label), n_hands=np.array(n), size=np.array([w, h]))
sp = np.nanmean([np.ptp(q[0], 0).max() for q in np.array(pts) if not np.isnan(q[0, 0, 0])] or [np.nan]); print(f'mean hand span {sp:.0f}px; {i} frames, hand found in {np.mean(np.array(n) > 0) * 100:.0f}%, two hands in {np.mean(np.array(n) > 1) * 100:.0f}%')
