"""
MediaPipe's hand pipeline (palm detector -> rotated ROI -> landmark net -> next-frame ROI from the landmarks), re-implemented
on the two ONNX networks extracted from hand_landmarker.task (models/hand_trt, made with tf2onnx). One tracker per camera.

Why: MediaPipe's Python wrapper costs 13-35 ms per step on this rig, while the two networks alone run in 3.5 ms (palm) and
0.9 ms (landmarks) on plain CPU under onnxruntime (scripts/trt/bench_onnx.py). In steady state only the landmark net runs
(the ROI follows the hand); the palm detector runs only while fewer than num_hands hands are tracked, like MediaPipe's VIDEO mode.

    tracker = OnnxHandTracker(models_dir, num_hands=2)
    hands = tracker.step(rgb_frame)      # [{'xy': (21, 2) pixels, 'score': presence prob, 'right': handedness prob}, ...]

ROI maths follows MediaPipe's calculators: DetectionsToRects (rotation from palm keypoints 0 -> 2 to point up), RectTransformation
(scale 2.6, shift_y -0.5, square) for a detection, HandLandmarksToRect (scale 2.0, shift_y -0.1) for a tracked hand.
`providers` takes any onnxruntime provider list, so the same code runs on CUDA / TensorRT once those libraries are installed.
"""
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

PALM_IN, LM_IN = 192, 224


def _anchors():
    out = []
    for fm, per_cell in ((24, 2), (12, 6)):          # strides 8 (2 anchors/cell) and 16 (3 layers merged = 6/cell)
        for y in range(fm):
            for x in range(fm):
                out += [((x + 0.5) / fm, (y + 0.5) / fm)] * per_cell
    return np.array(out, np.float32)


ANCHORS = _anchors()


def _norm_angle(a):
    return a - 2 * np.pi * np.floor((a + np.pi) / (2 * np.pi))


def _rect(cx, cy, w, h, rot, scale, shift_y):
    """MediaPipe RectTransformation with square_long, pixel units: shift (in the rect's own axes), then scale."""
    cx, cy = cx - h * shift_y * np.sin(rot), cy + h * shift_y * np.cos(rot)
    size = max(w, h) * scale
    return cx, cy, size, rot


def _iou(a, b):
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    iw, ih = max(0, min(ax1, bx1) - max(ax0, bx0)), max(0, min(ay1, by1) - max(ay0, by0))
    inter = iw * ih
    return inter / (((ax1 - ax0) * (ay1 - ay0)) + ((bx1 - bx0) * (by1 - by0)) - inter + 1e-9)


class OnnxHandTracker:
    def __init__(self, models_dir, num_hands=2, det_conf=0.5, presence_conf=0.5, providers=None, palm_range=(0.0, 1.0)):
        models_dir = Path(models_dir)
        providers = providers or ['CPUExecutionProvider']
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 2
        self.palm = ort.InferenceSession(str(models_dir / 'hand_detector.onnx'), opts, providers=providers)
        self.lm = ort.InferenceSession(str(models_dir / 'hand_landmarks_detector.onnx'), opts, providers=providers)
        self.num_hands, self.det_conf, self.presence_conf, self.palm_range = num_hands, det_conf, presence_conf, palm_range
        self.rois = []            # tracked (cx, cy, size, rot) per hand, pixels
        self.timing = {'palm': 0.0, 'landmark': 0.0}

    # ------------------------------------------------------------ palm detector
    def detect_palms(self, rgb):
        h, w = rgb.shape[:2]
        s = PALM_IN / max(h, w)
        nw, nh = round(w * s), round(h * s)
        padx, pady = (PALM_IN - nw) // 2, (PALM_IN - nh) // 2
        img = np.zeros((PALM_IN, PALM_IN, 3), np.float32)
        img[pady:pady + nh, padx:padx + nw] = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_LINEAR) / 255.0
        lo, hi = self.palm_range
        img = img * (hi - lo) + lo
        reg, sc = self.palm.run(None, {'input_1': img[None]})
        reg, sc = reg[0], 1 / (1 + np.exp(-np.clip(sc[0, :, 0], -100, 100)))
        keep = np.flatnonzero(sc >= self.det_conf)
        if not len(keep):
            return []
        keep = keep[np.argsort(-sc[keep])[:16]]      # only the best candidates go to NMS (python loop)
        r, a = reg[keep], ANCHORS[keep]
        cx, cy = r[:, 0] / PALM_IN + a[:, 0], r[:, 1] / PALM_IN + a[:, 1]
        bw, bh = r[:, 2] / PALM_IN, r[:, 3] / PALM_IN
        kps = r[:, 4:18].reshape(-1, 7, 2) / PALM_IN + a[:, None, :]
        order = np.argsort(-sc[keep])
        out, taken = [], []
        for i in order:                                              # NMS, IoU 0.3
            box = (cx[i] - bw[i] / 2, cy[i] - bh[i] / 2, cx[i] + bw[i] / 2, cy[i] + bh[i] / 2)
            if any(_iou(box, t) > 0.3 for t in taken):
                continue
            taken.append(box)
            to_px = lambda p: np.array([(p[0] * PALM_IN - padx) / s, (p[1] * PALM_IN - pady) / s])
            c = to_px((cx[i], cy[i]))
            out.append({'c': c, 'wh': np.array([bw[i] * PALM_IN / s, bh[i] * PALM_IN / s]), 'k0': to_px(kps[i, 0]),
                        'k2': to_px(kps[i, 2]), 'score': float(sc[keep][i])})
            if len(out) >= self.num_hands:
                break
        return out

    @staticmethod
    def palm_to_roi(d):
        (x0, y0), (x2, y2) = d['k0'], d['k2']
        rot = _norm_angle(0.5 * np.pi - np.arctan2(-(y2 - y0), x2 - x0))
        return _rect(d['c'][0], d['c'][1], d['wh'][0], d['wh'][1], rot, 2.6, -0.5)

    # ------------------------------------------------------------ landmark net
    def landmarks(self, rgb, roi):
        cx, cy, size, rot = roi
        c, s = np.cos(rot), np.sin(rot)
        k = size / LM_IN                                           # crop pixel -> frame pixel
        # dst (crop) -> src (frame): p = center + R @ ((u - 112) * k)
        M = np.array([[c * k, -s * k, cx - (c * k * LM_IN / 2) + (s * k * LM_IN / 2)],
                      [s * k, c * k, cy - (s * k * LM_IN / 2) - (c * k * LM_IN / 2)]], np.float32)
        crop = cv2.warpAffine(rgb, M, (LM_IN, LM_IN), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_CONSTANT)
        lm, presence, right, _ = self.lm.run(None, {'input_1': (crop.astype(np.float32) / 255.0)[None]})
        p = lm[0].reshape(21, 3)[:, :2]
        u, v = (p[:, 0] / LM_IN - 0.5) * size, (p[:, 1] / LM_IN - 0.5) * size
        xy = np.stack([cx + c * u - s * v, cy + s * u + c * v], 1)
        return xy, float(presence[0, 0]), float(right[0, 0])

    @staticmethod
    def landmarks_to_roi(xy):
        """HandLandmarksToRect: rotation from wrist -> mean of PIPs (index, ring, middle), box in the rotated frame."""
        x0, y0 = xy[0]
        x1, y1 = (xy[6] + xy[14]) / 2 * 0.5 + xy[10] * 0.5
        rot = _norm_angle(0.5 * np.pi - np.arctan2(-(y1 - y0), x1 - x0))
        mid = (xy.min(0) + xy.max(0)) / 2
        ca, sa = np.cos(-rot), np.sin(-rot)
        d = xy - mid
        rx, ry = d[:, 0] * ca - d[:, 1] * sa, d[:, 0] * sa + d[:, 1] * ca
        w, h = rx.max() - rx.min(), ry.max() - ry.min()
        ax, ay = (rx.max() + rx.min()) / 2, (ry.max() + ry.min()) / 2
        cx, cy = ax * np.cos(rot) - ay * np.sin(rot) + mid[0], ax * np.sin(rot) + ay * np.cos(rot) + mid[1]
        return _rect(cx, cy, w, h, rot, 2.0, -0.1)

    # ------------------------------------------------------------ one frame
    def step(self, rgb):
        import time
        hands, new_rois = [], []
        t0 = time.perf_counter()
        for roi in self.rois:
            xy, presence, right = self.landmarks(rgb, roi)
            if presence >= self.presence_conf:
                hands.append({'xy': xy, 'score': presence, 'right': right})
                new_rois.append(self.landmarks_to_roi(xy))
        self.timing['landmark'] = time.perf_counter() - t0
        self.timing['palm'] = 0.0
        if len(new_rois) < self.num_hands:                           # look for more hands, like MediaPipe's VIDEO mode
            t0 = time.perf_counter()
            for d in self.detect_palms(rgb):
                roi = self.palm_to_roi(d)
                box = lambda r: (r[0] - r[2] / 2, r[1] - r[2] / 2, r[0] + r[2] / 2, r[1] + r[2] / 2)
                if len(new_rois) >= self.num_hands or any(_iou(box(roi), box(r)) > 0.5 for r in new_rois):
                    continue
                xy, presence, right = self.landmarks(rgb, roi)
                if presence >= self.presence_conf:
                    hands.append({'xy': xy, 'score': presence, 'right': right})
                    new_rois.append(self.landmarks_to_roi(xy))
            self.timing['palm'] = time.perf_counter() - t0
        self.rois = new_rois
        return hands
