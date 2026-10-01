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
PALM_SCALE, PALM_SHIFT_Y = 2.6, -0.5       # DetectionsToRects/RectTransformation for a palm detection
TRACK_SCALE, TRACK_SHIFT_Y = 2.0, -0.1     # HandLandmarksToRect for a tracked hand


def _anchors():
    """The palm detector's 2016 SSD anchor centres (normalised x, y): 24x24 cells x 2 + 12x12 cells x 6."""
    out = []
    for fm, per_cell in ((24, 2), (12, 6)):          # strides 8 (2 anchors/cell) and 16 (3 layers merged = 6/cell)
        for y in range(fm):
            for x in range(fm):
                out += [((x + 0.5) / fm, (y + 0.5) / fm)] * per_cell
    return np.array(out, np.float32)


ANCHORS = _anchors()


def _norm_angle(a):
    """Wrap an angle in radians to [-pi, pi)."""
    return a - 2 * np.pi * np.floor((a + np.pi) / (2 * np.pi))


def _rect(cx, cy, w, h, rot, scale, shift_y):
    """MediaPipe RectTransformation with square_long, pixel units: shift (in the rect's own axes), then scale."""
    cx, cy = cx - h * shift_y * np.sin(rot), cy + h * shift_y * np.cos(rot)
    size = max(w, h) * scale
    return cx, cy, size, rot


DEDUP_IOU = 0.3      # landmark-box overlap above which two hands are the same hand


def _landmark_box(xy):
    """(x0, y0, x1, y1) bounding box of a hand's landmarks."""
    return (*xy.min(0), *xy.max(0))


def _same_hand(a, b):
    """Two landmark sets are one hand if their boxes overlap (IoU > DEDUP_IOU) or their palms (wrist + MCPs) are closer
    than half the hand's size (catches weak, squashed duplicates at the frame edge)."""
    if _iou(_landmark_box(a), _landmark_box(b)) > DEDUP_IOU:
        return True
    palm = [0, 5, 9, 13, 17]
    size = max(np.ptp(a, 0).max(), np.ptp(b, 0).max())
    return np.linalg.norm(a[palm].mean(0) - b[palm].mean(0)) < 0.5 * size


def _inside(x, y, box):
    """True if point (x, y) lies inside box (x0, y0, x1, y1)."""
    return box[0] <= x <= box[2] and box[1] <= y <= box[3]


def _iou(a, b):
    """Intersection over union of two (x0, y0, x1, y1) boxes."""
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    iw, ih = max(0, min(ax1, bx1) - max(ax0, bx0)), max(0, min(ay1, by1) - max(ay0, by0))
    inter = iw * ih
    return inter / (((ax1 - ax0) * (ay1 - ay0)) + ((bx1 - bx0) * (by1 - by0)) - inter + 1e-9)


def prepare_providers(providers, cache_dir):
    """
    onnxruntime provider list from config (YAML lists -> tuples). Loads the CUDA libraries that torch's pip packages ship
    (ort.preload_dlls) and, for the TensorRT provider, libnvinfer from the `tensorrt-cu12-libs` pip package, so no
    LD_LIBRARY_PATH is needed. TensorRT engines are built once per model (about a minute) and cached in cache_dir.
    """
    if not providers:
        return ['CPUExecutionProvider']
    names = [p if isinstance(p, str) else p[0] for p in providers]
    if any(n in ('CUDAExecutionProvider', 'TensorrtExecutionProvider') for n in names):
        ort.preload_dlls()
    if 'TensorrtExecutionProvider' in names:
        try:
            import tensorrt_libs  # noqa: F401   (loads libnvinfer*.so on import)
        except ImportError:
            pass
    out = []
    for p in providers:
        if isinstance(p, str):
            out.append(p)
        else:
            opts = dict(p[1] if len(p) > 1 else {})
            if p[0] == 'TensorrtExecutionProvider':
                opts.setdefault('trt_engine_cache_enable', True)
                opts.setdefault('trt_engine_cache_path', str(cache_dir))
            out.append((p[0], opts))
    return out


class OnnxHandTracker:
    """One camera's hand tracker: MediaPipe's palm detector + landmark net on onnxruntime, with MediaPipe's VIDEO-mode
    tracking (the palm detector only runs while fewer than num_hands hands are being followed)."""

    def __init__(self, models_dir, num_hands=2, det_conf=0.5, presence_conf=0.5, providers=None, palm_range=(0.0, 1.0), always_detect=False):
        """
        models_dir: folder with hand_detector.onnx and hand_landmarks_detector.onnx.
        num_hands: hands to follow per camera. det_conf / presence_conf: MediaPipe's min_hand_detection_confidence /
        min_hand_presence_confidence. providers: onnxruntime providers (see prepare_providers); None = CPU.
        palm_range: input value range of the palm detector (0..1 for this model). always_detect: no tracking.
        """
        models_dir = Path(models_dir)
        providers = prepare_providers(providers, models_dir / 'trt_cache')
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 2
        self.palm = ort.InferenceSession(str(models_dir / 'hand_detector.onnx'), opts, providers=providers)
        self.lm = ort.InferenceSession(str(models_dir / 'hand_landmarks_detector.onnx'), opts, providers=providers)
        self.num_hands, self.det_conf, self.presence_conf, self.palm_range = num_hands, det_conf, presence_conf, palm_range
        self.always_detect = always_detect     # ignore tracking, detect every frame (to compare with MediaPipe's IMAGE mode)
        self.rois = []            # tracked (cx, cy, size, rot) per hand, pixels
        self.timing = {'palm': 0.0, 'landmark': 0.0}

    # ------------------------------------------------------------ palm detector
    def detect_palms(self, rgb):
        """Palm detections in one RGB frame, best first: [{'c', 'wh', 'k0' (wrist), 'k2' (middle MCP), 'score'}] in pixels.

        Letterboxes the frame to 192x192, decodes the SSD boxes/keypoints against ANCHORS and merges overlapping
        candidates with MediaPipe's weighted NMS (IoU 0.3).
        """
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
        keep = keep[np.argsort(-sc[keep])[:48]]      # best candidates only: NMS below is O(n^2)
        r, a, score = reg[keep], ANCHORS[keep], sc[keep]
        cx, cy = r[:, 0] / PALM_IN + a[:, 0], r[:, 1] / PALM_IN + a[:, 1]
        bw, bh = r[:, 2] / PALM_IN, r[:, 3] / PALM_IN
        kps = r[:, 4:18].reshape(-1, 7, 2) / PALM_IN + a[:, None, :]
        boxes = np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], 1)
        # weighted NMS like MediaPipe: each surviving detection is the score-weighted mean of every candidate overlapping it
        ix0, iy0 = np.maximum(boxes[:, None, 0], boxes[None, :, 0]), np.maximum(boxes[:, None, 1], boxes[None, :, 1])
        ix1, iy1 = np.minimum(boxes[:, None, 2], boxes[None, :, 2]), np.minimum(boxes[:, None, 3], boxes[None, :, 3])
        inter = np.clip(ix1 - ix0, 0, None) * np.clip(iy1 - iy0, 0, None)
        area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        iou = inter / (area[:, None] + area[None, :] - inter + 1e-9)
        remaining, out = np.ones(len(score), bool), []
        to_px = lambda p: np.array([(p[0] * PALM_IN - padx) / s, (p[1] * PALM_IN - pady) / s])
        for i in range(len(score)):                                  # candidates are sorted by score
            if not remaining[i]:
                continue
            grp = remaining & (iou[i] > 0.3)
            wgt = score[grp] / score[grp].sum()
            remaining &= ~grp
            c = np.array([(wgt * cx[grp]).sum(), (wgt * cy[grp]).sum()])
            wh = np.array([(wgt * bw[grp]).sum(), (wgt * bh[grp]).sum()])
            k = (wgt[:, None, None] * kps[grp]).sum(0)
            out.append({'c': to_px(c), 'wh': wh * PALM_IN / s, 'k0': to_px(k[0]), 'k2': to_px(k[2]), 'score': float(score[i])})
            if len(out) >= self.num_hands:
                break
        return out

    @staticmethod
    def palm_to_roi(d):
        """Palm detection -> rotated square hand ROI (cx, cy, size, rotation): wrist->middle MCP points up, 2.6x the palm box."""
        (x0, y0), (x2, y2) = d['k0'], d['k2']
        rot = _norm_angle(0.5 * np.pi - np.arctan2(-(y2 - y0), x2 - x0))
        return _rect(d['c'][0], d['c'][1], d['wh'][0], d['wh'][1], rot, PALM_SCALE, PALM_SHIFT_Y)

    # ------------------------------------------------------------ landmark net
    def landmarks(self, rgb, roi):
        """Run the landmark net on the rotated 224x224 crop of roi -> (21 x/y in frame pixels, presence, P(right hand))."""
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
        return _rect(cx, cy, w, h, rot, TRACK_SCALE, TRACK_SHIFT_Y)

    # ------------------------------------------------------------ one frame
    def step(self, rgb):
        """One frame: follow the tracked hands, look for new ones if needed.

        A hand is kept only if its landmark box does not overlap an already accepted hand's (IoU > DEDUP_IOU), like
        MediaPipe's own de-duplication. Without it, the palm detector -- which runs while fewer than num_hands hands are
        followed -- re-finds the hand already being tracked and that duplicate takes the second hand's slot.

        Returns [{'xy': (21, 2) pixels, 'score': presence, 'right': handedness probability}, ...].
        """
        import time
        hands, new_rois = [], []

        def accept(xy, presence, right):
            if presence < self.presence_conf:
                return False
            if any(_same_hand(xy, h['xy']) for h in hands):
                return False
            hands.append({'xy': xy, 'score': presence, 'right': right})
            new_rois.append(self.landmarks_to_roi(xy))
            return True

        t0 = time.perf_counter()
        tracked = [self.landmarks(rgb, roi) for roi in ([] if self.always_detect else self.rois)]
        for xy, presence, right in sorted(tracked, key=lambda h: -h[1]):     # most confident first wins an overlap
            accept(xy, presence, right)
        self.timing['landmark'] = time.perf_counter() - t0
        self.timing['palm'] = 0.0
        if len(hands) < self.num_hands:                              # look for more hands, like MediaPipe's VIDEO mode
            t0 = time.perf_counter()
            for d in self.detect_palms(rgb):
                if len(hands) >= self.num_hands:
                    break
                cx, cy = d['c']
                if any(_inside(cx, cy, _landmark_box(h['xy'])) for h in hands):
                    continue                                         # a palm inside a hand we already have
                accept(*self.landmarks(rgb, self.palm_to_roi(d)))
            self.timing['palm'] = time.perf_counter() - t0
        self.rois = new_rois
        return hands

    def close(self):
        """Nothing to release (sessions are freed with the object); here so callers can treat it like a landmarker."""
