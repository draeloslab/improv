"""
DLC one-hand-per-crop hand model (SSDLite detector -> up to N crops -> RTMPose-m, SimCC decode) on ONNX Runtime, without
DLC's runners. Made with scripts/trt/export_dlc_rtmpose_onnx.py (models/dlc_rtmpose). The crop is DLC's own (data/image.py
top_down_crop, shared with actors/dlc_onnx.py) and the decode is DLC's SimCCPredictor (models/predictors/sim_cc.py),
re-written in numpy so no torch / deeplabcut import is needed at run time.

    net = RTMPoseHands(models_dir, det_score=0.3, max_hands=2, pose_providers=[...])
    hands = net.inference([rgb_cam0, rgb_cam1, ...])     # -> per frame, a list of hands (best detector score first)
    hands[0][0] -> {'xy': (21, 2) frame px (NaN where DLC would drop the point), 'lik': (21,), 'det_score': float, 'box': xywh}

The detector runs on CUDA (its NMS has data-dependent shapes, which TensorRT cannot take); the pose net can use
TensorRT fp16. Frames must be the size the detector was exported for. All crops of all frames go through the pose net in
one batch.
"""
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort

try:
    from .hand_onnx import prepare_providers
    from .dlc_onnx import top_down_crop
except ImportError:                                   # run as a script from actors/
    from hand_onnx import prepare_providers
    from dlc_onnx import top_down_crop


def simcc_decode(sx, sy, split_ratio, sigma, decode_beta, apply_softmax=True):
    """DLC SimCCPredictor (normalize_outputs False): logits (n, K, Wx), (n, K, Wy) -> xy (n, K, 2) in crop pixels, score (n, K).
    Points whose peak probability is not above chance (1 / bins) are dropped, as DLC does (xy NaN, score 0)."""
    sx = sx * (sigma[0] * decode_beta)
    sy = sy * (sigma[1] * decode_beta)
    out = []
    for s in (sx, sy):
        if apply_softmax:
            e = np.exp(s - s.max(-1, keepdims=True))
            peak = 1.0 / e.sum(-1)                              # = max of the softmax
        else:
            peak = s.max(-1)
        out.append((s.argmax(-1), peak))
    (ax, px), (ay, py) = out
    val = np.minimum(px, py)
    xy = np.stack([ax, ay], -1).astype(np.float32) / split_ratio
    bad = val <= (1.0 / sx.shape[-1] if apply_softmax else 0.0)
    xy[bad] = np.nan
    return xy, np.where(bad, 0.0, val)


class RTMPoseHands:
    """Detector + pose net for every frame passed in; one box -> one hand."""

    def __init__(self, models_dir, frame_size=None, det_providers=None, pose_providers=None, det_score=0.3, max_hands=2,
                 margin=None, max_overlap=0.6, pad_batch=0):
        """models_dir: folder with meta.json, dlc_rtmpose_detector_<W>x<H>.onnx and dlc_rtmpose_pose.onnx.
        det_score: detector confidence a box needs to count as a hand (the detector itself reports from 0.01).
        max_hands: boxes kept per frame (best first). margin: pixels added around each box before cropping (default:
        the training config's bbox_margin). max_overlap: of two boxes overlapping more than this (IoU), only the better stays.
        pad_batch: always run the pose net on exactly this many crops (the last crop repeated, extra outputs dropped). With a
        varying batch size the GPU re-plans every time the number of hands changes: ~32 ms instead of ~2 ms (measured 2026-10-15,
        100% of size changes, 18% of steps over a 60 fps budget). Set it to max_hands x number of cameras; 0 = off."""
        models_dir = Path(models_dir)
        self.meta = json.loads((models_dir / 'meta.json').read_text())
        self.frame_size = tuple(frame_size or self.meta['frame_size'])
        self.crop = tuple(self.meta['crop'])
        self.margin = self.meta['bbox_margin'] if margin is None else margin
        self.simcc = self.meta['simcc']
        self.det_score, self.max_hands, self.max_overlap = float(det_score), int(max_hands), float(max_overlap)
        self.pad_batch = int(pad_batch)
        det_p = prepare_providers(det_providers or ['CUDAExecutionProvider', 'CPUExecutionProvider'], models_dir / 'trt_cache')
        pose_p = prepare_providers(pose_providers or ['CUDAExecutionProvider', 'CPUExecutionProvider'], models_dir / 'trt_cache')
        w, h = self.frame_size
        self.det = ort.InferenceSession(str(models_dir / f'dlc_rtmpose_detector_{w}x{h}.onnx'), providers=det_p)
        self.pose = ort.InferenceSession(str(models_dir / 'dlc_rtmpose_pose.onnx'), providers=pose_p)

    def detect(self, rgb):
        """Boxes of the hands in one frame: [(score, (x, y, w, h))], best first, at most max_hands."""
        if rgb.shape[1::-1] != self.frame_size:
            raise ValueError(f'frame {rgb.shape[1::-1]} != exported detector size {self.frame_size}')
        boxes, scores = self.det.run(None, {'x': np.ascontiguousarray(rgb)[None]})
        keep = []
        for i in np.argsort(-scores):
            if scores[i] < self.det_score or len(keep) == self.max_hands:
                break
            x0, y0, x1, y1 = boxes[i]
            if x1 - x0 <= 1 or y1 - y0 <= 1:
                continue
            if any(_iou(boxes[i], boxes[j]) > self.max_overlap for j in keep):
                continue
            keep.append(i)
        return [(float(scores[i]), np.array([boxes[i][0], boxes[i][1], boxes[i][2] - boxes[i][0], boxes[i][3] - boxes[i][1]], np.float32))
                for i in keep]

    def inference(self, frames):
        """RGB frames -> per frame, a list of hand dicts (see the module docstring)."""
        found = [self.detect(f) for f in frames]
        crops, meta = [], []
        for fi, hands in enumerate(found):
            for score, box in hands:
                crop, off, sc = top_down_crop(frames[fi], box, self.crop, self.margin)
                crops.append(crop); meta.append((fi, score, box, off, sc))
        out = [[] for _ in frames]
        if not crops:
            return out
        batch = np.stack(crops)
        n = len(batch)
        if n < self.pad_batch:
            batch = np.concatenate([batch, np.repeat(batch[-1:], self.pad_batch - n, 0)])
        sx, sy = self.pose.run(None, {'x': batch})
        sx, sy = sx[:n], sy[:n]
        xy, lik = simcc_decode(sx, sy, self.simcc['split_ratio'], self.simcc['sigma'], self.simcc['decode_beta'],
                               self.simcc['apply_softmax'])
        for j, (fi, score, box, (ox, oy), (scx, scy)) in enumerate(meta):
            p = xy[j].copy()
            p[:, 0] = p[:, 0] * scx + ox
            p[:, 1] = p[:, 1] * scy + oy
            out[fi].append(dict(xy=p, lik=lik[j], det_score=score, box=box))
        return out


def _iou(a, b):
    """IoU of two xyxy boxes."""
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0
