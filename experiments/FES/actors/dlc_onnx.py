"""
DLC top-down inference (SSDLite detector -> crop -> ResNet pose net -> heatmap/locref decode) on ONNX Runtime, without DLC's runners.
Made with scripts/trt/export_dlc_onnx.py (models/dlc_trt). The crop and the decode are DLC's own maths (data/image.py top_down_crop,
models/predictors/single_predictor.py HeatmapPredictor), re-written in numpy so no torch/deeplabcut import is needed at run time.

    net = DlcOnnxTopDown(models_dir, frame_size=(960, 720), providers=[...])
    poses = net.inference([rgb_cam0, rgb_cam1])     # -> list of (K, 3) [x, y, score] in frame pixels, None where no box was found

The detector runs on CUDA (its NMS has data-dependent shapes, which TensorRT cannot take); the pose net can use TensorRT fp16.
Frames must be the size the detector was exported for (frame_size).
"""
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

try:
    from .hand_onnx import prepare_providers
except ImportError:                                   # run as a script from actors/
    from hand_onnx import prepare_providers

CROP = (256, 256)


def top_down_crop(image, bbox, output_size, margin=0):
    """DLC data/image.py top_down_crop with crop_with_context=True, center_padding=False. bbox is xywh."""
    image_h, image_w, c = image.shape
    out_w, out_h = output_size
    x, y, w, h = bbox
    cx, cy = x + w / 2, y + h / 2
    w += 2 * margin
    h += 2 * margin
    ratio_in, ratio_out = w / h, out_w / out_h
    if ratio_in > ratio_out:
        h = w / ratio_out
    elif ratio_in < ratio_out:
        w = h * ratio_out
    x1, y1 = int(round(cx - w / 2)), int(round(cy - h / 2))
    x2, y2 = int(round(cx + w / 2)), int(round(cy + h / 2))
    pad_left = pad_right = pad_top = pad_bottom = 0
    if x1 < 0:
        pad_left, x1 = -x1, 0
    if x2 > image_w:
        pad_right, x2 = x2 - image_w, image_w
    if y1 < 0:
        pad_top, y1 = -y1, 0
    if y2 > image_h:
        pad_bottom, y2 = y2 - image_h, image_h
    w, h = x2 - x1, y2 - y1
    pad_x, pad_y = pad_left + pad_right, pad_top + pad_bottom
    crop = np.zeros((h + pad_y, w + pad_x, c), dtype=image.dtype)
    crop[pad_top:pad_top + h, pad_left:pad_left + w] = image[y1:y2, x1:x2]
    crop = cv2.resize(crop, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
    return crop, (x1 - pad_left, y1 - pad_top), ((w + pad_x) / out_w, (h + pad_y) / out_h)


def decode(heatmap, locref, locref_std=7.2801):
    """HeatmapPredictor (apply_sigmoid False, clip_scores True, location refinement): (b, K, h, w) maps -> (b, K, 3) in crop pixels."""
    b, k, hh, ww = heatmap.shape
    stride = CROP[0] / hh
    flat = heatmap.reshape(b, k, hh * ww)
    top = flat.argmax(2)
    ys, xs = top // ww, top % ww
    score = np.clip(np.take_along_axis(flat, top[..., None], 2)[..., 0], 0, 1)
    lr = locref.reshape(b, k, 2, hh * ww)                     # channel layout (joint, [x, y]) like DLC's reshape(b, h, w, K, 2)
    dx = np.take_along_axis(lr[:, :, 0], top[..., None], 2)[..., 0] * locref_std
    dy = np.take_along_axis(lr[:, :, 1], top[..., None], 2)[..., 0] * locref_std
    return np.stack([xs * stride + 0.5 * stride + dx, ys * stride + 0.5 * stride + dy, score], -1)


class DlcOnnxTopDown:
    def __init__(self, models_dir, frame_size=(960, 720), det_providers=None, pose_providers=None, margin=0, min_score=0.0):
        models_dir = Path(models_dir)
        self.frame_size, self.margin, self.min_score = tuple(frame_size), margin, min_score
        det_p = prepare_providers(det_providers or ['CUDAExecutionProvider', 'CPUExecutionProvider'], models_dir / 'trt_cache')
        pose_p = prepare_providers(pose_providers or ['CUDAExecutionProvider', 'CPUExecutionProvider'], models_dir / 'trt_cache')
        self.det = ort.InferenceSession(str(models_dir / 'dlc_detector.onnx'), providers=det_p)
        self.pose = ort.InferenceSession(str(models_dir / 'dlc_pose.onnx'), providers=pose_p)

    def detect(self, rgb):
        """Best box as xywh in frame pixels, or None."""
        if rgb.shape[1::-1] != self.frame_size:
            raise ValueError(f'frame {rgb.shape[1::-1]} != exported detector size {self.frame_size}')
        boxes, scores = self.det.run(None, {'x': np.ascontiguousarray(rgb)[None]})
        if not len(scores) or scores[0] < self.min_score:
            return None
        x0, y0, x1, y1 = boxes[0]
        return np.array([x0, y0, x1 - x0, y1 - y0], np.float32)

    def inference(self, frames):
        boxes = [self.detect(f) for f in frames]
        live = [i for i, b in enumerate(boxes) if b is not None and b[2] > 1 and b[3] > 1]
        out = [None] * len(frames)
        if not live:
            return out
        crops, meta = [], []
        for i in live:
            crop, off, sc = top_down_crop(frames[i], boxes[i], CROP, self.margin)
            crops.append(crop); meta.append((off, sc))
        heat, loc = self.pose.run(None, {'x': np.stack(crops)})
        poses = decode(heat, loc)
        for j, i in enumerate(live):
            (ox, oy), (sx, sy) = meta[j]
            p = poses[j].copy()
            p[:, 0] = p[:, 0] * sx + ox
            p[:, 1] = p[:, 1] * sy + oy
            out[i] = p
        return out
