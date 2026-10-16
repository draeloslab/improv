"""
DLC one-hand-per-crop hand model (SSDLite detector -> up to N crops -> RTMPose-m, SimCC decode) on ONNX Runtime, without
DLC's runners. Made with scripts/trt/export_dlc_rtmpose_onnx.py (models/dlc_rtmpose). The crop is DLC's own (data/image.py
top_down_crop, shared with actors/dlc_onnx.py) and the decode is DLC's SimCCPredictor (models/predictors/sim_cc.py),
re-written in numpy so no torch / deeplabcut import is needed at run time.

    net = RTMPoseHands(models_dir, det_score=0.3, max_hands=2, pose_providers=[...], track=True)
    hands = net.inference([rgb_cam0, rgb_cam1, ...], keys=[0, 1, ...])   # -> per frame, a list of hands
    hands[0][0] -> {'xy': (21, 2) frame px (NaN where DLC would drop the point), 'lik': (21,), 'det_score': float, 'box': xywh,
                    'track_id': int, 'tracked': bool (this crop came from the previous frame's keypoints, not the detector)}

One step is: detector on the cameras that need it (one batched call), then ONE pose-net pass over every crop of every camera.
With track=True (the live setting) the detector is skipped for a camera while it holds a hand: the next crop is the box around the
hand's own keypoints from the previous step (grown by track_pad), like MediaPipe's landmark-driven ROI. The detector then runs only
  - on a camera with no live track (every step, until a hand shows up),
  - every `redetect_every` steps on a camera holding fewer than max_hands (to find the second hand),
  - every `refresh_every` steps on any camera, if set (re-anchors drift),
and a track ends when its keypoints fall below `min_keypoints` at `min_likelihood` (the camera then goes back to the detector on
the next step). Tracks give each hand a stable `track_id` and a stable order in the camera's list.

The detector runs on CUDA (its NMS has data-dependent shapes, which TensorRT cannot take); the pose net can use
TensorRT fp16. Frames must be the size the detector was exported for. If models_dir also has a batched detector
(dlc_rtmpose_detector_<W>x<H>_b<B>.onnx, export_dlc_rtmpose_onnx.py --det-batch B), the cameras that need detection go through it in
one call; otherwise they run one by one.
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
                 margin=None, max_overlap=0.6, pad_batch=0, track=False, redetect_every=15, refresh_every=0, track_pad=0.1,
                 min_keypoints=8, min_likelihood=0.3, match_iou=0.3):
        """models_dir: folder with meta.json, dlc_rtmpose_detector_<W>x<H>.onnx and dlc_rtmpose_pose.onnx.
        det_score: detector confidence a box needs to count as a hand (the detector itself reports from 0.01).
        max_hands: boxes kept per frame (best first). margin: pixels added around each box before cropping (default:
        the training config's bbox_margin). max_overlap: of two boxes overlapping more than this (IoU), only the better stays.
        pad_batch: always run the pose net on exactly this many crops (the last crop repeated, extra outputs dropped). With a
        varying batch size the GPU re-plans every time the number of hands changes: ~32 ms instead of ~2 ms (measured 2026-10-15,
        100% of size changes, 18% of steps over a 60 fps budget). Set it to max_hands x number of cameras; 0 = off.
        track: follow each hand from its previous keypoints instead of running the detector every step (see the module
        docstring). redetect_every / refresh_every: steps between detector runs on a camera that holds fewer than max_hands hands /
        on any camera (0 = never). track_pad: fraction of the box's longer side added on every side of a tracked crop, on top of
        margin (the hand moves between steps). min_keypoints / min_likelihood: a track survives while this many keypoints reach
        this likelihood. match_iou: a detector box this close to a track is that track."""
        models_dir = Path(models_dir)
        self.meta = json.loads((models_dir / 'meta.json').read_text())
        self.frame_size = tuple(frame_size or self.meta['frame_size'])
        self.crop = tuple(self.meta['crop'])
        self.margin = self.meta['bbox_margin'] if margin is None else margin
        self.simcc = self.meta['simcc']
        self.det_score, self.max_hands, self.max_overlap = float(det_score), int(max_hands), float(max_overlap)
        self.pad_batch = int(pad_batch)
        self._init_tracking(track, redetect_every, refresh_every, track_pad, min_keypoints, min_likelihood, match_iou)
        det_p = prepare_providers(det_providers or ['CUDAExecutionProvider', 'CPUExecutionProvider'], models_dir / 'trt_cache')
        pose_p = prepare_providers(pose_providers or ['CUDAExecutionProvider', 'CPUExecutionProvider'], models_dir / 'trt_cache')
        w, h = self.frame_size
        self.det = ort.InferenceSession(str(models_dir / f'dlc_rtmpose_detector_{w}x{h}.onnx'), providers=det_p)
        self.pose = ort.InferenceSession(str(models_dir / 'dlc_rtmpose_pose.onnx'), providers=pose_p)
        self.det_b, self.det_batch = None, 1                     # optional batched detector: (B, H, W, 3) -> boxes, scores, image index
        found = sorted(models_dir.glob(f'dlc_rtmpose_detector_{w}x{h}_b*.onnx'))
        if found:
            self.det_b = ort.InferenceSession(str(found[-1]), providers=det_p)
            self.det_batch = int(self.det_b.get_inputs()[0].shape[0])

    def _init_tracking(self, track, redetect_every, refresh_every, track_pad, min_keypoints, min_likelihood, match_iou):
        self.track, self.redetect_every, self.refresh_every = bool(track), int(redetect_every), int(refresh_every)
        self.track_pad, self.min_keypoints, self.min_likelihood = float(track_pad), int(min_keypoints), float(min_likelihood)
        self.match_iou = float(match_iou)
        self._tracks, self._since, self._next_id = {}, {}, 0
        self.stats = dict(steps=0, frames=0, detector_frames=0, detector_calls=0, tracked_crops=0, detected_crops=0,
                          lost_tracks=0)

    def reset(self, key=None):
        """Forget the tracks of one camera (key) or of all."""
        for d in (self._tracks, self._since):
            if key is None:
                d.clear()
            else:
                d.pop(key, None)

    # ---- detector ----
    def _raw_detect(self, frames):
        """Detector output for each frame: [(boxes xyxy (n, 4), scores (n,))]. One call through the batched detector if there is one
        (padded to its batch size with the first frame; more frames than that go in several calls), else one call per frame."""
        for f in frames:
            if f.shape[1::-1] != self.frame_size:
                raise ValueError(f'frame {f.shape[1::-1]} != exported detector size {self.frame_size}')
        if len(frames) > 1 and self.det_b is not None:
            out, B = [], self.det_batch
            for s in range(0, len(frames), B):
                chunk = list(frames[s:s + B])
                n = len(chunk)
                boxes, scores, idx = self.det_b.run(None, {'x': np.stack(chunk + [chunk[0]] * (B - n))})
                self.stats['detector_calls'] += 1
                out += [(boxes[idx == i], scores[idx == i]) for i in range(n)]
            return out
        self.stats['detector_calls'] += len(frames)
        return [tuple(self.det.run(None, {'x': np.ascontiguousarray(f)[None]})) for f in frames]

    def _filter_boxes(self, boxes, scores):
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

    def detect(self, rgb):
        """Boxes of the hands in one frame: [(score, (x, y, w, h))], best first, at most max_hands."""
        return self._filter_boxes(*self._raw_detect([rgb])[0])

    # ---- tracks ----
    def _needs_detect(self, tracks, since):
        if not tracks:
            return True
        if len(tracks) < self.max_hands and since >= self.redetect_every:
            return True
        return bool(self.refresh_every) and since >= self.refresh_every

    def _merge(self, tracks, dets):
        """Detector boxes (best first) into a camera's tracks: a box on a track re-anchors it, a free box starts a track."""
        matched = set()
        for score, box in dets:
            best, best_iou = None, self.match_iou
            for j, t in enumerate(tracks):
                if j not in matched:
                    v = _iou(_xyxy(box), _xyxy(t['box']))
                    if v >= best_iou:
                        best, best_iou = j, v
            if best is not None:
                matched.add(best)
                tracks[best].update(box=box, src='det', det_score=score)
            elif len(tracks) < self.max_hands:
                tracks.append(dict(id=self._next_id, box=box, src='det', det_score=score))
                self._next_id += 1

    def _crop_box(self, t):
        x, y, w, h = t['box']
        if t['src'] == 'det':
            return np.array([x, y, w, h], np.float32)
        p = self.track_pad * max(w, h)
        return np.array([x - p, y - p, w + 2 * p, h + 2 * p], np.float32)

    # ---- pose ----
    def _pose(self, batch):
        """(n, h, w, 3) crops -> xy (n, K, 2) crop px, lik (n, K). One pass; padded to pad_batch so the batch size never changes."""
        n = len(batch)
        if n < self.pad_batch:
            batch = np.concatenate([batch, np.repeat(batch[-1:], self.pad_batch - n, 0)])
        sx, sy = self.pose.run(None, {'x': batch})
        return simcc_decode(sx[:n], sy[:n], self.simcc['split_ratio'], self.simcc['sigma'], self.simcc['decode_beta'],
                            self.simcc['apply_softmax'])

    def inference(self, frames, keys=None):
        """RGB frames -> per frame, a list of hand dicts (see the module docstring). keys: one hashable id per frame (the camera
        slot) so the tracks follow a camera when the frames passed change from step to step; default = position in `frames`."""
        keys = list(range(len(frames))) if keys is None else list(keys)
        tracks, since = (self._tracks, self._since) if self.track else ({}, {})
        per_frame = []
        for k in keys:
            if k not in since:      # stagger the periodic detector runs so all cameras do not do them in the same step
                since[k] = (3 * k) % max(self.redetect_every, 1) if isinstance(k, int) else 0
            per_frame.append(tracks.setdefault(k, []))
        self.stats['steps'] += 1
        self.stats['frames'] += len(frames)

        # 1. detector, only for the cameras that need it, all in one batched call
        need = [i for i, k in enumerate(keys) if self._needs_detect(per_frame[i], since[k])]
        if need:
            self.stats['detector_frames'] += len(need)
            for i, (boxes, scores) in zip(need, self._raw_detect([frames[i] for i in need])):
                self._merge(per_frame[i], self._filter_boxes(boxes, scores))
                since[keys[i]] = 0
        for i, k in enumerate(keys):
            if i not in need:
                since[k] += 1

        # 2. crops of every hand in every camera
        crops, meta = [], []
        for fi, tr in enumerate(per_frame):
            for t in tr:
                box = self._crop_box(t)
                crop, off, sc = top_down_crop(frames[fi], box, self.crop, self.margin)
                crops.append(crop); meta.append((fi, t, box, off, sc, t['src'] == 'kp'))
        out = [[] for _ in frames]
        if not crops:
            return out

        # 3. ONE pose pass over all of them
        xy_all, lik_all = self._pose(np.stack(crops))

        # 4. back to frame pixels; tracks follow their keypoints
        alive = [[] for _ in frames]
        for j, (fi, t, box, (ox, oy), (scx, scy), was_tracked) in enumerate(meta):
            p = xy_all[j].copy()
            p[:, 0] = p[:, 0] * scx + ox
            p[:, 1] = p[:, 1] * scy + oy
            keep = np.isfinite(p).all(1) & (lik_all[j] >= self.min_likelihood)
            if self.track and keep.sum() < self.min_keypoints:
                self.stats['lost_tracks'] += 1
                continue                                          # the track ends; this camera uses the detector next step
            if self.track:
                lo, hi = p[keep].min(0), p[keep].max(0)
                t['box'], t['src'] = np.array([lo[0], lo[1], hi[0] - lo[0], hi[1] - lo[1]], np.float32), 'kp'
            self.stats['tracked_crops' if was_tracked else 'detected_crops'] += 1
            alive[fi].append((t, dict(xy=p, lik=lik_all[j], det_score=t['det_score'], box=box, track_id=t['id'], tracked=was_tracked),
                              float(lik_all[j][keep].mean()) if keep.any() else 0.0))
        for fi, items in enumerate(alive):
            if self.track:                                        # two tracks on one hand: the better stays
                items.sort(key=lambda it: -it[2])
                kept = []
                for it in items:
                    if all(_iou(_xyxy(it[0]['box']), _xyxy(o[0]['box'])) <= self.max_overlap for o in kept):
                        kept.append(it)
                keep_ids = {id(it[0]) for it in kept}
                per_frame[fi][:] = [t for t in per_frame[fi] if id(t) in keep_ids]
                order = {id(t): n for n, t in enumerate(per_frame[fi])}      # track order = stable hand order
                items = sorted((it for it in items if id(it[0]) in keep_ids), key=lambda it: order[id(it[0])])
            out[fi] = [it[1] for it in items]
        return out


def _xyxy(b):
    return (b[0], b[1], b[0] + b[2], b[1] + b[3])


def _iou(a, b):
    """IoU of two xyxy boxes."""
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0
