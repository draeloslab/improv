"""
Export a one-hand-per-crop DLC hand model (SSDLite detector + RTMPose-m pose net) to ONNX for
actors/dlc_rtmpose_onnx.py. Run in the fes env (it has onnx + onnxruntime-gpu), from ~ (not from a folder with utils.py):

    python <improv>/experiments/FES/scripts/trt/export_dlc_rtmpose_onnx.py \
        --project /path/to/dlc-project [--frame-size 960 720] [--shuffle 1] [--snapshot best]

Writes models/dlc_rtmpose/ (inside experiments/FES):
  dlc_rtmpose_detector_<W>x<H>.onnx   uint8 RGB frame (1, H, W, 3) -> boxes xyxy, scores   (NMS inside; export once per live frame size)
  dlc_rtmpose_detector_<W>x<H>_b<B>.onnx   only with --det-batch B: B frames (B, H, W, 3) in one call -> boxes xyxy, scores, image index
                                      (actors/dlc_rtmpose_onnx.py uses it for the cameras that need the detector in the same step; the
                                      graph is fixed at B frames and shorter batches are padded, so B = number of cameras)
  dlc_rtmpose_pose.onnx               uint8 RGB crops (b, 256, 256, 3) -> simcc_x, simcc_y   (ImageNet normalisation inside)
  meta.json                           everything the runtime needs: decode constants, crop size and margin, snapshots used
and checks both graphs against the PyTorch models on random inputs. The best pose / detector snapshots are used unless
--snapshot names an epoch (e.g. 150).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

FES = Path(__file__).resolve().parents[2]
ap = argparse.ArgumentParser()
ap.add_argument('--project', required=True)
ap.add_argument('--shuffle', type=int, default=1)
ap.add_argument('--frame-size', type=int, nargs=2, default=[960, 720])
ap.add_argument('--snapshot', default='best', help="'best' or an epoch number, for both the pose net and the detector")
ap.add_argument('--det-batch', type=int, default=0, help='also export a detector that takes this many frames per call (0 = no)')
ap.add_argument('--out', default=str(FES / 'models' / 'dlc_rtmpose'))
a = ap.parse_args()
W, H = a.frame_size

project = Path(a.project).expanduser()
pct = int(round(100 * yaml.safe_load((project / 'config.yaml').read_text())['TrainingFraction'][0]))
it = yaml.safe_load((project / 'config.yaml').read_text()).get('iteration', 0)       # config.yaml's iteration
train = next(project.glob(f'dlc-models-pytorch/iteration-{it}/*trainset{pct}shuffle{a.shuffle}/train'))
cfg = yaml.safe_load((train / 'pytorch_config.yaml').read_text())


def pick(prefix):
    if a.snapshot == 'best':
        found = sorted(train.glob(f'{prefix}-best-*.pt'))
    else:
        found = sorted(train.glob(f'{prefix}-{int(a.snapshot):03d}.pt'))
    if not found:
        sys.exit(f"no {prefix} snapshot for {a.snapshot!r} in {train}")
    return found[-1]


pose_snap, det_snap = pick('snapshot'), pick('snapshot-detector')
print('pose snapshot    ', pose_snap.name)
print('detector snapshot', det_snap.name)

from deeplabcut.pose_estimation_pytorch.models import PoseModel                       # noqa: E402
from deeplabcut.pose_estimation_pytorch.models.detectors import DETECTORS             # noqa: E402

model_cfg = cfg['model']
pose = PoseModel.build(model_cfg, pretrained_backbone=False)
pose.load_state_dict(torch.load(pose_snap, map_location='cpu', weights_only=False)['model'])
pose.eval()
det_cfg = {k: v for k, v in cfg['detector']['model'].items() if v is not None}
det = DETECTORS.build(det_cfg)
det.load_state_dict(torch.load(det_snap, map_location='cpu', weights_only=False)['model'])
det.eval()

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def prep(x):
    """uint8 (b, h, w, 3) RGB -> DLC-normalised float (b, 3, h, w), inside the graph so the CPU never converts a frame."""
    return (x.permute(0, 3, 1, 2).float() / 255.0 - MEAN) / STD


class PoseNet(torch.nn.Module):
    def __init__(self, m):
        super().__init__(); self.m = m
    def forward(self, x):
        o = self.m(prep(x))['bodypart']
        return o['x'], o['y']


class Detector(torch.nn.Module):
    """One normalised frame -> (boxes xyxy, scores) after the torchvision SSD's own NMS."""
    def __init__(self, m):
        super().__init__(); self.m = m
    def forward(self, x):
        res = self.m(prep(x))
        r = (res[1] if isinstance(res, tuple) else res)[0]
        return r['boxes'], r['scores']


class BatchDetector(Detector):
    """B normalised frames -> (boxes xyxy, scores, image index) of all the images' detections together, each image after its own NMS."""
    def forward(self, x):
        res = self.m(prep(x))
        rs = res[1] if isinstance(res, tuple) else res
        return (torch.cat([r['boxes'] for r in rs]), torch.cat([r['scores'] for r in rs]),
                torch.cat([torch.zeros_like(r['scores'], dtype=torch.int64) + i for i, r in enumerate(rs)]))      # from scores, not len(): len() is traced as a constant


out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
pm, dm = PoseNet(pose), Detector(det.model if hasattr(det, 'model') else det)
crop_w, crop_h = cfg['data']['inference']['top_down_crop']['width'], cfg['data']['inference']['top_down_crop']['height']
# PyTorch reference BEFORE exporting: tracing leaves the in-memory modules in a different state, so a check made after it
# would compare the file with a modified model
crops = np.random.default_rng(0).integers(0, 255, (4, crop_h, crop_w, 3), dtype=np.uint8)
with torch.no_grad():
    tx, ty = pm(torch.from_numpy(crops))
with torch.no_grad():
    torch.onnx.export(pm, torch.randint(0, 255, (1, crop_h, crop_w, 3), dtype=torch.uint8), out / 'dlc_rtmpose_pose.onnx',
                      input_names=['x'], output_names=['simcc_x', 'simcc_y'], opset_version=17,
                      dynamic_axes={'x': {0: 'b'}, 'simcc_x': {0: 'b'}, 'simcc_y': {0: 'b'}}, dynamo=False)
    torch.onnx.export(dm, torch.randint(0, 255, (1, H, W, 3), dtype=torch.uint8), out / f'dlc_rtmpose_detector_{W}x{H}.onnx',
                      input_names=['x'], output_names=['boxes', 'scores'], opset_version=17,
                      dynamic_axes={'boxes': {0: 'n'}, 'scores': {0: 'n'}}, dynamo=False)
if a.det_batch > 1:
    B = a.det_batch
    bm = BatchDetector(det.model if hasattr(det, 'model') else det)
    with torch.no_grad():
        torch.onnx.export(bm, torch.randint(0, 255, (B, H, W, 3), dtype=torch.uint8), out / f'dlc_rtmpose_detector_{W}x{H}_b{B}.onnx',
                          input_names=['x'], output_names=['boxes', 'scores', 'image'], opset_version=17,
                          dynamic_axes={'boxes': {0: 'n'}, 'scores': {0: 'n'}, 'image': {0: 'n'}}, dynamo=False)
print('exported to', out)

# --- check the graphs against PyTorch on random inputs ---
import onnxruntime as ort                                                              # noqa: E402
sp = ort.InferenceSession(str(out / 'dlc_rtmpose_pose.onnx'), providers=['CPUExecutionProvider'])
ox, oy = sp.run(None, {'x': crops})
print(f"pose net  max |onnx - torch|: x {np.abs(ox - tx.numpy()).max():.2e}  y {np.abs(oy - ty.numpy()).max():.2e}  (logit scale {np.abs(tx.numpy()).max():.1f})")

if a.det_batch > 1:       # the batched detector must give, per image, what the one-frame detector gives for it
    rng = np.random.default_rng(1)
    frames = rng.integers(0, 255, (a.det_batch, H, W, 3), dtype=np.uint8)
    d1 = ort.InferenceSession(str(out / f'dlc_rtmpose_detector_{W}x{H}.onnx'), providers=['CPUExecutionProvider'])
    db = ort.InferenceSession(str(out / f'dlc_rtmpose_detector_{W}x{H}_b{a.det_batch}.onnx'), providers=['CPUExecutionProvider'])
    bb, bs, bi = db.run(None, {'x': frames})
    worst = 0.0
    for i in range(a.det_batch):      # the 10 best boxes of each image (random frames give hundreds of ~0.01 boxes whose count flips at the cut-off)
        b1, s1 = d1.run(None, {'x': frames[i:i + 1]})
        m = bi == i
        o1, ob = np.argsort(-s1)[:10], np.argsort(-bs[m])[:10]
        worst = max(worst, float(np.abs(bb[m][ob] - b1[o1]).max()), float(np.abs(bs[m][ob] - s1[o1]).max()))
    assert worst < 1e-2, f'batched detector differs from the one-frame detector by {worst}'
    print(f"batched detector (x{a.det_batch}) vs one frame at a time, 10 best boxes per random frame: max |diff| {worst:.2e}")

head = model_cfg['heads']['bodypart']
pred = head['predictor']
meta = dict(
    bodyparts=cfg['metadata']['bodyparts'], individuals=cfg['metadata'].get('individuals'),
    crop=[crop_w, crop_h], bbox_margin=cfg['data'].get('bbox_margin', 0), frame_size=[W, H],
    simcc=dict(split_ratio=pred['simcc_split_ratio'], sigma=pred['sigma'], decode_beta=pred['decode_beta'],
               apply_softmax=pred['apply_softmax'], normalize_outputs=pred.get('normalize_outputs', False)),
    pose_snapshot=pose_snap.name, detector_snapshot=det_snap.name, train_folder=str(train), project=str(project))
(out / 'meta.json').write_text(json.dumps(meta, indent=1))
print('wrote', out / 'meta.json')
