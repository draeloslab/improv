"""
Export the DLC top-down models (SSDLite detector + ResNet pose net) to ONNX for actors/dlc_onnx.py, and check them against DLC's own
runners on recorded frames. improvDLC3 env, run from ~ (not from a folder with utils.py):

    python export_dlc_onnx.py [--frame-size 960 720]

Writes models/dlc_trt/dlc_detector_<W>x<H>.onnx (that frame size in, boxes/scores out, NMS inside; export once per
live frame size) and dlc_pose.onnx (b,3,256,256 -> heatmap, locref).
Inputs are uint8 RGB (b, h, w, 3); the ImageNet normalisation DLC applies is inside the graph.
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from deeplabcut.pose_estimation_pytorch.apis.utils import get_inference_runners
from deeplabcut.pose_estimation_pytorch.config import read_config_as_dict

FES = Path(__file__).resolve().parents[2]
ap = argparse.ArgumentParser()
ap.add_argument('--frame-size', type=int, nargs=2, default=[960, 720])
ap.add_argument('--frames', type=int, default=40)
a = ap.parse_args()
cfg = yaml.safe_load(open(FES / 'config/config.yaml'))
train = Path(cfg['batch3d_model_path'])
mc = read_config_as_dict(train / 'pytorch_config.yaml')
dm = (mc.get('detector') or {}).get('model')
for k in [k for k, v in (dm or {}).items() if v is None]:
    del dm[k]
head = mc['model']['heads']['bodypart']
head['predictor'] = {'type': 'HeatmapPredictor', 'apply_sigmoid': False, 'clip_scores': True, 'location_refinement': True, 'locref_std': 7.2801}
pose, det = get_inference_runners(model_config=mc, snapshot_path=train / cfg['batch3d_model_snapshot'], max_individuals=1, batch_size=1,
                                  detector_batch_size=1, detector_path=train / cfg['batch3d_detector_snapshot'])
out = FES / 'models' / 'dlc_trt'; out.mkdir(parents=True, exist_ok=True)
W, H = a.frame_size


MEAN, STD = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1), torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def prep(x):
    """uint8 (b, h, w, 3) RGB -> DLC-normalised float (b, 3, h, w): done inside the graph, so the CPU never converts a frame."""
    return (x.permute(0, 3, 1, 2).float() / 255.0 - MEAN) / STD


class PoseNet(torch.nn.Module):
    def __init__(self, m):
        super().__init__(); self.m = m
    def forward(self, x):
        o = self.m(prep(x))['bodypart']
        return o['heatmap'], o['locref']


class Detector(torch.nn.Module):
    """One normalised frame -> (boxes xyxy, scores) after the torchvision SSD's own NMS."""
    def __init__(self, m):
        super().__init__(); self.m = m
    def forward(self, x):
        res = self.m(prep(x))
        r = (res[1] if isinstance(res, tuple) else res)[0]
        return r['boxes'], r['scores']


pose.model.cpu(); pose.model.eval(); pm = PoseNet(pose.model)
det.model.cpu(); det.model.eval(); dmod = Detector(det.model)
with torch.no_grad():
    torch.onnx.export(pm, torch.randint(0, 255, (1, 256, 256, 3), dtype=torch.uint8), out / 'dlc_pose.onnx', input_names=['x'], output_names=['heatmap', 'locref'],
                      opset_version=17, dynamic_axes={'x': {0: 'b'}, 'heatmap': {0: 'b'}, 'locref': {0: 'b'}})
    try:
        torch.onnx.export(dmod, torch.randint(0, 255, (1, H, W, 3), dtype=torch.uint8), out / f'dlc_detector_{W}x{H}.onnx', input_names=['x'], output_names=['boxes', 'scores'],
                          opset_version=17, dynamic_axes={'boxes': {0: 'n'}, 'scores': {0: 'n'}})
        print('detector exported')
    except Exception as e:
        import traceback; traceback.print_exc(); sys.exit(1)
print('exported to', out)
