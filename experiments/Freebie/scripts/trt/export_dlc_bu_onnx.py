"""
Export a bottom-up (single-stage, no detector) DLC model, e.g. the 1-keypoint per-camera models config.yaml model_path_N, to ONNX
for actors/dlc_onnx.py DlcOnnxBottomUp, and check it against the PyTorch net on random frames. fes env, run from ~:

    python export_dlc_bu_onnx.py --camera 2 [--frame-size 960 720]      # model_path_2 / model_snapshot_2 from config.yaml
    python export_dlc_bu_onnx.py --train-dir <.../train> --snapshot snapshot-200.pt --out models/dlc_trt/dlc_bu_cam2.onnx

Writes models/dlc_trt/dlc_bu_cam<N>.onnx: uint8 (b, h, w, 3) frames in the channel order the Processor feeds DLC (no swap) ->
heatmap, locref. The ImageNet normalisation DLC applies is inside the graph; h and w are dynamic, so any frame size works.
Re-export after changing the snapshot in config.yaml.
"""
import argparse
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
import yaml
from deeplabcut.pose_estimation_pytorch.apis.utils import get_inference_runners
from deeplabcut.pose_estimation_pytorch.config import read_config_as_dict

FES = Path(__file__).resolve().parents[2]
ap = argparse.ArgumentParser()
ap.add_argument('--camera', type=int, default=2, help='selects config.yaml model_path_N / model_snapshot_N and the default output name')
ap.add_argument('--train-dir', help='override the model folder (the one with pytorch_config.yaml)')
ap.add_argument('--snapshot', help='override the snapshot file name')
ap.add_argument('--out', help='output .onnx path')
ap.add_argument('--frame-size', type=int, nargs=2, default=[960, 720], help='W H used for the trace and the check')
a = ap.parse_args()

cfg = yaml.safe_load(open(FES / 'config/config.yaml'))
train = Path(a.train_dir or cfg[f'model_path_{a.camera}'])
snap = a.snapshot or cfg[f'model_snapshot_{a.camera}']
out = Path(a.out) if a.out else FES / 'models' / 'dlc_trt' / f'dlc_bu_cam{a.camera}.onnx'
out.parent.mkdir(parents=True, exist_ok=True)
W, H = a.frame_size

mc = read_config_as_dict(train / 'pytorch_config.yaml')
pose, _ = get_inference_runners(model_config=mc, snapshot_path=train / snap, max_individuals=1, batch_size=1,
                                detector_batch_size=1, detector_path=None)

MEAN, STD = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1), torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class PoseNet(torch.nn.Module):
    """uint8 (b, h, w, 3) -> DLC-normalised float -> heatmap (b, K, h/s, w/s), locref (b, 2K, h/s, w/s)."""
    def __init__(self, m):
        super().__init__(); self.m = m
    def forward(self, x):
        x = (x.permute(0, 3, 1, 2).float() / 255.0 - MEAN) / STD
        o = self.m(x)['bodypart']
        return o['heatmap'], o['locref']


pose.model.cpu(); pose.model.eval(); pm = PoseNet(pose.model)
x = torch.randint(0, 255, (1, H, W, 3), dtype=torch.uint8)
with torch.no_grad():
    torch.onnx.export(pm, x, out, input_names=['x'], output_names=['heatmap', 'locref'], opset_version=17,
                      dynamic_axes={'x': {0: 'b', 1: 'h', 2: 'w'}, 'heatmap': {0: 'b', 2: 'hh', 3: 'hw'}, 'locref': {0: 'b', 2: 'hh', 3: 'hw'}})
    ref_h, ref_l = pm(x)
# DLC decodes with the model's fixed stride (not frame size / heatmap size: the heatmap is padded up), so ship it with the model
stride = float(pose.model.get_stride('bodypart'))
locref_std = float(mc['model']['heads']['bodypart']['predictor'].get('locref_stdev', mc['model']['heads']['bodypart']['predictor'].get('locref_std', 7.2801)))
m = onnx.load(str(out))
for k, v in (('stride', stride), ('locref_std', locref_std)):
    e = m.metadata_props.add(); e.key, e.value = k, str(v)
onnx.save(m, str(out))
sess = ort.InferenceSession(str(out), providers=['CPUExecutionProvider'])
oh, ol = sess.run(None, {'x': x.numpy()})
print(f'exported {out}  ({train.name}, {snap})')
print(f'heatmap {tuple(oh.shape)}, model stride {stride:g}, locref_std {locref_std:g}; max |onnx - torch|: '
      f'heatmap {np.abs(oh - ref_h.numpy()).max():.2e}, locref {np.abs(ol - ref_l.numpy()).max():.2e}')
