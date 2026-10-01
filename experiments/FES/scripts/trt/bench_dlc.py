"""
Would the DLC backend (batch3d_backend: dlc) be faster as raw ONNX / TensorRT? Builds the ResNet-50 pose net (256x256 crops, 42
heatmaps + location refinement) from the model folder, exports it to ONNX and times PyTorch fp32/fp16 vs onnxruntime CUDA / TensorRT fp16
at batch 1 and 2 (two cameras). Network only: the top-down detector and crop/heatmap decoding are not included.  improvDLC3 env:

    python bench_dlc.py [--train-dir ~/HandTrackingVideo1Aug7-trainset80shuffle12/train] [--snapshot snapshot-best-100.pt]
"""
import argparse
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from deeplabcut.pose_estimation_pytorch.config import read_config_as_dict
from deeplabcut.pose_estimation_pytorch.models import PoseModel

ap = argparse.ArgumentParser()
ap.add_argument('--train-dir', default=str(Path.home() / 'HandTrackingVideo1Aug7-trainset80shuffle12' / 'train'))
ap.add_argument('--snapshot', default='snapshot-best-100.pt')
ap.add_argument('--out', default=str(Path(__file__).resolve().parents[2] / 'models' / 'dlc_trt'))
a = ap.parse_args()
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
cfg = read_config_as_dict(Path(a.train_dir) / 'pytorch_config.yaml')
model = PoseModel.build(cfg['model'])
model.load_state_dict(torch.load(Path(a.train_dir) / a.snapshot, map_location='cpu', weights_only=False)['model'])
model = model.eval().cuda()


class Wrap(torch.nn.Module):
    """Pose net -> (heatmaps, locref) tensors only (the dict the DLC head returns is not exportable)."""
    def __init__(self, m):
        super().__init__(); self.m = m
    def forward(self, x):
        o = self.m(x)['bodypart']
        return o['heatmap'], o['locref']


w = Wrap(model).eval()
x1, x2 = torch.randn(1, 3, 256, 256).cuda(), torch.randn(2, 3, 256, 256).cuda()
with torch.no_grad():
    h, l = w(x1); print('heatmap', tuple(h.shape), 'locref', tuple(l.shape))
    torch.onnx.export(w.cpu(), x1.cpu(), out / 'dlc_pose.onnx', input_names=['x'], output_names=['heatmap', 'locref'], opset_version=17,
                      dynamic_axes={'x': {0: 'b'}, 'heatmap': {0: 'b'}, 'locref': {0: 'b'}})
w = w.cuda()


def bench(fn, n=100):
    for _ in range(15):
        fn()
    torch.cuda.synchronize(); t = []
    for _ in range(n):
        t0 = time.perf_counter(); fn(); torch.cuda.synchronize(); t.append((time.perf_counter() - t0) * 1e3)
    return np.median(t)


ort.preload_dlls()
for name, x in (('batch 1', x1), ('batch 2', x2)):
    with torch.no_grad():
        print(f'{name}: PyTorch fp32 {bench(lambda: w(x)):.2f} ms', end='')
        with torch.autocast('cuda', dtype=torch.float16):
            print(f' | PyTorch fp16 autocast {bench(lambda: w(x)):.2f} ms', end='')
    for label, prov in (('ORT CUDA', ['CUDAExecutionProvider']),
                        ('ORT TensorRT fp16', [('TensorrtExecutionProvider', {'trt_fp16_enable': True, 'trt_engine_cache_enable': True,
                                                                              'trt_engine_cache_path': str(out / 'trt_cache'),
                                                                              'trt_profile_min_shapes': 'x:1x3x256x256',
                                                                              'trt_profile_opt_shapes': 'x:2x3x256x256',
                                                                              'trt_profile_max_shapes': 'x:4x3x256x256'}), 'CUDAExecutionProvider'])):
        try:
            s = ort.InferenceSession(str(out / 'dlc_pose.onnx'), providers=prov)
            xin = x.cpu().numpy()
            print(f' | {label} {bench(lambda: s.run(None, {"x": xin})):.2f} ms', end='')
        except Exception as e:
            print(f' | {label} failed: {str(e)[:80]}', end='')
    print()
