"""
Pure model time of the two MediaPipe hand networks (palm detector 192x192, landmark net 224x224) as ONNX on
onnxruntime's CPU / CUDA / TensorRT providers, one image per call, 2 calls per step (= 2 cameras). No pre/post-processing:
this is the ceiling for what moving the networks to the GPU can save. Run in ~/envs_trt:

    python bench_onnx.py [--models ../../models/hand_trt]
"""
import argparse
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

ap = argparse.ArgumentParser()
ap.add_argument('--models', default=str(Path(__file__).resolve().parents[2] / 'models' / 'hand_trt'))
ap.add_argument('--iters', type=int, default=300)
a = ap.parse_args()

PROVIDERS = {
    'CPU': ['CPUExecutionProvider'],
    'CUDA': ['CUDAExecutionProvider'],
    'TensorRT fp16': [('TensorrtExecutionProvider', {'trt_fp16_enable': True, 'trt_engine_cache_enable': True,
                                                      'trt_engine_cache_path': str(Path(a.models) / 'trt_cache')}),
                      'CUDAExecutionProvider'],
}
for name in ('hand_detector', 'hand_landmarks_detector'):
    for label, prov in PROVIDERS.items():
        sess = ort.InferenceSession(str(Path(a.models) / f'{name}.onnx'), providers=prov)
        inp = sess.get_inputs()[0]
        x = np.random.rand(*[d if isinstance(d, int) else 1 for d in inp.shape]).astype(np.float32)
        for _ in range(20):
            sess.run(None, {inp.name: x})
        t = []
        for _ in range(a.iters):
            t0 = time.perf_counter()
            sess.run(None, {inp.name: x})
            t.append((time.perf_counter() - t0) * 1e3)
        print(f'{name:24s} {str(inp.shape):22s} {label:14s} median {np.median(t):6.2f} ms  p95 {np.percentile(t, 95):6.2f} ms  '
              f'(active: {sess.get_providers()[0]})')
