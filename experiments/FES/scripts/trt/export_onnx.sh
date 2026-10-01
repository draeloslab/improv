#!/bin/bash
# Rebuild models/hand_trt/*.onnx from the MediaPipe .task bundle. Needs a venv with tensorflow-cpu + tf2onnx (kept separate
# from ~/envs_trt, which only has onnxruntime-gpu/tensorrt/opencv):  python3 -m venv ~/envs_trt_convert && ~/envs_trt_convert/bin/pip install tensorflow-cpu tf2onnx onnx
set -e
TASK=${1:-/home/chesteklab/Desktop/hand-tracking-notebooks-jake-2026-08-17/mediapipe/hand_landmarker.task}
OUT=$(dirname "$0")/../../models/hand_trt
mkdir -p "$OUT" && cd "$OUT" && python3 -c "import zipfile,sys; zipfile.ZipFile(sys.argv[1]).extractall('.')" "$TASK"
for m in hand_detector hand_landmarks_detector; do
  ~/envs_trt_convert/bin/python -m tf2onnx.convert --tflite $m.tflite --output $m.onnx --opset 17
done
