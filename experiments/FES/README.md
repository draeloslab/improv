# FES hand tracking on improv

Real-time multi-camera hand tracking for functional electrical stimulation (FES) experiments, built from
[improv](../../README.md) actors. USB cameras -> pose estimation (MediaPipe or DeepLabCut) -> 3D triangulation ->
joint angles -> UDP to BRAND, or UART to an xPC target.

```
CameraReader x N ──> ProcessorBatch3D ──> SenderUDP ──UDP──> BRAND
 (or Generator          2D pose, cross-camera association,
  for replay)           triangulation, smoothing, joint angles
                     └──> VideoScreen3D (GUI)
```

## Layout

```
config/    config.yaml (processing, documented inline), camera_config.yaml (camera serials, exposure), video_config.yaml
graphs/    improv graphs: which actors run and how they are wired
actors/    the improv actors; processor_batch3d.py is the core 3D pipeline
scripts/   fes-run.sh (launcher), fes_graph.py (graph expansion), trt/ (ONNX / TensorRT export and benchmarks)
```

## Install

System packages (Ubuntu): [tiscamera](https://github.com/TheImagingSource/tiscamera/releases), GStreamer, redis, ffmpeg.

```bash
sudo apt install gstreamer1.0-tools gstreamer1.0-plugins-base gstreamer1.0-plugins-good \
                 gir1.2-gstreamer-1.0 libgirepository1.0-dev libcairo2-dev redis-server ffmpeg
conda env create -f experiments/FES/environment.yml    # from the improv repo root; env "fes"
conda activate fes
```

The GPU engines need an NVIDIA driver >= 550 (CUDA 12.9).

Then set your local paths in `config/config.yaml` (`output_path`, `mediapipe_model_path`, `batch3d_model_path`,
`calibration_toml`, `video_paths`) and your camera serials (`tcam-ctrl -l`) in `config/camera_config.yaml`.

For the ONNX / TensorRT engines (the default), export the models once:

```bash
scripts/trt/export_onnx.sh /path/to/hand_landmarker.task      # MediaPipe -> models/hand_trt/
python scripts/trt/export_dlc_onnx.py --frame-size 960 720     # DeepLabCut -> models/dlc_trt/
python scripts/trt/export_dlc_rtmpose_onnx.py --project /path/to/dlc-project   # one-hand DLC model -> models/dlc_rtmpose/
```

## Run

```bash
cd experiments/FES
export FES_CONDA_ENV=fes
scripts/fes-run.sh mediapipe_live.yaml --cameras 0,1,2   # live; opens improv's TUI (setup, run, stop, quit)
scripts/fes-run.sh --auto mediapipe_gen.yaml             # replay a recording, headless
scripts/fes-run.sh --dry-run dlc_live.yaml               # preflight checks only
```

Set `SENDER_UDP_IP` / `SENDER_UDP_PORT` for the UDP destination. Add `--save-video` to record raw video.

| Graph | Input | Pose | Output |
|---|---|---|---|
| `mediapipe_live`, `dlc_live` | live cameras | MediaPipe / DLC | GUI, UDP |
| `mediapipe_gen`, `dlc_hand_gen`, `dlc_rtmpose_gen` | recorded video | MediaPipe / DLC | GUI, UDP |
| `bo_stim_live` | live cameras | DLC | Bayesian-optimised stimulation requests |
| `4cam_noproc`, `5cam_test`, `7cam_test` | live cameras | none | raw video |
| `1dof_dlc_udp`, `2dof_*`, `latency_benchmarking` | live cameras | 2D DLC per camera | UART / UDP to xPC |

A graph can override any `config.yaml` key with `config_overrides:`; set `mediapipe_engine: mediapipe` or
`dlc_engine: pytorch` there to use the stock engines instead of ONNX.

## Output

Each run writes to `<output_path>/<YYYYMMDD>/<YYYYMMDD-HHMM>/`: `logs/`, per-frame camera timestamps (`TIS*.npy`),
2D / 3D keypoints and joint angles (`batch3d_*.npy`), and what was sent (`sender_*.npy`). If a run is killed,
`python -m actors.chunk_log <run folder>` reassembles its logs.
