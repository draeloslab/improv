# FES hand tracking on improv

Real-time hand tracking for the FES experiments: TIS USB cameras -> pose estimation (MediaPipe or DeepLabCut) ->
3D triangulation -> joint angles -> UDP to BRAND (hand-control task, closed-loop stimulation) or UART to the xPC.
Everything runs as [improv](../../README.md) actors wired together by a graph YAML.

```
CameraReader x N ──> ProcessorBatch3D ──> SenderUDP ──UDP──> BRAND (joint angles, 21 3D keypoints, stim requests)
   (or Generator      │   pose estimation, all cameras
    for replay)       │   cross-camera association
                      │   triangulation + smoothing
                      │   joint angles
                      └──> VideoScreen3D (GUI)        BayesOptStim <── BrandReceiver (closed-loop stimulation)
```

## Contents

1. [Repository layout](#repository-layout)
2. [Install](#install)
3. [Running](#running)
4. [What happens to the data, live](#what-happens-to-the-data-live)
5. [The config, block by block](#the-config-block-by-block)
6. [Fast engines: ONNX Runtime / TensorRT](#fast-engines-onnx-runtime--tensorrt)
7. [What a run saves](#what-a-run-saves)
8. [Troubleshooting](#troubleshooting)

## Repository layout

```
experiments/FES/
├── environment.yml          conda env (python 3.10, mediapipe, DLC, onnxruntime/TensorRT, ...)
├── requirements.txt         the same pins for pip
├── config/
│   ├── config.yaml          everything the processing actors read (documented inline)
│   ├── camera_config.yaml   camera list (serials), resolution, exposure/gain/white balance
│   └── video_config.yaml    raw-video buffering for the VideoSavers
├── graphs/                  improv graphs: which actors run and how they are wired (see "Graphs" below)
├── actors/
│   ├── camera_reader.py     CameraReader: one live camera -> store            (uses TIS.py, the GStreamer pipeline)
│   ├── generator.py         Generator: a recorded video -> store (replay, no cameras)
│   ├── processor_batch3d.py ProcessorBatch3D: N cameras -> 2D -> 3D -> joint angles   <- the core
│   ├── hand_onnx.py         MediaPipe's hand pipeline on ONNX Runtime (mediapipe_engine: onnx)
│   ├── dlc_onnx.py          DLC top-down inference on ONNX Runtime (dlc_engine: onnx)
│   ├── hand_assoc3d.py      geometric cross-camera hand association + One-Euro 3D smoothing
│   ├── continuity.py        hold-and-glide on the final 3D points
│   ├── held_hand3d.py       "region" association: one hand held in a fixed place (DLC)
│   ├── kalmanfilter.py      2D Kalman smoothing (label association, and the 2D Processor)
│   ├── sender_udp.py        SenderUDP: angles / keypoints / stim requests -> UDP
│   ├── bayes_opt.py, stim_gp.py, brand_link.py    closed-loop stimulation (BO over stimulation parameters)
│   ├── video_screen_3d.py, front_end_3d.py        the 3D GUI
│   ├── video_saver.py, video_converter.py         raw video recording, buffer -> mp4
│   ├── cpu_affinity.py      pins every actor to P-/E-cores
│   ├── run_paths.py, chunk_log.py                 run folder, logging, crash-safe logs
│   └── processor.py, sender.py, recieverActor.py, video_screen.py, front_end5.py   the 2D xPC pipeline
├── scripts/
│   ├── fes-run.sh           start a run (picks the run id, activates the env, preflight, launches improv)
│   ├── improv_drive.py      headless driver used by fes-run.sh --auto
│   ├── setup_realtime.sh    CPU governor / C-states / camera IRQ placement for latency work
│   └── trt/                 export + benchmark + validation for the ONNX / TensorRT engines
└── models/                  (not in git) exported ONNX models and TensorRT engine caches
```

Not in git (local to the rig): `calibration/` (anipose calibrations), `analysis/`, `notebooks/`, `models/`.

## Install

**1. System packages** (Ubuntu): the TIS camera driver and GStreamer, redis and ffmpeg.

```bash
# tiscamera: https://github.com/TheImagingSource/tiscamera/releases (install the .deb for your Ubuntu)
sudo apt install gstreamer1.0-tools gstreamer1.0-plugins-base gstreamer1.0-plugins-good \
                 gir1.2-gstreamer-1.0 libgirepository1.0-dev libcairo2-dev redis-server ffmpeg
tcam-ctrl -l                       # should list every camera's serial
```

An NVIDIA driver >= 550 is needed for the CUDA 12.9 wheels (the TensorRT engine is optional).

**2. Python environment**, from the repo root:

```bash
conda env create -f experiments/FES/environment.yml      # creates "fes", installs improv itself editable
conda activate fes
export FES_CONDA_ENV=fes                                  # fes-run.sh activates this env (default: improvDLC3)
```

**3. Local paths.** Edit `config/config.yaml`: `output_path`, `mediapipe_model_path` (MediaPipe's
`hand_landmarker.task`), `calibration_toml`, `batch3d_model_path` (DLC), `video_paths` (replay). Edit
`config/camera_config.yaml` `active_cameras` to your cameras' serials (`tcam-ctrl -l`).

**4. ONNX models** (only for the fast engines):

```bash
cd experiments/FES
# MediaPipe: extract the two networks from hand_landmarker.task and convert them (needs a venv with tf2onnx,
# kept apart from the run env because TensorFlow's numpy/protobuf pins clash with mediapipe's):
python3 -m venv ~/envs_trt_convert && ~/envs_trt_convert/bin/pip install tensorflow-cpu tf2onnx onnx
scripts/trt/export_onnx.sh                                   # -> models/hand_trt/*.onnx
# DLC: export the detector (per live frame size) and the pose net
python scripts/trt/export_dlc_onnx.py --frame-size 960 720   # -> models/dlc_trt/
```

**5. Check it.**

```bash
python scripts/trt/bench_onnx_tracker.py VIDEO_A.mp4 VIDEO_B.mp4 --providers trt   # ms per 2-camera step
python scripts/trt/validate_dlc_onnx.py                                            # ONNX DLC vs DLC's runners
```

## Running

Every run goes through `scripts/fes-run.sh`: it picks one run id, activates the env, checks cameras / redis / the
xPC link, and launches improv. Every actor then writes its logs and data to `<output_path>/<YYYYMMDD>/<run id>/`.

```bash
cd ~/improv/experiments/FES
scripts/fes-run.sh mediapipe_live_onnx.yaml      # bare names resolve from graphs/; opens improv's TUI
scripts/fes-run.sh --auto mediapipe_gen.yaml     # headless: ENTER to start, ENTER to stop
scripts/fes-run.sh --dry-run 7cam_3d.yaml        # preflight only
```

In the TUI: `setup` (actors load models, cameras open), `run` (data flows), `stop`, `quit`. Closing the GUI window
also stops the run. Add `alias fes-run=~/improv/experiments/FES/scripts/fes-run.sh` to `~/.bashrc` to save typing.

### Examples

```bash
# Replay: MediaPipe on four recorded cameras, no hardware. Compare the stock and the ONNX/TensorRT engine:
scripts/fes-run.sh mediapipe_gen.yaml
scripts/fes-run.sh mediapipe_gen_onnx.yaml       # first start builds the TensorRT engines (~1 min), then cached

# Live, GUI only: two cameras, ONNX/TensorRT engine. Watch the "infer N ms" lines in processor_batch3d.log:
scripts/fes-run.sh mediapipe_live_onnx.yaml
tail -f ~/predictions/$(date +%Y%m%d)/*/logs/processor_batch3d.log

# Live hand-control task with BRAND (joint angles on :11115, 21 keypoints to BRAND's hand3d node on :11118):
SENDER_UDP_IP=192.168.137.201 scripts/fes-run.sh mediapipe_hand3d_brand_onnx.yaml   # fast engine
SENDER_UDP_IP=192.168.137.201 scripts/fes-run.sh mediapipe_hand3d_brand.yaml        # stock MediaPipe

# Replay with the DLC model and the held-hand association (stimulated hand), stock vs ONNX:
scripts/fes-run.sh dlc_hand_gen.yaml
scripts/fes-run.sh dlc_hand_gen_onnx.yaml        # needs models/dlc_trt/dlc_detector_960x540.onnx

# Closed-loop stimulation (Bayesian optimisation over electrodes x pulse width x frequency x current):
SENDER_UDP_IP=<brand ip> scripts/fes-run.sh bo_stim_live.yaml

# Record cameras only (no processing):
scripts/fes-run.sh 4cam_noproc.yaml

# 2D per-camera DLC -> xPC over UART (monkey sessions):
scripts/fes-run.sh 2dof_test.yaml
```

### Graphs

| Graph | Input | Pose | Output |
|---|---|---|---|
| `mediapipe_hand3d_brand[_onnx].yaml` | 2 live cameras | MediaPipe (stock / ONNX+TensorRT) | GUI, UDP to BRAND |
| `mediapipe_live_onnx.yaml` | 2 live cameras | MediaPipe ONNX+TensorRT | GUI (test) |
| `mediapipe_live.yaml`, `mediapipe_nosave.yaml` | 5 / 4 live cameras | MediaPipe | GUI (+ raw video) |
| `mediapipe_gen[_onnx].yaml` | 4 recorded videos | MediaPipe (stock / ONNX+TensorRT) | GUI |
| `dlc_hand_gen[_onnx].yaml` | 2 recorded videos | DLC top-down, held hand (stock / ONNX) | GUI |
| `3cam_3d` ... `7cam_3d.yaml` | 3-7 live cameras | per `config.yaml` | GUI, UDP, raw video |
| `bo_stim_live.yaml`, `bo_stim_replay.yaml` | live / recorded | DLC, held hand | stim requests to BRAND |
| `4cam_noproc.yaml`, `5cam_test.yaml`, `7cam_test.yaml` | live cameras | none | raw video |
| `1dof_dlc_udp.yaml`, `2dof_*.yaml`, `latency_benchmarking.yaml` | live cameras | 2D DLC per camera | UART / UDP to the xPC |

A graph can change any `config.yaml` key for ProcessorBatch3D with `config_overrides:` (that is how the `_onnx`
graphs switch engine). `camera_nums` maps each `frames{slot}_in` slot to a physical camera (index into
`camera_config.yaml` `active_cameras`, and the camera's name in the calibration).

## What happens to the data, live

Every stage, in order, for the main 3D path (MediaPipe, `hand_association: geometric`). Numbers are medians from the
2026-09-30 human 1-DOF run (2 cameras) unless noted.

| # | Stage | Where | What it does to the signal | Config | Cost |
|---|---|---|---|---|---|
| 1 | Capture | camera, `TIS.py` | 960x720 RGB at 30 fps, fixed exposure 2 ms / gain 0 / white balance. The two cameras free-run ~17 ms out of phase (no hardware sync). | `camera_config.yaml` | the driver timestamp is a constant 53 ms before the Python callback (unverified: may be a clock-offset artifact) |
| 2 | Hand-off | `TIS.py` | numpy copy -> redis store -> queue. `appsink_max_buffers: 1` keeps only the newest frame. | `appsink_max_buffers` | 1.5 ms |
| 3 | Gather | `ProcessorBatch3D._gather_newest` | newest frame per camera; waits up to `gather_wait_ms` (12) for the other cameras. Older frames are skipped, never queued. | graph `gather_wait_ms` | ~25 ms until both cameras' frames are in (mostly the phase offset) |
| 4 | Pose | `_infer_mediapipe` | per camera, MediaPipe's palm detector + 21-landmark net. **Stock MediaPipe in VIDEO mode also filters the landmarks internally**; the ONNX engine does not. | `mediapipe_*` | stock 34 ms, ONNX CPU 12.6 ms, TensorRT 2.4 ms (both cameras) |
| 5 | Score gate | `_geometric_3d` | hands whose handedness score < `threshold` are dropped. | `threshold` | - |
| 6 | Association + triangulation | `hand_assoc3d.py` | detections from different cameras form a 3D hand only if they reproject within `tol_px`; tracks are kept frame to frame; right/left from accumulated votes. | `geometric_association.*` | 2.7 ms |
| 7 | **One-Euro 3D smoothing** | `hand_assoc3d.OneEuroHand` | per keypoint, adaptive low-pass (1 Hz at rest, opening up with speed). When the hand is lost the last hand is repeated for `hold_frames` (5), then dropped. | `geometric_association.smooth` | **~50 ms lag** (measured) |
| 8 | **Hold and glide** | `continuity.py` | a missing keypoint holds its last value *indefinitely* (until it is found again); after a gap or a jump > 40 mm it glides back at 15% of the distance per frame (63% in ~200 ms). | `continuity` | lag only after gaps / jumps |
| 9 | Joint angles | `_joint_angles` | bone-to-bone angle at each joint, degrees from straight (dlc2kinematics' definition). | - | 0.05 ms |
| 10 | Send | `sender_udp.py` | latest angle per joint as JSON (NaN -> null); 21 keypoints of one hand to BRAND's hand3d node. No smoothing. | env `SENDER_UDP_IP`, graph `keypoint_port` | 0.5 ms |
| 11 | BRAND `hand3d` node | brand-monkeyrig | maps keypoints to DOFs, normalises with its calibration ranges, resamples 30 -> 100 Hz through **two cascaded low-pass stages (`output_tau_ms: 40`, ~40 ms lag)**. | BRAND graph | ~40 ms |

Reported end-to-end (`sender_joint_e2e.npy`: Python callback of the older camera -> UDP send) was ~62 ms with stock
MediaPipe: 25.6 gather + 33.9 inference + 2.7 triangulation + < 1 send. With TensorRT the inference term drops to
~2.4 ms. Stages 1, 7, 8 and 11 add lag the latency logs do not show.

Things that feel like lag but are not latency:
- **Hand lost.** Stage 8 holds the last value until the hand is found again, so a lost hand freezes the cursor
  (153 s once on 2026-09-30). `batch3d_points_3d_raw.npy` shows the raw gaps.
- **Skipped frames.** With stock MediaPipe a step (39 ms) was longer than a frame (33 ms), so ~8% of frames were
  skipped and the output came every ~39 ms instead of 33. The fast engines remove this.

Other paths:
- `hand_association: label`: per-camera 2D Kalman filter (`kalman_*`) instead of stages 6-7: low-confidence keypoints are
  estimated for up to `kalman_max_coast_frames`, then triangulated.
- `hand_association: region` (DLC): `held_hand3d.py` learns where the held hand is during `warmup_s`, then triangulates
  that hand only and smooths it with a One-Euro filter (`held_hand.smooth`).
- 2D xPC pipeline (`processor.py`): DLC -> 2D Kalman filter (2 ms forward prediction) -> the PIP keypoint's y pixel as
  the "angle" -> `sender.py` scales it to 0-1023 and sends it over UART. No other smoothing.

## The config, block by block

`config/config.yaml` documents every key inline; this is the map of which keys matter for what.

- **General**: `fps`, `output_path`; `resize` / `camera_prescaled` describe how frames were scaled (leave as is).
- **Replay** (`video_paths`, `video_path`): which recording each `Generator` plays.
- **2D DLC models** (`model_path_N`, `model_snapshot_N`): the per-camera xPC pipeline.
- **`threshold`**: the one confidence threshold (hand score for MediaPipe, keypoint likelihood for DLC).
- **Pose estimator**: `batch3d_backend` (mediapipe / dlc), then how it runs: `mediapipe_engine` / `onnx_providers`
  or `dlc_engine` / `dlc_onnx_*`. `mediapipe_num_hands` trades speed (1) for not latching onto the wrong hand (2).
- **Smoothing**, in pipeline order: `kalman_*` (label association only), `geometric_association.smooth`
  (One-Euro: lower `min_cutoff` = smoother and laggier, higher `beta` = less lag when moving; `false` turns it off),
  `continuity` (hold and glide; `null` turns it off; then a lost hand outputs NaN -> null instead of freezing).
- **3D**: `calibration_toml` + `calibration_camera_names`, `triangulation_min_cameras`, `hand_association` and its block.
- **`held_hand`**: the stimulated-hand (region) mode.
- **`cpu_affinity`**: which actor gets which cores (this CPU's core 0 is faulty and excluded).
- **`brand_link`, `bayes_opt`**: closed-loop stimulation ports, search space, trial timing, acquisition.

`camera_config.yaml`: `resolution` / `stream_resolution`, `fps`, `appsink_max_buffers` (1 for live tracking, 5 for
recording), `camera_settings` (exposure / gain / white balance, fixed on purpose), `active_cameras` (serials).

## Fast engines: ONNX Runtime / TensorRT

The networks are small: MediaPipe's palm detector takes 2.3 ms and its landmark net 0.9 ms on CPU in ONNX Runtime
(0.4 / 0.25 ms with TensorRT fp16). DLC's ResNet-50 pose net takes ~2 ms in PyTorch on the 4090. What made the steps
slow was the code around the networks:

- **MediaPipe's Python wrapper** wraps every frame in an `mp.Image`, runs a C++ graph (image conversion, letterbox,
  rotated crop, TFLite on CPU through XNNPACK, landmark projection, its own smoothing) and returns Python objects per
  landmark. With `num_hands: 2` and one hand in view it also reruns the palm detector every frame, looking for the
  second hand. All of that costs 13 ms with no hand and 25-35 ms with one, for two cameras on the CPU.
  `hand_onnx.py` keeps the same networks and the same tracking rules (palm detector only while fewer than
  `num_hands` hands are followed) but does the crop with one `cv2.warpAffine`, decodes with numpy and runs the nets
  on the GPU: 2.4 ms per 2-camera step. Its landmarks match MediaPipe's to ~6-8 px, about as much as MediaPipe's own
  VIDEO and IMAGE modes differ from each other (9.7 px); it has no landmark smoothing of its own.
- **DLC's runners** run the SSDLite detector and the pose net through generic batching / preprocessing / postprocessing
  code (thread queues, torchvision transforms, per-image dicts) and normalise full frames on the CPU. The detector
  step alone took 14.7 ms. `dlc_onnx.py` exports both networks with the normalisation inside the graph (uint8 in, no
  CPU float conversion), keeps DLC's crop and heatmap-decode maths in numpy, and runs on CUDA / TensorRT:
  5-7 ms per 2-camera step, keypoints within 0.03 px of DLC's.

Both are opt-in (`mediapipe_engine: onnx`, `dlc_engine: onnx`; default off) and the `_onnx` graphs turn them on.
Export / benchmark / validation scripts are in `scripts/trt/`.

## What a run saves

`<output_path>/<YYYYMMDD>/<run id>/`, where the run id is the start time as `YYYYMMDD-HHMM` (e.g. `20260930-1834`):

| Files | From | Contents |
|---|---|---|
| `logs/*.log` | every actor | setup, per-200-frame timing lines, warnings |
| `TIS*_<camera>.npy` | TIS | per frame: callback time (`TISstarts`), driver capture time (`TIScapture`), reader stages |
| `batch3d_points_2d.npy`, `batch3d_points_3d.npy`, `batch3d_joint_angles.npy` | processor | per step: 2D per camera, 3D (after smoothing), angles |
| `batch3d_points_3d_raw.npy` | processor | 3D before hold-and-glide (shows real gaps) |
| `batch3d_lat_*.npy`, `batch3d_frame_nums.npy`, `batch3d_time_skew_ms.npy` | processor | per-step timing, frames used, camera skew |
| `batch3d_detections.npz` | processor | every 2D detection (replay the association offline) |
| `sender_*.npy` | sender | what was sent and when; `sender_joint_e2e.npy` = callback -> send latency |
| `bo_trials.jsonl`, `brand_feedback.jsonl` | BO | one line per stimulation trial / BRAND message |

Raw video goes to `~/camera_video/<date>/<HHMMSS>/camera_N/buffer_*.bin`; convert it with the GUI button or the
`vision-camera-conversion` tools. A killed run leaves its logs in `.parts/`: `python -m actors.chunk_log <run folder>`
assembles them.

## Troubleshooting

- **Cameras missing after a reboot** (the preflight says "wants N, USB shows M"): the USB root hub needs
  re-enumerating; replug, or reset the controller.
- **A camera opens but gives no frames**: `camera_reader.log` says "pipeline could not be started" / "no frames for
  2 s". Check `tcam-ctrl -l` and the serial in `camera_config.yaml`.
- **First `_onnx` run is slow to start**: TensorRT is building engines (~1 min); they are cached in `models/*/trt_cache`.
- **`TensorrtExecutionProvider` not available**: `pip install --no-deps "tensorrt-cu12-libs>=10.13,<10.14"`; without
  it the providers list falls back to CUDA (3.4 ms instead of 2.4 ms).
- **xPC receiver records nothing**: the spoofed xPC link is not up (see `scripts/setup-xpc-link.sh`).
- **Random import errors / freezes**: something ran on CPU 0/1 (faulty). `fes-run.sh` and `cpu_affinity` avoid them;
  run things through `fes-run.sh`.
