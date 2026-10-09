#!/usr/bin/env python
"""Expand a master graph's `fes:` block into a full improv graph.

A master graph (graphs/mediapipe_live.yaml, mediapipe_gen.yaml, dlc_live.yaml, dlc_gen.yaml) lists only the
shared actors (GUI, VideoScreen3D, ProcessorBatch3D, SenderUDP) and a `fes:` block:

    fes:
      source: live            # live: CameraReader per camera | replay: Generator per camera (recorded video, never saved)
      cameras: [0, 1, 3, 6]   # physical camera numbers in slot order (slot i = frames{i}_in / images{i}_in)
      save_video: false       # live only: one VideoSaver per camera
      generator: {session: '2026-10-07/111100'}    # replay only: extra kwargs for every Generator (subfolder: raw ...)

This fills in what depends on the camera count -- one reader (+ saver) per camera, the connections, VideoScreen3D
num_active_cameras, ProcessorBatch3D num_cameras / camera_nums -- and writes a plain improv yaml. A yaml without a
`fes:` block is not touched. fes-run.sh calls this (--cameras 0,3,5 / --save-video / --no-save-video override the
yaml); the resolved graph is kept in the run folder as graph.yaml.
"""
import argparse
import sys

import yaml

READER = {"live": ("actors.camera_reader", "CameraReader"), "replay": ("actors.generator", "Generator")}


def expand(graph, cameras=None, save_video=None):
    fes = graph.pop("fes", None)
    if fes is None:
        return graph
    source = fes.get("source", "live")
    if source not in READER:
        raise ValueError(f"fes.source must be live or replay, not {source!r}")
    cams = list(cameras if cameras is not None else fes.get("cameras") or [])
    save = bool(fes.get("save_video", False) if save_video is None else save_video)
    if not cams:
        raise ValueError("fes.cameras is empty")
    if len(set(cams)) != len(cams):
        raise ValueError(f"duplicate camera in {cams}: each slot needs its own physical camera")
    if source == "replay" and save:
        raise ValueError("replay graphs never save video (the Generator plays the recording)")

    actors = graph.setdefault("actors", {})
    conns = graph.setdefault("connections", {})
    package, cls = READER[source]
    for slot, cam in enumerate(cams):
        extra = dict(fes.get("generator") or {}) if source == "replay" else {}
        actors[f"Generator{slot}"] = {"package": package, "class": cls, **extra, "camera_num": cam}
        sinks = [f"VideoScreen3D.images{slot}_in", f"ProcessorBatch3D.frames{slot}_in"]
        if save:
            actors[f"VideoSaver{slot}"] = {"package": "actors.video_saver", "class": "VideoSaver",
                                           "camera_num": cam, "method": "spawn"}
            sinks.append(f"VideoSaver{slot}.q_in")
        conns[f"Generator{slot}.q_out"] = sinks

    actors["VideoScreen3D"]["num_active_cameras"] = len(cams)
    actors["ProcessorBatch3D"]["num_cameras"] = len(cams)
    actors["ProcessorBatch3D"]["camera_nums"] = cams
    if len(cams) < 2 and actors["ProcessorBatch3D"].get("triangulate", True):
        print(f"[fes_graph] warning: {len(cams)} camera: nothing can be triangulated", file=sys.stderr)
    return graph


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("graph")
    ap.add_argument("out")
    ap.add_argument("--cameras", help="comma-separated physical camera numbers, e.g. 0,3,5,6")
    ap.add_argument("--save-video", dest="save_video", action="store_true", default=None)
    ap.add_argument("--no-save-video", dest="save_video", action="store_false")
    a = ap.parse_args()
    with open(a.graph) as f:
        graph = yaml.safe_load(f)
    cams = [int(c) for c in a.cameras.split(",")] if a.cameras else None
    try:
        graph = expand(graph, cams, a.save_video)
    except ValueError as e:
        sys.exit(f"[fes_graph] {a.graph}: {e}")
    with open(a.out, "w") as f:
        f.write(f"# resolved from {a.graph} by scripts/fes_graph.py -- do not edit, edit the master\n")
        yaml.safe_dump(graph, f, sort_keys=False, default_flow_style=None, width=140)


if __name__ == "__main__":
    main()
