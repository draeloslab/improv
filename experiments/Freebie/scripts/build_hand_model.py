"""Build the fixed hand model the kinematic fit uses (config.yaml `kinematic_fit.model`) from runs you trust.

    python scripts/build_hand_model.py ~/predictions/20261010/20261010-1412 [more run folders] \
        --out models/hand_models/<subject>.json [--min-frames 200]

Reads each run's batch3d_points_3d_raw.npy (or batch3d_points_3d.npy): the 3D hands BEFORE any smoothing. For each
hand block (right_*, left_*) it keeps frames where all 21 keypoints were triangulated, drops frames whose bone lengths
disagree with the consensus (> 20 % on any bone, twice), and stores the median palm shape and bone lengths. A block
with too few frames is left out; the fit then mirrors the other side.

Build it once the pose model and calibration are good, and keep it: the lengths are in mm, so any later session with a
metric (board-scaled) calibration uses the same file. Rebuild only if the subject's hand changes (growth) or the
keypoint definitions change (a retrained model that places joints differently).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from actors.kinematic_hand import FINGERS, build_model     # noqa: E402

BONES = [(idx[m], idx[m + 1]) for idx in FINGERS.values() for m in range(3)] + [(0, k) for k in (1, 5, 9, 13, 17)]


def consistent(H, tol=0.2, rounds=2):
    """Frames whose every bone is within tol of the median length (median recomputed after each round)."""
    keep = np.ones(len(H), bool)
    for _ in range(rounds):
        L = np.stack([np.linalg.norm(H[:, a] - H[:, b], axis=1) for a, b in BONES], 1)
        med = np.median(L[keep], 0)
        keep &= (np.abs(L / med - 1) < tol).all(1)
    return keep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('runs', nargs='+')
    ap.add_argument('--out', required=True)
    ap.add_argument('--min-frames', type=int, default=200)
    a = ap.parse_args()

    blocks = {'right': [], 'left': []}
    for r in a.runs:
        r = Path(r).expanduser()
        f = r / 'batch3d_points_3d_raw.npy'
        P = np.load(f if f.exists() else r / 'batch3d_points_3d.npy')
        if P.shape[1] == 42:
            blocks['right'].append(P[:, :21]); blocks['left'].append(P[:, 21:])
        else:
            blocks['right'].append(P[:, :21])

    out = dict(built=time.strftime('%Y-%m-%d %H:%M'), runs=[str(r) for r in a.runs])
    for side, parts in blocks.items():
        if not parts:
            continue
        H = np.concatenate(parts)
        H = H[np.isfinite(H).all(axis=(1, 2))]
        keep = consistent(H) if len(H) else np.zeros(0, bool)
        print(f"{side}: {len(H)} complete hands, {keep.sum()} with consistent bones")
        if keep.sum() < a.min_frames:
            print(f"  {side}: fewer than {a.min_frames} -- left out")
            continue
        model, rep = build_model(H[keep], min_frames=a.min_frames)
        model['report'] = rep
        out[side] = model
        print(f"  bones (mm): " + ", ".join(f"{f} {model['bones'][f]}" for f in FINGERS))
        print(f"  bone length CV in the kept frames: {rep['bone_cv']}")
    if 'right' not in out and 'left' not in out:
        sys.exit("no hand had enough frames")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(f"wrote {a.out}")


if __name__ == '__main__':
    main()
