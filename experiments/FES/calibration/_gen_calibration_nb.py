"""Generates calibration.ipynb (run: python _gen_calibration_nb.py).

The notebook is built as JSON by hand because nbformat is broken in the
anipose-cal env (see memory: anipose-cal-env). Edit this file, not the .ipynb.
"""
import json
from pathlib import Path

C = []
def md(s): C.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n").splitlines(keepends=True)})
def co(s): C.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.strip("\n").splitlines(keepends=True)})


md(r"""
# Multi-camera calibration (any number of cameras)

Pure **aniposelib** pipeline. Point it at a recording of a checkerboard and it
works out the rest: which cameras exist, which of them saw the board, how well
they are linked, and how many of them it can calibrate together.

**Only the next cell needs editing.** Kernel: `anipose-cal`.

1. **Detect** the board in every video (cached, so a re-run is instant)
2. **Graph** — board views per camera and simultaneous views per camera pair
3. **Repair** the detections (frame offsets between videos, 180-degree corner flips)
4. **Calibrate by growing** — solve the strongest 3-camera core first, then add
   one camera at a time, keeping each only if the error stays low. A weak camera
   is reported and left out; it never poisons the cameras that are good.
5. **Report** — writes `calibration.toml` and prints the exact lines to paste
   into `config/config.yaml`
6. **Checks** — 2-D corner overlays and a 3-D reconstruction
""")

co(r"""
# ============================ EDIT THIS CELL ============================
SESSION       = "/home/chesteklab/camera_video/2026-09-23/124500"  # folder holding camera_video_<N>_*.mp4
BOARD_TYPE    = "charuco"   # "charuco" (recommended) or "checkerboard"
# ChArUco: squares across/down (NOT inner corners), sizes in mm, and the ArUco dictionary the board was made with.
CHARUCO       = dict(squares_x=7, squares_y=5, square_mm=40.0, marker_mm=30.0, marker_bits=4, dict_size=50)
# Checkerboard: INNER corners across/down. Use one odd + one even count (e.g. 7x6): an even x even or odd x odd
# grid looks the same rotated 180 degrees and the corner order flips at random.
CHECKERBOARD  = dict(corners_x=8, corners_y=6, square_mm=25.0)

CAM_NUMS      = None    # None = every camera_video_<N> found in SESSION; or e.g. [0, 3, 4, 6]
EXCLUDE_CAMS  = []      # camera numbers to ignore, e.g. [1, 2]

OUT_ROOT      = "/home/chesteklab/improv/experiments/FES/calibration"   # results go in OUT_ROOT/<date>_<time>/
PRED_ROOT     = "/home/chesteklab/predictions"   # where the run logs/timestamps live (optional; used for the frame-rate check)

# --- tuning (the defaults suit a fixed rig of identical cameras) ---
MIN_BOARDS     = 10     # a camera needs this many board detections to be considered
MIN_CORNERS    = 8      # a detection with fewer corners than this is ignored (matters for ChArUco, which can see part of the board)
MIN_SHARED     = 5      # two cameras count as linked with this many simultaneous views
PER_PAIR       = 40     # frames kept per camera pair when thinning (bundle-adjustment cost scales with frames)
MAX_OWN        = 25     # extra frames kept per camera (helps its intrinsics)
MAX_ERR_PX     = 4.0    # a camera is only added if the mean reprojection error stays below this
MAX_FOCAL_RATIO = 1.3   # identical cameras should agree on focal length within this ratio (None = off)
FIX_SYNC       = True   # estimate and correct per-camera frame offsets (videos are rarely frame-synced)
MAX_LAG        = 60     # largest frame offset to search for, in frames (+-)
MIN_HALF       = 60     # shared frames needed in each half of the recording to test whether a camera pair's lag is constant
FIX_FLIPS      = True   # repair 180-degree corner-order flips (checkerboard only; ChArUco corners carry ids)
SOLVER         = dict(n_iters=2, max_nfev=30, ftol=1e-2)
SOLVE_TRIES    = 3      # aniposelib resamples points at random; a borderline camera gets this many seeded attempts
SOLVE_TIMEOUT  = 30     # seconds; a solve that hasn't converged by then is abandoned and that camera rejected
#   (aniposelib ignores max_nfev on its final pass and allows 200 evaluations, so a camera that
#    doesn't fit can otherwise take many minutes. Healthy solves take a few seconds.)
# ========================================================================
""")

co(r"""
import re, json, time, itertools
from pathlib import Path
from collections import defaultdict
import numpy as np, cv2
import matplotlib
matplotlib.use("Agg")          # headless-safe; figures are saved as PNG and also shown inline
import matplotlib.pyplot as plt
from aniposelib.boards import Checkerboard, CharucoBoard
from aniposelib.cameras import CameraGroup

SESSION = Path(SESSION)
found = defaultdict(list)
for p in sorted(SESSION.glob("camera_video_*_*.mp4")):
    m = re.match(r"camera_video_(\d+)_", p.name)
    if m: found[int(m.group(1))].append(str(p))
if not found:
    raise SystemExit(f"no camera_video_<N>_*.mp4 files in {SESSION}")

NUMS   = [n for n in (CAM_NUMS if CAM_NUMS is not None else sorted(found))
          if n in found and n not in EXCLUDE_CAMS]
missing = [n for n in (CAM_NUMS or []) if n not in found]
if missing: print("requested but no video found for cameras:", missing)
CAMS   = [f"cam{n}" for n in NUMS]            # names in calibration.toml == PHYSICAL camera numbers
VIDEOS = [found[n] for n in NUMS]
NC     = len(NUMS)

HERE = Path(OUT_ROOT) / f"{SESSION.parent.name}_{SESSION.name}"
HERE.mkdir(parents=True, exist_ok=True)

if BOARD_TYPE == "charuco":
    c = CHARUCO
    board = CharucoBoard(c["squares_x"], c["squares_y"], c["square_mm"], c["marker_mm"],
                         marker_bits=c["marker_bits"], dict_size=c["dict_size"])
    GRID_W, GRID_H, SQUARE_MM = c["squares_x"] - 1, c["squares_y"] - 1, c["square_mm"]
    FIX_FLIPS = False                      # ids make the corner order unambiguous
elif BOARD_TYPE == "checkerboard":
    c = CHECKERBOARD
    board = Checkerboard(c["corners_x"], c["corners_y"], square_length=c["square_mm"])
    GRID_W, GRID_H, SQUARE_MM = c["corners_x"], c["corners_y"], c["square_mm"]
    if (GRID_W + GRID_H) % 2 == 0:
        print(f"WARNING: a {GRID_W}x{GRID_H} inner-corner checkerboard is symmetric under 180 degree rotation, so corner order "
              "will flip at random (repaired below where possible). Use one odd + one even count, or a ChArUco board.")
else:
    raise SystemExit(f"BOARD_TYPE must be 'charuco' or 'checkerboard', not {BOARD_TYPE!r}")
N_CORNERS = GRID_W * GRID_H            # corners per board (dense, id-indexed)

cap = cv2.VideoCapture(VIDEOS[0][0]); FRAME_W = int(cap.get(3)); FRAME_H = int(cap.get(4)); cap.release()
print(f"session {SESSION}\n{NC} cameras: {CAMS}   frame {FRAME_W}x{FRAME_H}\noutput  {HERE}")

# Recording health. Two independent checks:
#  1. every video must be readable (a truncated file reports 0 frames) -- unreadable cameras are dropped;
#  2. every camera should have run at the same frame rate. The saved videos are all nominally 30 fps and often
#     come out with identical frame counts, which HIDES a camera that really ran slow (auto-exposure lengthening
#     past the frame period does this), so the true rate is read from the run's saved per-frame timestamps
#     (predictions/<date>/<date>-<HHMM>/saverstarts_cam_<N>.npy) when that folder exists.
counts = []
for vids in VIDEOS:
    n = 0
    for v in vids:
        cap = cv2.VideoCapture(v); n += max(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 0); cap.release()
    counts.append(n)
bad = [i for i, n in enumerate(counts) if n == 0]
for i in bad: print(f"UNREADABLE: {CAMS[i]} video has no frames (truncated or corrupt file) -- ignoring this camera")
keep = [i for i in range(NC) if i not in bad]
NUMS, CAMS, VIDEOS, counts = ([x[i] for i in keep] for x in (NUMS, CAMS, VIDEOS, counts)); NC = len(NUMS)
if not NC: raise SystemExit('no readable videos')
cap = cv2.VideoCapture(VIDEOS[0][0]); FRAME_W = int(cap.get(3)); FRAME_H = int(cap.get(4)); cap.release()

def _rate(n):
    d = Path(PRED_ROOT) / SESSION.parent.name.replace("-", "") / (SESSION.parent.name.replace("-", "") + "-" + SESSION.name[:4]) / f"saverstarts_cam_{n}.npy"
    try:
        t = np.load(d); return len(t) / (t[-1] - t[0]) if len(t) > 10 else None
    except Exception:
        return None
rates = [_rate(n) for n in NUMS]
have = [r for r in rates if r]
med_rate = float(np.median(have)) if have else None
med_cnt = float(np.median(counts))
print("\nrecording health:")
for c_, n_, r_ in zip(CAMS, counts, rates):
    msg = f"  {c_:5s} {n_:6d} frames"
    if r_: msg += f"   captured at {r_:6.2f} fps"
    slow = (r_ is not None and med_rate and r_ < med_rate * 0.995) or (r_ is None and n_ < med_cnt * 0.995)
    print(msg + ("   <-- SLOWER than the other cameras: it will drift against them; fix its exposure/frame rate" if slow else ""))
if not have: print("  (no run timestamps found under PRED_ROOT, so only frame counts were checked -- they cannot reveal a slow camera)")
""")

md(r"""
## 1 · Detect the checkerboard in every video

`cv2.findChessboardCorners` over every frame of every video (a few minutes for
~7 videos). Cached to `all_rows_raw.npy` in the output folder -- delete it to
re-detect. The cache is keyed by the camera list and the board, so changing `CAM_NUMS`,
`EXCLUDE_CAMS` or the board invalidates it.
""")

co(r"""
raw = HERE / "all_rows_raw.npy"
meta = HERE / "all_rows_raw.cams.json"
cache_key = dict(cams=CAMS, board=BOARD_TYPE, params=CHARUCO if BOARD_TYPE == "charuco" else CHECKERBOARD)
cache_ok = raw.exists() and meta.exists() and json.loads(meta.read_text()) == cache_key
cg0 = CameraGroup.from_names(CAMS, fisheye=False)
if cache_ok:
    all_rows = list(np.load(raw, allow_pickle=True))
    print("loaded cached detections")
else:
    t0 = time.time()
    all_rows = cg0.get_rows_videos(VIDEOS, board, verbose=True)
    np.save(raw, np.array(all_rows, dtype=object), allow_pickle=True)
    meta.write_text(json.dumps(cache_key))
    print(f"detected in {time.time()-t0:.0f} s")

fkey = lambda r: r["framenum"]
all_rows = [[r for r in rows if r["ids"].size >= MIN_CORNERS] for rows in all_rows]
fsets = [set(fkey(r) for r in rows) for rows in all_rows]
by_frame = defaultdict(dict)
for ci, rows in enumerate(all_rows):
    for row in rows:
        p = row.get("filled", row["corners"])
        by_frame[fkey(row)][ci] = np.asarray(p).reshape(-1, 2)
""")

md(r"""
## 2 · The calibration graph

Bundle adjustment needs the cameras to be *connected*: every camera has to share
board views (simultaneous detections) with the rest, directly or through
others. More shared views on a link means a better-constrained link.
""")

co(r"""
shared = np.array([[len(fsets[i] & fsets[j]) if i != j else len(fsets[i]) for j in range(NC)]
                   for i in range(NC)])
print("board detections per camera (diagonal) / simultaneous views per pair:\n")
print("       " + " ".join(f"{c:>6s}" for c in CAMS))
for i in range(NC):
    print(f"{CAMS[i]:>6s} " + " ".join(f"{shared[i, j]:6d}" for j in range(NC)))
mv = defaultdict(int)
for v in by_frame.values(): mv[len(v)] += 1
print("\nframes by number of cameras seeing the board:", dict(sorted(mv.items())))

fig, ax = plt.subplots(figsize=(1 + .8 * NC, .8 + .8 * NC))
show = shared.astype(float); np.fill_diagonal(show, np.nan)
im = ax.imshow(np.log10(show + 1), cmap="viridis")
ax.set_xticks(range(NC)); ax.set_xticklabels(CAMS); ax.set_yticks(range(NC)); ax.set_yticklabels(CAMS)
for i in range(NC):
    for j in range(NC):
        ax.text(j, i, str(shared[i, j]), ha="center", va="center", color="w", fontsize=8)
ax.set_title("simultaneous board views (diagonal = own detections)")
plt.tight_layout(); plt.savefig(HERE / "graph.png", dpi=110); plt.show()
""")


md(r"""
## 2b · Repair the detections: frame sync and corner flips

Two things silently ruin a multi-camera calibration, and both are checked here
against **epipolar geometry** (two views of the same rigid board must agree on one
fundamental matrix):

- **Frame offset.** Free-running cameras rarely start on the same frame. Pairing
  frame *N* of one video with frame *N* of another then pairs a board that has moved.
  For each linked pair this scans lags of up to `MAX_LAG` frames for the one where the
  most corners agree, chains the offsets to a reference camera, and re-keys the
  detections. A pair only counts if one lag clearly wins.
- **Corner flips.** A board whose inner-corner grid is even x even (or odd x odd) looks
  identical rotated 180 degrees, so `findChessboardCorners` sometimes returns the corners
  in reverse order. Each detection is compared against the cameras already fixed and
  reversed when that fits the epipolar geometry better. (Prevent it at the source with a
  board whose inner-corner counts are one odd, one even, e.g. 7x6 or 8x5.)

Cameras that cannot be verified are left as detected and reported.
""")

co(r"""
def _samp(F, a, b):
    a1 = np.c_[a, np.ones(len(a))]; b1 = np.c_[b, np.ones(len(b))]
    Fa = a1 @ F.T; Ftb = b1 @ F
    return np.sqrt(np.sum(b1 * Fa, axis=1) ** 2 / (Fa[:, 0]**2 + Fa[:, 1]**2 + Ftb[:, 0]**2 + Ftb[:, 1]**2 + 1e-12))

def _fund(A, B, th=3.0):
    if len(A) < 16: return None
    F, _ = cv2.findFundamentalMat(np.float64(A), np.float64(B), cv2.FM_RANSAC, th, 0.99)
    return F if F is not None and F.shape == (3, 3) else None

def frame_no(r): return r["framenum"][1] if isinstance(r["framenum"], tuple) else r["framenum"]
def vid_no(r):   return r["framenum"][0] if isinstance(r["framenum"], tuple) else 0

def corners_of(r): return np.asarray(r.get("filled", r["corners"])).reshape(-1, 2)   # (N_CORNERS, 2), NaN where not seen

D = [{frame_no(r): corners_of(r) for r in rows} for rows in all_rows]      # cam -> frame -> (N_CORNERS, 2)

MIN_PTS = 6
def _matched(a, b):
    m = np.isfinite(a).all(1) & np.isfinite(b).all(1)      # corners seen by both cameras
    return a[m], b[m]

def _frame_err(F, a, b):
    a, b = _matched(a, b)
    return np.median(_samp(F, a, b)) if len(a) >= MIN_PTS else np.nan

def pair_fit(i, j, L, flips=True, cap=160, sel=None):
    # frames of i whose frame+L exists in j -> (n, fraction consistent with one epipolar geometry, F, {frame: flipped})
    do_flip = flips and FIX_FLIPS
    fr = sorted(f for f in D[i] if f + L in D[j] and (sel is None or sel(f)))
    if len(fr) > cap: fr = [fr[int(k)] for k in np.linspace(0, len(fr) - 1, cap)]
    A = {f: D[i][f] for f in fr}; B = {f: D[j][f + L] for f in fr}
    fr = [f for f in fr if len(_matched(A[f], B[f])[0]) >= MIN_PTS
          or (do_flip and len(_matched(A[f], B[f][::-1])[0]) >= MIN_PTS)]
    if len(fr) < 20: return len(fr), 0.0, None, {}
    def stack(orient):
        pcs = [_matched(A[f], B[f][::-1] if orient[f] else B[f]) for f in fr]
        return np.vstack([p[0] for p in pcs]), np.vstack([p[1] for p in pcs])
    flip = {f: False for f in fr}
    F = _fund(*stack(flip))
    if F is None: return len(fr), 0.0, None, {}
    for _ in range(4 if do_flip else 0):
        for f in fr:
            e_keep, e_flip = _frame_err(F, A[f], B[f]), _frame_err(F, A[f], B[f][::-1])
            flip[f] = bool(np.isfinite(e_flip) and (not np.isfinite(e_keep) or e_flip < e_keep))
        F2 = _fund(*stack(flip))
        if F2 is None: break
        F = F2
    errs = [_frame_err(F, A[f], B[f][::-1] if flip[f] else B[f]) for f in fr]
    good = sum(1 for e in errs if np.isfinite(e) and e < 3) / len(fr)
    return len(fr), good, F, flip

offset = {i: 0 for i in range(NC)}
verified = {i: False for i in range(NC)}
edges = []                                   # (i, j, lag, score) with a clear winner
drifting = []                                # pairs whose lag is not constant over the recording
if FIX_SYNC or FIX_FLIPS:
    t0 = time.time()
    for i, j in itertools.combinations(range(NC), 2):
        if shared[i, j] < MIN_SHARED and not FIX_SYNC: continue
        lags = range(-MAX_LAG, MAX_LAG + 1) if FIX_SYNC else [0]
        sc = {L: pair_fit(i, j, L, flips=False)[1] for L in lags}
        if max(sc.values()) < 0.3: continue
        top = sorted(sc, key=lambda L: -sc[L])[:4]
        fine = {L: pair_fit(i, j, L, flips=True)[1] for L in top}          # flip-aware on the finalists
        L = max(fine, key=fine.get)
        rival = max([v for k, v in fine.items() if abs(k - L) > 1] + [0.0])
        if fine[L] >= 0.75 and fine[L] - rival >= 0.15:
            # A constant lag only exists if both cameras ran at the same frame rate. Re-estimate it on the
            # first and second half of the shared frames; if it moves, one camera is drifting.
            shared_f = sorted(f for f in D[i] if f + L in D[j]); mid = shared_f[len(shared_f) // 2]
            halves = None
            if len(shared_f) >= 2 * MIN_HALF:                 # too few frames per half and the split is just noise
                halves = []
                for sel in (lambda f: f < mid, lambda f: f >= mid):
                    sc2 = {l2: pair_fit(i, j, l2, flips=False, sel=sel)[1] for l2 in range(L - 15, L + 16)}
                    halves.append(max(sc2, key=sc2.get) if max(sc2.values()) >= 0.5 else None)
            if halves is not None and (None in halves or abs(halves[0] - halves[1]) > 2):
                drifting.append((i, j, halves))
            else:
                edges.append((i, j, L, fine[L]))
    print(f"pair scan: {time.time()-t0:.0f} s")
    print("pairs with a clear, verified alignment (cam_a, cam_b, lag b-a in frames, consistent frames):")
    for i, j, L, g in sorted(edges, key=lambda e: -e[3]):
        print(f"  {CAMS[i]}-{CAMS[j]}: lag {L:+d}   {g:.0%}")
    for i, j, h in drifting:
        print(f"  {CAMS[i]}-{CAMS[j]}: lag is NOT constant (first half {h[0]}, second half {h[1]} frames) -> "
              "one of these cameras ran at a different frame rate; not used for alignment")
    if not edges: print("  none -- offsets left at 0; the calibration below may be unreliable")

    # chain to a reference camera (the one with the most verified links)
    deg = defaultdict(int)
    for i, j, _, _ in edges: deg[i] += 1; deg[j] += 1
    if deg:
        ref = max(range(NC), key=lambda c: (deg[c], shared[c, c]))
        verified[ref] = True; queue = [ref]
        while queue:
            u = queue.pop(0)
            for i, j, L, g in sorted(edges, key=lambda e: -e[3]):
                if i == u and not verified[j]: offset[j] = offset[i] - L; verified[j] = True; queue.append(j)
                elif j == u and not verified[i]: offset[i] = offset[j] + L; verified[i] = True; queue.append(i)
        print(f"reference camera {CAMS[ref]};  frame offsets to add:",
              {CAMS[c]: offset[c] for c in range(NC)}, " unverified:", [CAMS[c] for c in range(NC) if not verified[c]])

# apply offsets: re-key detections so the same physical instant has the same key in every camera
def rekey(r, off):
    r2 = dict(r)
    v, f = (r["framenum"] if isinstance(r["framenum"], tuple) else (0, r["framenum"]))
    r2["framenum"] = (v, f + off); return r2
rows_fixed = [[rekey(r, offset[i]) for r in rows] for i, rows in enumerate(all_rows)]
D = [{frame_no(r): corners_of(r) for r in rows} for rows in rows_fixed]

# flip repair, camera by camera against the ones already fixed
n_flipped = {}
if FIX_FLIPS:
    order = [c for c in np.argsort([-deg.get(c, 0) for c in range(NC)]) if verified[c]] if edges else []
    done = order[:1]
    for c in order[1:]:
        ref_c = max(done, key=lambda x: len(set(D[x]) & set(D[c])))
        n, good, F, flip = pair_fit(ref_c, c, 0, flips=True, cap=10**9)
        n_flipped[CAMS[c]] = sum(flip.values()) if flip else 0
        if flip:
            fl = {f for f, v in flip.items() if v}
            new = []
            for r in rows_fixed[c]:
                if frame_no(r) in fl:
                    r = dict(r); r["corners"] = r["corners"][::-1].copy()
                    if "filled" in r: r["filled"] = r["filled"][::-1].copy()
                new.append(r)
            rows_fixed[c] = new
            D[c] = {frame_no(r): corners_of(r) for r in rows_fixed[c]}
        done.append(c)
    print("corner-order flips repaired per camera (frames):", n_flipped)

all_rows_raw_backup, all_rows = all_rows, rows_fixed
fsets = [set(fkey(r) for r in rows) for rows in all_rows]
by_frame = defaultdict(dict)
for ci, rows in enumerate(all_rows):
    for row in rows: by_frame[fkey(row)][ci] = corners_of(row)
shared = np.array([[len(fsets[i] & fsets[j]) if i != j else len(fsets[i]) for j in range(NC)] for i in range(NC)])
print("\nsimultaneous views per pair after repair:\n")
print("       " + " ".join(f"{c:>6s}" for c in CAMS))
for i in range(NC): print(f"{CAMS[i]:>6s} " + " ".join(f"{shared[i, j]:6d}" for j in range(NC)))
""")

md(r"""
## 3 · Calibrate by growing

1. **Candidates**: cameras with at least `MIN_BOARDS` detections that are linked
   (>= `MIN_SHARED` simultaneous views) into one connected group.
2. **Seed**: the 3 cameras (2 if that is all there is) with the most shared
   views among themselves, solved first. If a seed fails, the next-best seed is tried.
3. **Grow**: add the remaining cameras one at a time, strongest link to the current
   group first, re-solving the whole group each time. A camera is **kept only if**
   the mean reprojection error stays under `MAX_ERR_PX` and the focal lengths still
   agree; otherwise it is reported as rejected and the group stays as it was.

Every solve is capped (`SOLVER`), so nothing here can run for hours, and the
time of each solve is printed.
""")

co(r"""
def spread(frames, n):
    frames = sorted(frames)
    if len(frames) <= n: return set(frames)
    return {frames[int(k)] for k in np.linspace(0, len(frames) - 1, n)}

def subsample_rows(idx):
    # thin to a well-spread set of frames; every linked pair keeps PER_PAIR shared
    # frames so a weak link isn't thinned away
    keep = set()
    for a, b in itertools.combinations(idx, 2):
        keep |= spread(fsets[a] & fsets[b], PER_PAIR)
    for a in idx:
        keep |= spread(fsets[a], MAX_OWN)
    return [[r for r in all_rows[i] if fkey(r) in keep] for i in idx]

def intrinsics_ok(cg):
    fs = [c.get_camera_matrix()[0, 0] for c in cg.cameras]
    if any(abs(c.get_distortions()[0]) > 1 for c in cg.cameras): return False
    return MAX_FOCAL_RATIO is None or max(fs) / min(fs) <= MAX_FOCAL_RATIO

def _solve_worker(q, idx, rows, seed):
    try:
        np.random.seed(seed)
        cg = CameraGroup.from_names([CAMS[i] for i in idx], fisheye=False)
        cg.set_camera_sizes_videos([VIDEOS[i] for i in idx])
        err = cg.calibrate_rows(rows, board, init_intrinsics=True, init_extrinsics=True,
                                verbose=False, **SOLVER)
        q.put(("ok", cg.get_dicts(), float(err)))
    except Exception as ex:
        q.put(("err", f"{type(ex).__name__}: {ex}", np.inf))

def _solve_once(idx, rows, seed):
    import multiprocessing as mp, queue as _queue
    ctx = mp.get_context("fork")
    q = ctx.Queue(); t0 = time.time()
    pr = ctx.Process(target=_solve_worker, args=(q, list(idx), rows, seed)); pr.start()
    try:
        status, payload, err = q.get(timeout=SOLVE_TIMEOUT)
    except _queue.Empty:
        pr.terminate(); pr.join()
        print(f"   attempt {seed + 1}: gave up after {SOLVE_TIMEOUT} s"); return None, np.inf
    pr.join()
    if status != "ok":
        print(f"   attempt {seed + 1}: solve failed: {payload}"); return None, np.inf
    print(f"   attempt {seed + 1}: solved in {time.time()-t0:.0f} s  ->  mean reprojection error {err:.2f} px")
    return CameraGroup.from_dicts(payload), err

def solve(idx):
    # seeded, time-limited solve; retried with a new seed until one passes the acceptance test
    rows = subsample_rows(idx)
    print(f"   frames used: { {CAMS[i]: len(r) for i, r in zip(idx, rows)} }")
    best = (None, np.inf)
    for seed in range(SOLVE_TRIES):
        cg, e = _solve_once(idx, rows, seed)
        if cg is not None and e < best[1]: best = (cg, e)
        if cg is not None and e < MAX_ERR_PX and intrinsics_ok(cg): return cg, e
    return best

def components(cands):
    seen, comps = set(), []
    for s in cands:
        if s in seen: continue
        comp, stack = [], [s]
        while stack:
            u = stack.pop()
            if u in seen: continue
            seen.add(u); comp.append(u)
            stack += [v for v in cands if v not in seen and shared[u, v] >= MIN_SHARED]
        comps.append(sorted(comp))
    return max(comps, key=len) if comps else []

cands = [i for i in range(NC) if shared[i, i] >= MIN_BOARDS]
group0 = components(cands)
unusable = [CAMS[i] for i in range(NC) if i not in cands]
unlinked = [CAMS[i] for i in cands if i not in group0]
print("too few detections:", unusable)
print("not linked to the main group:", unlinked)
if len(group0) < 2:
    raise SystemExit("fewer than 2 linked cameras with board detections -- record more board coverage")

# Cameras whose alignment was verified against another camera (section 2b) are trusted
# to seed the solve; unverified ones are only tried afterwards, one at a time.
ver = [i for i in group0 if verified[i]]
core = ver if len(ver) >= 2 else list(group0)
late = [i for i in group0 if i not in core]
print("seed pool:", [CAMS[i] for i in core], "  tried last:", [CAMS[i] for i in late])

# seeds: best-linked triples (or the pair if only 2 cameras), strongest first
k = min(3, len(core))
seeds = sorted(itertools.combinations(core, k),
               key=lambda s: -sum(shared[a, b] for a, b in itertools.combinations(s, 2)))
seeds = [s for s in seeds if all(any(shared[a, b] >= MIN_SHARED for b in s if b != a) for a in s)]

cgroup, err, accepted = None, np.inf, []
for seed in seeds[:4]:
    print(f"\n=== seed {[CAMS[i] for i in seed]} ===")
    cg, e = solve(list(seed))
    if cg is not None and e < MAX_ERR_PX and intrinsics_ok(cg):
        cgroup, err, accepted = cg, e, list(seed); break
    print(f"   seed rejected (error {e:.2f} px, intrinsics_ok={cg is not None and intrinsics_ok(cg)})")
if cgroup is None:
    raise SystemExit("no seed converged under MAX_ERR_PX. Look at the graph above: the cameras likely "
                     "don't share enough board views, or the board was held too far / too tilted.")

rejected = {}
remaining = [i for i in group0 if i not in accepted]
in_core = set(core)
while remaining:
    nxt = max(remaining, key=lambda c: (c in in_core, sum(shared[c, a] for a in accepted)))
    remaining.remove(nxt)
    link = sum(shared[nxt, a] for a in accepted)
    print(f"\n=== adding {CAMS[nxt]} (shared views with current group: {link}) ===")
    if link < MIN_SHARED:
        rejected[CAMS[nxt]] = f"only {link} shared views with the group"; print("   skipped:", rejected[CAMS[nxt]]); continue
    cg, e = solve(sorted(accepted + [nxt]))
    if cg is not None and e < MAX_ERR_PX and intrinsics_ok(cg):
        cgroup, err, accepted = cg, e, sorted(accepted + [nxt])
        print(f"   kept {CAMS[nxt]}")
    else:
        why = "solve failed" if cg is None else f"error {e:.2f} px" + ("" if intrinsics_ok(cg) else ", intrinsics implausible")
        rejected[CAMS[nxt]] = why; print(f"   REJECTED {CAMS[nxt]}: {why}")
for c in unusable: rejected[c] = f"fewer than {MIN_BOARDS} board detections"
for c in unlinked: rejected[c] = f"not linked to the main group (< {MIN_SHARED} shared views)"

PAIR = accepted
used = [CAMS[i] for i in PAIR]
cgroup.dump(HERE / "calibration.toml")
print(f"\nCALIBRATED: {used}   mean reprojection error {err:.2f} px")
print("LEFT OUT   :", rejected or "none")
""")

md(r"""
## 4 · Result and what to put in the config

Per-camera intrinsics and the lines to paste into `config/config.yaml`.
Camera names in the toml are the **physical** camera numbers (`cam3` = the
camera wired as `camera_num: 3`), so `calibration_camera_names` can stay empty.
""")

co(r"""
print("intrinsics:")
for cam in cgroup.cameras:
    K = cam.get_camera_matrix(); d = np.round(cam.get_distortions().ravel(), 3)
    print(f"  {cam.get_name():5s} fx={K[0,0]:7.1f} fy={K[1,1]:7.1f} cx={K[0,2]:6.1f} cy={K[1,2]:6.1f}  dist={d}")

toml_rel = (HERE / "calibration.toml").relative_to(Path(OUT_ROOT).parent)
print("\n--- paste into experiments/FES/config/config.yaml ---")
print(f"calibration_toml: '{toml_rel}'")
print("calibration_camera_names: {}")
print(f"calibration_frame_size: [{FRAME_W}, {FRAME_H}]")
print(f"# in the graph yaml: camera_nums must list the PHYSICAL numbers, calibrated ones are {NUMS and [NUMS[i] for i in PAIR]}")

report = dict(session=str(SESSION), calibrated=used, left_out=rejected, mean_error_px=float(err),
              frame_size=[FRAME_W, FRAME_H], board=dict(type=BOARD_TYPE, **(CHARUCO if BOARD_TYPE == 'charuco' else CHECKERBOARD)), solver=SOLVER,
              shared_views=shared.tolist(), cameras=CAMS,
              frame_offsets={CAMS[c]: int(offset[c]) for c in range(NC)},
              sync_verified={CAMS[c]: bool(verified[c]) for c in range(NC)})
(HERE / "report.json").write_text(json.dumps(report, indent=1))
""")

md(r"""
## 5 · Check: detected corners, per-camera error

The frame seen by the most calibrated cameras, with every camera's detected
corners overlaid, then the reprojection error per camera over up to 60 frames.
""")

co(r"""
pos = {ci: k for k, ci in enumerate(PAIR)}
def n_used(fn): return sum(1 for c in by_frame[fn] if c in pos)
best = max(by_frame, key=lambda fn: (n_used(fn), str(fn)))
frame_idx = best[1] if isinstance(best, tuple) else best
vid_idx = best[0] if isinstance(best, tuple) else 0
print("frame", frame_idx, "-> cameras", [CAMS[i] for i in sorted(by_frame[best]) if i in pos])

def grab(ci, frame, vi=0):
    cap = cv2.VideoCapture(VIDEOS[ci][min(vi, len(VIDEOS[ci]) - 1)]); cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame))
    ok, im = cap.read(); cap.release()
    return im if ok else None

colors = plt.cm.turbo(np.linspace(0, 1, N_CORNERS))
ncol = min(4, len(PAIR)); nrow = -(-len(PAIR) // ncol)
fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 2.9 * nrow), squeeze=False)
for ax in axes.ravel(): ax.axis("off")
for ax, ci in zip(axes.ravel(), PAIR):
    ax.axis("on"); im = grab(ci, frame_idx, vid_idx)
    if im is not None: ax.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
    ax.set_title(CAMS[ci]); ax.set_xticks([]); ax.set_yticks([])
    if ci in by_frame[best]:
        p = by_frame[best][ci]; g = p.reshape(GRID_H, GRID_W, 2)
        for r_ in range(GRID_H): ax.plot(g[r_, :, 0], g[r_, :, 1], "-", c="w", lw=.7, alpha=.6)
        for k in range(GRID_W):  ax.plot(g[:, k, 0], g[:, k, 1], "-", c="w", lw=.7, alpha=.6)
        ax.scatter(p[:, 0], p[:, 1], c=colors, s=25, edgecolors="k", linewidths=.4)
    else:
        ax.text(.5, .5, "board not detected", transform=ax.transAxes, ha="center", va="center", color="r")
fig.suptitle(f"Detected corners, frame {frame_idx}"); plt.tight_layout()
plt.savefig(HERE / "corners_2d.png", dpi=110); plt.show()

# per-camera reprojection error over many multi-camera frames
multi = [fn for fn in by_frame if n_used(fn) >= 2]
multi = sorted(multi, key=str)[:: max(1, len(multi) // 60)]
per_cam = defaultdict(list)
for fn in multi:
    pts = np.full((len(PAIR), N_CORNERS, 2), np.nan)
    for ci, k in pos.items():
        if ci in by_frame[fn]: pts[k] = by_frame[fn][ci]
    p3 = cgroup.triangulate(pts, progress=False)
    for ci, k in pos.items():
        if np.isfinite(pts[k]).any():
            proj = np.asarray(cgroup.cameras[k].project(p3)).reshape(-1, 2)
            per_cam[CAMS[ci]].append(np.nanmean(np.linalg.norm(proj - pts[k], axis=1)))
print(f"\nreprojection error per camera over {len(multi)} multi-camera frames (px):")
for c in used: print(f"  {c:5s} median {np.nanmedian(per_cam[c]):5.2f}   90th pct {np.nanpercentile(per_cam[c], 90):5.2f}   n={len(per_cam[c])}")
""")

md(r"""
## 6 · Check: 3-D reconstruction

Triangulate the board from the frame above and plot it with the recovered camera
poses (▲ position, line = viewing direction). The mean square edge should be
close to the real `SQUARE_MM`.
""")

co(r"""
pts = np.full((len(PAIR), N_CORNERS, 2), np.nan)
for ci, k in pos.items():
    if ci in by_frame[best]: pts[k] = by_frame[best][ci]
p3d = cgroup.triangulate(pts, progress=False)
rep = cgroup.reprojection_error(p3d, pts, mean=True)
g3 = p3d.reshape(GRID_H, GRID_W, 3)
edges = np.r_[np.linalg.norm(np.diff(g3, axis=0), axis=2).ravel(), np.linalg.norm(np.diff(g3, axis=1), axis=2).ravel()]
print(f"triangulated {int(np.sum(np.isfinite(p3d[:,0])))}/{N_CORNERS} corners, mean reprojection error {np.nanmean(rep):.2f} px")
print(f"mean square edge {np.nanmean(edges):.2f} mm  (nominal {SQUARE_MM})")

cams_C = []
for cam in cgroup.cameras:
    R, _ = cv2.Rodrigues(np.asarray(cam.get_rotation()))
    cams_C.append((cam.get_name(), -R.T @ np.asarray(cam.get_translation()), R.T @ np.array([0, 0, 200.0])))
P = np.vstack([p3d] + [c for _, c, _ in cams_C]); mid = np.nanmean(P, axis=0); rng = np.nanmax(np.abs(P - mid)) * 1.1

fig = plt.figure(figsize=(10, 8)); ax = fig.add_subplot(111, projection="3d")
ax.scatter(*p3d.T, c=colors, s=45, edgecolors="k", lw=.4, depthshade=False)
for r_ in range(GRID_H): ax.plot(*g3[r_].T, c="0.45", lw=.8)
for k in range(GRID_W):  ax.plot(*g3[:, k].T, c="0.45", lw=.8)
for name, Cc, look in cams_C:
    ax.scatter(*Cc, marker="^", s=160, c="k"); ax.plot(*np.c_[Cc, Cc + look], c="k", lw=1.5); ax.text(*Cc, "  " + name)
ax.set_xlim(mid[0]-rng, mid[0]+rng); ax.set_ylim(mid[1]-rng, mid[1]+rng); ax.set_zlim(mid[2]-rng, mid[2]+rng)
ax.set_xlabel("X (mm)"); ax.set_ylabel("Y (mm)"); ax.set_zlabel("Z (mm)"); ax.view_init(elev=18, azim=-70)
try: ax.set_box_aspect((1, 1, 1))
except Exception: pass
ax.set_title(f"Board + camera poses, frame {frame_idx}"); plt.tight_layout()
plt.savefig(HERE / "reconstruction_3d.png", dpi=110); plt.show()
""")

md(r"""
## Reading the result

| check | good sign |
|---|---|
| board detections per camera | hundreds, not tens |
| simultaneous views per pair | 50+ on the links that matter |
| mean reprojection error | under ~2 px (each added camera must stay under `MAX_ERR_PX`) |
| focal lengths | within ~10-20% across identical cameras |
| distortion `k1` | roughly within ±1 |
| triangulated square edge | close to `SQUARE_MM` |

**A camera was left out?** The printed reason says why. Usually it saw the board
too rarely, or never at the same time as the others. Record more board coverage
for that camera (near, far, tilted, at the edges of its view, while other cameras
also see it) and re-run; delete `all_rows_raw.npy` only if the *videos* changed.
""")

nb = {"cells": C,
      "metadata": {"kernelspec": {"display_name": "anipose-cal", "language": "python", "name": "anipose-cal"},
                   "language_info": {"name": "python", "version": "3.10"}},
      "nbformat": 4, "nbformat_minor": 5}
out = Path(__file__).resolve().parent / "calibration.ipynb"
out.write_text(json.dumps(nb, indent=1))
print("wrote", out, len(C), "cells")
