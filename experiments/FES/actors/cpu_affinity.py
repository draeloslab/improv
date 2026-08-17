"""Deterministic CPU core placement for the FES pipeline actors.

Why this exists
---------------
This machine is an i9-14900K: a *hybrid* CPU. cpu0-15 are the SMT thread pairs
of 8 performance cores (5.7 GHz, and 6.0 GHz on the two favoured cores), and
cpu16-31 are 16 efficiency cores (4.4 GHz, lower IPC, no SMT). Linux's
scheduler is free to put any process on any of them, and it does.

That matters here because `Processor.runStep`'s DLC timer wraps the *whole*
`pose_runner.inference()` call -- numpy->tensor, host-to-device copy, forward
pass, argmax predictor, post back to numpy -- and everything except the forward
pass is host-side work that scales with CPU clock. Measured with the real model
on a real 960x540 frame, one process at a time, pinned:

    P-core cpu0  (5.7 GHz)    6.61 ms       E-core cpu20 (4.4 GHz)   11.36 ms
    P-core cpu8  (6.0 GHz)    6.32 ms       E-core cpu24 (4.4 GHz)   12.02 ms

That is a 1.72x penalty purely for landing on an E-core. The live 4-camera run
20260804-1432 -- identical weights on all four cameras -- showed a 1.79x spread
(cam0 11.56 ms .. cam3 20.65 ms). Same number. Core placement, not the GPU, is
what makes the cameras unequal.

GPU contention is real but it is *symmetric*, so it cannot produce an ordering:
concurrent copies of the same inference measured 5.60 ms (1 process),
10.32 ms (2), 10.28 ms (4). Going 1->2 doubles it and 2->4 adds nothing; every
process pays the same. It raises the floor, it does not tilt it.

With ~15 busy processes (4 processors + 4 camera readers + 4 savers + GUI +
sender + redis) competing for 8 physical P-cores, some processors get an E-core.
Processor0 launches first and reliably wins a P-core, which is exactly the
ordering seen in the logs.

What this module does
---------------------
Hands each actor a *fixed, non-overlapping* set of CPUs, chosen as a pure
function of the actor's role and slot index so that separate processes agree
without having to coordinate at runtime:

    compute    (Processor)      one whole physical P-core each, fastest first
    capture    (CameraReader)   one whole physical P-core each, after compute
    background (savers, GUI,    all E-cores, shared
                sender)

Each compute/capture actor gets *both* SMT threads of its physical core, so the
CUDA driver's helper threads have somewhere to run without preempting the main
thread, while no other actor is ever placed on that core.

Affinity is inherited by threads created after the call, so `pin_actor()` must
run at the *top* of an actor's `setup()` -- before `get_inference_runners()`
spins up CUDA threads, and before TIS builds the GStreamer pipeline.
"""

import os
import logging

logger = logging.getLogger(__name__)

_CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "config.yaml")

# Roles, in the order they claim physical P-cores.
COMPUTE = "compute"
CAPTURE = "capture"
BACKGROUND = "background"

_SYSFS = "/sys/devices/system/cpu"


def load_config():
    """Read config.yaml, so actors that don't otherwise need it can still pin.

    Returns {} if it cannot be read -- pinning then falls back to its defaults
    rather than taking the pipeline down.
    """
    try:
        import yaml
        with open(_CONFIG_PATH) as f:
            return yaml.safe_load(f) or {}
    except Exception as e:  # noqa: BLE001 - never block startup on this
        logger.warning(f"cpu_affinity: could not read {_CONFIG_PATH} ({e}); "
                       f"using defaults")
        return {}


def _read(path):
    try:
        with open(path) as f:
            return f.read().strip()
    except OSError:
        return None


def _parse_cpu_list(spec):
    """Parse a sysfs cpu list ("0-1", "4", "0-3,8") into a sorted list of ints."""
    cpus = []
    for part in spec.split(","):
        if "-" in part:
            lo, hi = part.split("-")
            cpus.extend(range(int(lo), int(hi) + 1))
        else:
            cpus.append(int(part))
    return sorted(cpus)


def topology():
    """Discover physical cores, split into performance and efficiency tiers.

    Returns (p_cores, e_cores) where p_cores is a list of CPU-index lists, one
    entry per physical performance core, ordered fastest first and then by core
    id so the ordering is stable across processes and reboots; e_cores is a flat
    list of efficiency CPU indices.

    The performance/efficiency split is made on max frequency rather than on
    "has SMT siblings", because that stays correct if hyperthreading is disabled
    in the BIOS. On a non-hybrid CPU every core reports the same frequency, they
    all land in the performance tier, and e_cores comes back empty -- which the
    callers below handle by falling back to the performance pool.
    """
    online = _read(f"{_SYSFS}/online")
    if not online:
        return [], []

    cores = {}  # core_id -> {"cpus": set, "khz": int}
    for cpu in _parse_cpu_list(online):
        core_id = _read(f"{_SYSFS}/cpu{cpu}/topology/core_id")
        if core_id is None:
            continue
        khz = _read(f"{_SYSFS}/cpu{cpu}/cpufreq/cpuinfo_max_freq")
        entry = cores.setdefault(int(core_id), {"cpus": set(), "khz": 0})
        entry["cpus"].add(cpu)
        entry["khz"] = max(entry["khz"], int(khz) if khz else 0)

    if not cores:
        return [], []

    top_khz = max(c["khz"] for c in cores.values())
    # Anything within 15% of the fastest core is a performance core. On this
    # part that is 5.7-6.0 GHz vs 4.4 GHz for the E-cores, a 27% gap.
    cutoff = top_khz * 0.85

    p_cores, e_cores = [], []
    for core_id, c in sorted(cores.items()):
        cpus = sorted(c["cpus"])
        if c["khz"] >= cutoff:
            p_cores.append((-c["khz"], core_id, cpus))
        else:
            e_cores.extend(cpus)

    # Fastest first, then by core id. Deterministic across processes.
    p_cores.sort()
    return [cpus for _, _, cpus in p_cores], sorted(e_cores)


def _core_pool(avoid_cpus):
    """Physical P-cores in claim order, with any core touching `avoid_cpus` last.

    Cores are demoted rather than dropped so that a machine with fewer cores
    than actors still gets a usable (if shared) assignment instead of an empty
    pool.
    """
    p_cores, e_cores = topology()
    avoid = set(avoid_cpus or ())
    clean = [c for c in p_cores if not avoid.intersection(c)]
    dirty = [c for c in p_cores if avoid.intersection(c)]
    return clean + dirty, e_cores


def pin_actor(role, slot=0, config=None, label=None):
    """Pin the calling process to the CPUs assigned to (role, slot).

    role   COMPUTE / CAPTURE / BACKGROUND
    slot   which actor of that role this is (camera_num works well)
    config the parsed config.yaml dict, or None to use defaults
    label  name to use in the log line, e.g. "Processor cam2"

    Returns the list of CPUs pinned to, or None if pinning was disabled,
    unavailable, or failed. Never raises -- a pipeline that cannot set affinity
    should still run, just with the old unequal timings.
    """
    config = load_config() if config is None else config
    settings = config.get("cpu_affinity") or {}
    label = label or f"{role}{slot}"

    if not settings.get("enabled", True):
        return None
    if not hasattr(os, "sched_setaffinity"):
        logger.info(f"{label}: os.sched_setaffinity unavailable, not pinning")
        return None

    # Explicit override wins over everything: cpu_affinity.overrides.<label>
    override = (settings.get("overrides") or {}).get(label)
    if override:
        return _apply(sorted(int(c) for c in override), label, "override")

    n_compute = int(settings.get("compute_slots",
                                 config.get("max_camera_slots", 4)))
    pool, e_cores = _core_pool(settings.get("avoid_cpus"))

    if role == BACKGROUND:
        # Savers, GUI and sender are throughput work, not latency work: the
        # saver buffers to disk, the GUI redraws at its own rate, and the
        # sender's own step costs ~0.1 ms. Keeping them off the P-cores is the
        # whole point -- it is what leaves the P-cores free for the processors.
        cpus = e_cores or sorted({c for core in pool for c in core})
        return _apply(cpus, label, "background pool")

    if not pool:
        logger.warning(f"{label}: no performance cores discovered, not pinning")
        return None

    index = slot if role == COMPUTE else n_compute + slot
    if index >= len(pool):
        # More actors than physical cores. Wrap rather than fail; the placement
        # is still deterministic, just shared. This is the >=5 camera regime,
        # where the answer is crop+batch in one process, not more processes.
        logger.warning(
            f"{label}: wants physical core #{index} but only {len(pool)} exist; "
            f"sharing core #{index % len(pool)}. Beyond {len(pool) - n_compute} "
            f"cameras, batch the crops into one process instead of adding "
            f"processes -- see MULTICAM_3D_PLAN.md."
        )
    core = pool[index % len(pool)]
    return _apply(core, label, f"physical core #{index % len(pool)}")


def _apply(cpus, label, why):
    try:
        os.sched_setaffinity(0, set(cpus))
        actual = sorted(os.sched_getaffinity(0))
        logger.info(f"{label}: pinned to CPUs {actual} ({why})")
        return actual
    except OSError as e:
        logger.warning(f"{label}: could not set CPU affinity ({e}); "
                       f"continuing unpinned")
        return None
