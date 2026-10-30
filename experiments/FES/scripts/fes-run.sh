#!/usr/bin/env bash
#
# Start an FES experiment with one command.
#
# Picks a run id up front, exports it as $IMPROV_RUN_ID, and every actor writes its logs and its .npy files
# into <output_path>/<date>/<run id>/ (output_path in config/config.yaml).
#
# USAGE
#   fes-run                                   # mediapipe_live.yaml, with the TUI
#   fes-run dlc_live.yaml                     # a graph in graphs/, or any path to a yaml
#   fes-run old_rig/graphs/2dof_test.yaml     # 2D xPC graphs
#   fes-run --auto mediapipe_gen.yaml         # headless: prompts ENTER to start/stop
#   fes-run --no-check dlc_live.yaml          # skip the camera/redis preflight
#   fes-run --dry-run dlc_live.yaml           # preflight only, launch nothing
#
# MASTER GRAPHS (mediapipe_live / dlc_live / mediapipe_gen / dlc_hand_gen: any yaml with a `fes:` block) are expanded
# first by scripts/fes_graph.py; the resolved graph is kept as <run folder>/graph.yaml. Options for those:
#   fes-run mediapipe_live.yaml --cameras 0,3,5,6    # camera count/order for this run (physical camera numbers)
#   fes-run mediapipe_live.yaml --save false         # record raw video? true (default) or false; overrides save_video in the yaml
#
set -uo pipefail

# Camera capture allocates/frees large buffers at a high rate. Past glibc's
# mmap threshold, free() munmaps those instead of keeping them on the heap,
# and munmap of a large mapping has to IPI every CPU to flush its TLB --
# at high enough frequency across many cores this has frozen the machine
# solid (soft lockups on multiple CPUs, needing a hard power-cycle; see
# journalctl -b -1 around the freeze). Raising the mmap threshold and
# disabling trim keeps freed buffers on the heap for reuse instead.
export MALLOC_MMAP_THRESHOLD_="${MALLOC_MMAP_THRESHOLD_:-33554432}"
export MALLOC_TRIM_THRESHOLD_="${MALLOC_TRIM_THRESHOLD_:--1}"

# Frames expire from the Redis store after this many seconds (improv/store.py; improv's default is 60).
# Every consumer takes a frame within milliseconds -- the VideoSavers stream them to disk as they arrive --
# so 15 s is ample and keeps ~6.5 GB of frames in RAM at 7 cameras instead of ~26 GB.
export IMPROV_STORE_TTL_S="${IMPROV_STORE_TTL_S:-15}"

FES_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_ENV="${FES_CONDA_ENV:-improvDLC3}"
DEFAULT_YAML="graphs/mediapipe_live.yaml"

# Pin THIS script (and so everything it starts: the preflight checks, `improv cleanup`, improv) to FES_CPUS
# before anything runs. The default leaves out cpu0/1 (unstable on our rig); cpu_affinity.exclude_cpus in
# config.yaml is the same list. Set FES_CPUS=0-$(( $(nproc --all) - 1 )) to use every core.
FES_CPUS="${FES_CPUS:-2-$(( $(nproc --all) - 1 ))}"
taskset -pc "$FES_CPUS" $$ >/dev/null 2>&1 || echo "[fes] warning: could not pin to CPUs $FES_CPUS" >&2

AUTO=0
CHECKS=1
DRY_RUN=0
YAML=""
SAVE=true
GRAPH_ARGS=()

while [ $# -gt 0 ]; do
    case "$1" in
        --auto)     AUTO=1 ;;
        --no-check) CHECKS=0 ;;
        --dry-run)  DRY_RUN=1 ;;
        --cameras)  shift; GRAPH_ARGS+=(--cameras "${1:?--cameras needs a list, e.g. 0,3,5,6}") ;;
        --record-only) shift; GRAPH_ARGS+=(--record-only "${1?--record-only needs a list, e.g. 1}") ;;
        --save)     shift; SAVE="${1:?--save needs true or false}"
                    case "$SAVE" in true|false) ;; *) echo "error: --save must be true or false, got '$SAVE'" >&2; exit 1 ;; esac ;;
        -h|--help)  awk 'NR>1 && /^#/ {sub(/^# ?/,""); print; next} NR>1 {exit}' \
                        "${BASH_SOURCE[0]}"; exit 0 ;;
        -*)         echo "unknown option: $1" >&2; exit 1 ;;
        *)          YAML="$1" ;;
    esac
    shift
done

YAML="${YAML:-$DEFAULT_YAML}"

# ---------------------------------------------------------------- environment

# `conda activate` is a shell function that only exists after the hook is
# sourced -- a non-interactive script does not get it from .bashrc.
CONDA_BASE="${CONDA_BASE:-$HOME/miniforge3}"
if [ -r "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
    # shellcheck disable=SC1091
    # conda's own activation scripts (conda.sh, and this env's
    # etc/conda/activate.d/*.sh hooks, e.g. env_vars.sh) are not written to be
    # sourced under `set -u` -- they reference vars like $LD_LIBRARY_PATH on
    # the assumption they may be unset. Under -u that isn't a warning, it's
    # an immediate exit of this whole script, silently, before the env is
    # actually active. Disable -u for exactly this block.
    set +u
    . "$CONDA_BASE/etc/profile.d/conda.sh"
    conda activate "$CONDA_ENV"
    activate_status=$?
    set -u
    [ "$activate_status" -eq 0 ] || { echo "could not activate $CONDA_ENV" >&2; exit 1; }

    # `conda activate` sets CONDA_PREFIX but only prepends its own PATH entry.
    # If another env's bin was put on PATH outside conda's control -- which is
    # what happens when you launch from a shell that already had one active --
    # that stale entry still wins and you silently run the wrong interpreter.
    # Check rather than assume, since the symptom is a confusing ImportError
    # deep inside an actor rather than an obvious environment error.
    if [ -n "${CONDA_PREFIX:-}" ] && \
       [ "$(command -v python)" != "$CONDA_PREFIX/bin/python" ]; then
        echo "[fes] correcting PATH: python resolved to $(command -v python)"
        export PATH="$CONDA_PREFIX/bin:$PATH"
    fi
    echo "[fes] conda env: $CONDA_ENV ($(command -v python))"
else
    echo "[fes] warning: $CONDA_BASE/etc/profile.d/conda.sh not found;" \
         "continuing in the current environment" >&2
fi

cd "$FES_DIR" || exit 1

# Pipeline yamls live in graphs/; accept a bare name, a graphs/ path, or any
# other path that exists.
if [ ! -f "$YAML" ] && [ -f "graphs/$YAML" ]; then
    YAML="graphs/$YAML"
fi

if [ ! -f "$YAML" ]; then
    echo "error: no such yaml: $FES_DIR/$YAML" >&2
    echo "available:" >&2
    ls -1 graphs/*.yaml | sed 's/^/  /' >&2
    exit 1
fi

# ------------------------------------------------------------------- run id

# Computed once, here, and inherited by every actor process, so they all agree on one output folder even if
# setup crosses a minute boundary. See actors/run_paths.py.
export IMPROV_RUN_ID="${IMPROV_RUN_ID:-$(date +%Y%m%d-%H%M)}"

RUN_DIR="$(python -c "
import sys; sys.path.insert(0, '$FES_DIR')
from actors.run_paths import run_folder, log_folder
log_folder()
print(run_folder())
")" || { echo "error: could not resolve the run folder" >&2; exit 1; }

LOG_DIR="$RUN_DIR/logs"
GLOBAL_LOG="$LOG_DIR/global.log"

echo "[fes] run id : $IMPROV_RUN_ID"
echo "[fes] output : $RUN_DIR"
echo "[fes] logs   : $LOG_DIR"

# ------------------------------------------------------------ master graph

# A yaml with a `fes:` block is expanded into a plain improv graph in the run folder (cameras, savers, connections).
if grep -qE '^fes:' "$YAML"; then
    RESOLVED="$RUN_DIR/graph.yaml"
    [ "$SAVE" = true ] && GRAPH_ARGS+=(--save-video) || GRAPH_ARGS+=(--no-save-video)
    python "$FES_DIR/scripts/fes_graph.py" "$YAML" "$RESOLVED" "${GRAPH_ARGS[@]}" \
        || { echo "error: could not expand $YAML" >&2; exit 1; }
    echo "[fes] graph  : $YAML -> $RESOLVED"
    YAML="$RESOLVED"
elif [ "${#GRAPH_ARGS[@]}" -gt 0 ]; then
    echo "error: --cameras / --record-only only apply to master graphs (a yaml with a fes: block)" >&2
    exit 1
fi

# ----------------------------------------------------------------- preflight

if [ "$CHECKS" -eq 1 ]; then
    echo
    echo "[fes] preflight"

    # Cameras. Counted against the yaml's Generator actors rather than the
    # config's active_cameras, since yamls routinely comment cameras out.
    want_cams=$(grep -cE '^\s*Generator[0-9]+:' "$YAML" || true)
    have_cams=$(lsusb -d 199e: 2>/dev/null | wc -l)
    if [ "$want_cams" -gt 0 ] && [ "$have_cams" -lt "$want_cams" ]; then
        echo "  ! $YAML wants $want_cams camera(s), USB shows $have_cams."
        echo "    Missing after a reboot? Unplug the camera, wait a few seconds, replug (or try another port)."
    elif [ "$want_cams" -gt 0 ]; then
        echo "  ok cameras: $have_cams visible, $want_cams wanted"
    fi

    # Camera USB link speed: one that fell back to USB2 (480 Mbit/s) can't carry its stream.
    slow_cams=0
    for d in /sys/bus/usb/devices/*; do
        [ -r "$d/idVendor" ] && [ "$(cat "$d/idVendor")" = 199e ] && [ "$(cat "$d/speed")" -lt 5000 ] && slow_cams=$((slow_cams + 1))
    done
    [ "$slow_cams" -gt 0 ] && echo "  ! $slow_cams camera(s) negotiated USB2 speed (480 Mbit/s) -- unplug and replug them"

    # libusb transfer pool: the 16 MB default can't feed 7 cameras at 60 fps (setup_realtime.sh on raises it).
    usbfs_mb=$(cat /sys/module/usbcore/parameters/usbfs_memory_mb 2>/dev/null || echo 0)
    if [ "$want_cams" -gt 1 ] && [ "$usbfs_mb" -lt 1000 ]; then
        echo "  ! usbfs_memory_mb is $usbfs_mb: too small for $want_cams cameras -- run  sudo scripts/setup_realtime.sh on"
    fi

    # Redis, if the yaml declares a port.
    redis_port=$(awk '/^redis_config:/{f=1;next} f&&/port:/{print $2;exit}' "$YAML")
    if [ -n "${redis_port:-}" ]; then
        if command -v redis-cli >/dev/null && \
           redis-cli -p "$redis_port" ping >/dev/null 2>&1; then
            echo "  ok redis on $redis_port"
        else
            echo "  . redis not answering on $redis_port (improv will start its own)"
        fi
    fi
fi

# ------------------------------------------------------------------- cleanup

if [ "$DRY_RUN" -eq 1 ]; then
    echo
    echo "[fes] dry run -- not launching. Would run:"
    if [ "$AUTO" -eq 1 ]; then
        echo "        python scripts/improv_drive.py --config $YAML --logfile $GLOBAL_LOG"
    else
        echo "        improv run -a $FES_DIR -f $GLOBAL_LOG $YAML"
    fi
    # The run folder was created while resolving paths; leave nothing behind.
    # rmdir only removes empty directories, so this cannot touch a real run.
    [ "$YAML" = "$RUN_DIR/graph.yaml" ] && rm -f "$RUN_DIR/graph.yaml"
    rmdir "$LOG_DIR" "$RUN_DIR" "$(dirname "$RUN_DIR")" 2>/dev/null
    exit 0
fi

echo
echo "[fes] improv cleanup"
# `improv cleanup` prompts "Is that okay [y/N]?" with no non-interactive flag.
printf 'y\n' | improv cleanup 2>&1 | sed 's/^/  /'

# ----------------------------------------------------------------------- run

echo
echo "[fes] CPUs: $FES_CPUS"

if [ "$AUTO" -eq 1 ]; then
    echo "[fes] launching $YAML (headless)"
    taskset -c "$FES_CPUS" python "$FES_DIR/scripts/improv_drive.py" \
        --config "$YAML" \
        --actor-path "$FES_DIR" \
        --logfile "$GLOBAL_LOG" \
        --run-folder "$RUN_DIR"
    status=$?
else
    echo "[fes] launching $YAML"
    echo "[fes] in the improv console: setup -> (wait for ready) -> run -> stop -> quit"
    taskset -c "$FES_CPUS" improv run -a "$FES_DIR" -f "$GLOBAL_LOG" "$YAML"
    status=$?
fi

# -------------------------------------------------------------------- report

echo
if [ -d "$RUN_DIR" ]; then
    npy_count=$(find "$RUN_DIR" -maxdepth 1 -name '*.npy' | wc -l)
    log_count=$(find "$LOG_DIR" -maxdepth 1 -name '*.log' 2>/dev/null | wc -l)
    echo "[fes] $RUN_DIR"
    echo "[fes]   $npy_count .npy file(s), $log_count log file(s)"
    if [ "$npy_count" -eq 0 ]; then
        echo "[fes]   ! nothing was recorded. Check $GLOBAL_LOG"
    fi
    # The raw video for this run, if any saver ran.
    video_dir="$HOME/camera_video/${IMPROV_RUN_ID:0:4}-${IMPROV_RUN_ID:4:2}-${IMPROV_RUN_ID:6:2}/${IMPROV_RUN_ID:9:4}00"
    if [ -d "$video_dir" ]; then
        echo "[fes]   video: $video_dir ($(du -sh "$video_dir" | cut -f1))"
    fi
else
    echo "[fes] ! run folder was never created: $RUN_DIR"
fi

exit "$status"
