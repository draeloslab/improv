#!/usr/bin/env bash
#
# Start an FES experiment with one command.
#
# Replaces this sequence:
#   conda activate improvPytorch2
#   cd improv/experiments/FES
#   rm *.log                      # because everything appended to one file
#   improv cleanup
#   improv run latency_benchmarking.yaml
#   ...setup / run / stop / quit...
#   copy the logs into the predictions folder by hand
#
# The log deletion and the copying are both gone: this picks a run id up front,
# exports it as $IMPROV_RUN_ID, and every actor writes its logs and its .npy
# files straight into ~/predictions/<date>/<run id>/. One folder per run, no
# cleanup, nothing to move afterwards.
#
# USAGE
#   fes-run                              # latency_benchmarking.yaml, with the TUI
#   fes-run 5cam_test.yaml               # any yaml in experiments/FES
#   fes-run --auto                       # headless: prompts ENTER to start/stop
#   fes-run --auto 5cam_test.yaml
#   fes-run --no-check 5cam_test.yaml    # skip the camera/redis/link preflight
#   fes-run --dry-run 5cam_test.yaml     # preflight only, launch nothing
#
set -uo pipefail

FES_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_ENV="${FES_CONDA_ENV:-improvPytorchJarvis}"
DEFAULT_YAML="latency_benchmarking.yaml"

AUTO=0
CHECKS=1
DRY_RUN=0
YAML=""

while [ $# -gt 0 ]; do
    case "$1" in
        --auto)     AUTO=1 ;;
        --no-check) CHECKS=0 ;;
        --dry-run)  DRY_RUN=1 ;;
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

if [ ! -f "$YAML" ]; then
    echo "error: no such yaml: $FES_DIR/$YAML" >&2
    echo "available:" >&2
    ls -1 ./*.yaml | sed 's/^/  /' >&2
    exit 1
fi

# ------------------------------------------------------------------- run id

# Computed once, here, and inherited by every actor process. This is what makes
# them all agree on one output folder -- previously each actor called strftime
# in its own setup(), so a run whose setup crossed a minute boundary split its
# outputs across two folders. See actors/run_paths.py.
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
        echo "    Post-reboot cameras often need re-enumerating:  usb-camera-rebind"
    elif [ "$want_cams" -gt 0 ]; then
        echo "  ok cameras: $have_cams visible, $want_cams wanted"
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

    # xPC link, only when the yaml actually uses the Receiver.
    if grep -qE '^\s*Receiver:' "$YAML"; then
        if python -m actors.xpc_link >/dev/null 2>&1; then
            echo "  ok xPC link"
        else
            echo "  ! xPC link is NOT ready -- the Receiver will get no packets:"
            python -m actors.xpc_link 2>&1 | sed 's/^/    /'
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
        echo "        improv run -f $GLOBAL_LOG $YAML"
    fi
    # The run folder was created while resolving paths; leave nothing behind.
    # rmdir only removes empty directories, so this cannot touch a real run.
    rmdir "$LOG_DIR" "$RUN_DIR" "$(dirname "$RUN_DIR")" 2>/dev/null
    exit 0
fi

echo
echo "[fes] improv cleanup"
# `improv cleanup` prompts "Is that okay [y/N]?" with no non-interactive flag.
printf 'y\n' | improv cleanup 2>&1 | sed 's/^/  /'

# ----------------------------------------------------------------------- run

echo
if [ "$AUTO" -eq 1 ]; then
    echo "[fes] launching $YAML (headless)"
    python "$FES_DIR/scripts/improv_drive.py" \
        --config "$YAML" \
        --logfile "$GLOBAL_LOG" \
        --run-folder "$RUN_DIR"
    status=$?
else
    echo "[fes] launching $YAML"
    echo "[fes] in the improv console: setup -> (wait for ready) -> run -> stop -> quit"
    improv run -f "$GLOBAL_LOG" "$YAML"
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
        echo "[fes]   convert it later with:  convert-day ${IMPROV_RUN_ID:0:4}-${IMPROV_RUN_ID:4:2}-${IMPROV_RUN_ID:6:2}"
    fi
else
    echo "[fes] ! run folder was never created: $RUN_DIR"
fi

exit "$status"
