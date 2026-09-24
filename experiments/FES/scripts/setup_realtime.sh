#!/usr/bin/env bash
# Host-level tuning for the FES pipeline. Run once before a session:
#
#     sudo ./scripts/setup_realtime.sh on
#     ... run improv ...
#     sudo ./scripts/setup_realtime.sh off
#
# Everything here is a runtime setting that reverts on reboot, so "off" is a
# convenience, not a requirement. Idempotent -- safe to run twice.
#
# What each step is worth, measured on this machine:
#
#   governor   The pipeline is duty-cycled: each actor works ~12 ms then
#              blocks. intel_pstate reads that as an idle machine and drops
#              the clock. Same GPU work, sleeping between iterations: 21.72 ms
#              @ 1168 MHz. Busy-spinning instead: 7.68 ms @ 3246 MHz. Pinning
#              the governor to `performance` is what stops the pipeline being
#              punished for finishing early.
#
#   USB IRQ    All ~750k xhci interrupts for the cameras land on cpu13, which
#              is one SMT thread of physical P-core 6 -- a core the pinning in
#              actors/cpu_affinity.py wants to hand to a CameraReader. Moving
#              it to cpu31 (an E-core) frees the core and stops interrupt
#              delivery preempting a latency-critical actor.
#
#   GPU clock  Same duty-cycle problem as the CPU governor, one layer down:
#              DLC inference runs ~15 ms then the GPU sits idle for the rest
#              of the ~33 ms frame period. Without persistence mode the driver
#              lets the clock fall in that gap (idle default ~210 MHz here vs
#              a 3105 MHz max) and pays a ramp-up cost on the next inference
#              call. `-pm 1` keeps the driver/clocks loaded between calls;
#              locking the clock with `-lgc` stops it dropping at all.
#
#   Wi-Fi      The cameras and the xPC link are both on wired, private
#              subnets (eno1/eno2) -- nothing in the pipeline needs the Wi-Fi
#              uplink. Leaving it up costs nothing most of the time, but
#              NetworkManager's periodic connectivity-check ping and any
#              background sync/update traffic on that interface are both
#              unrelated, un-budgeted CPU/network activity during a benchmark.
#
#   Timers     A handful of systemd timers (sysstat every 10 min, fwupd,
#              anacron, the daily apt timers, man-db) do real CPU/disk/network
#              work on their own schedule, with no regard for whether a
#              latency-critical recording is in progress.
#
# Not done here, deliberately:
#   - isolcpus / nohz_full would need a kernel cmdline change and a reboot.
#   - CUDA MPS is a separate opt-in; see MULTICAM_3D_PLAN.md.

set -euo pipefail

MODE="${1:-on}"
IRQ_TARGET_CPU=31          # an E-core, well away from the pinned P-cores
IRQ_MATCH="xhci_hcd"
QUIET_TIMERS=(sysstat fwupd-refresh.timer anacron.timer apt-daily.timer
              apt-daily-upgrade.timer man-db.timer)

if [[ $EUID -ne 0 ]]; then
    echo "error: needs root (sudo $0 $MODE)" >&2
    exit 1
fi

case "$MODE" in
  on)  GOV=performance ;;
  off) GOV=powersave ;;
  *)   echo "usage: $0 [on|off]" >&2; exit 1 ;;
esac

echo "== CPU governor -> $GOV"
for g in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
    echo "$GOV" > "$g"
done
echo "   now: $(cut -d' ' -f1 /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor) "\
"(cpu0), $(cat /sys/devices/system/cpu/cpu16/cpufreq/scaling_governor) (cpu16)"

# The `performance` governor still lets the CPU drop into deep C-states when
# every actor blocks, and coming back out of C6 costs more than the work does.
# Holding /dev/cpu_dma_latency open at 0 keeps the package shallow. The `on`
# branch backgrounds a holder process; `off` kills it.
PID_FILE=/run/fes_cpu_dma_latency.pid
if [[ "$MODE" == "on" ]]; then
    if [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
        echo "== C-state floor already held (pid $(cat "$PID_FILE"))"
    else
        echo "== Holding /dev/cpu_dma_latency at 0 (blocks deep C-states)"
        python3 -c "
import struct, time, signal
f = open('/dev/cpu_dma_latency', 'wb')
f.write(struct.pack('i', 0)); f.flush()
signal.pause()
" &
        echo $! > "$PID_FILE"
        sleep 0.2
    fi
else
    if [[ -f "$PID_FILE" ]]; then
        kill "$(cat "$PID_FILE")" 2>/dev/null || true
        rm -f "$PID_FILE"
        echo "== Released C-state floor"
    fi
fi

echo "== USB (xhci) interrupt affinity"
for irq_dir in /proc/irq/*/; do
    irq="$(basename "$irq_dir")"
    [[ "$irq" =~ ^[0-9]+$ ]] || continue
    # The device name shows up as a subdirectory under /proc/irq/<n>/
    if ls "$irq_dir" 2>/dev/null | grep -q "$IRQ_MATCH"; then
        if [[ "$MODE" == "on" ]]; then
            if echo "$IRQ_TARGET_CPU" > "$irq_dir/smp_affinity_list" 2>/dev/null; then
                echo "   IRQ $irq ($IRQ_MATCH) -> cpu$IRQ_TARGET_CPU"
            else
                echo "   IRQ $irq ($IRQ_MATCH): affinity is managed by the kernel, left alone"
            fi
        else
            # Restore the default: allow every online CPU.
            if echo "0-$(($(nproc) - 1))" > "$irq_dir/smp_affinity_list" 2>/dev/null; then
                echo "   IRQ $irq ($IRQ_MATCH) -> all CPUs"
            fi
        fi
    fi
done

# irqbalance will happily undo the above on its next pass.
if systemctl is-active --quiet irqbalance 2>/dev/null; then
    if [[ "$MODE" == "on" ]]; then
        systemctl stop irqbalance
        echo "== Stopped irqbalance (it would migrate the IRQ back)"
    fi
elif [[ "$MODE" == "off" ]]; then
    systemctl start irqbalance 2>/dev/null && echo "== Restarted irqbalance" || true
fi

if command -v nvidia-smi >/dev/null 2>&1; then
    echo "== GPU clocks"
    if [[ "$MODE" == "on" ]]; then
        if nvidia-smi -pm 1 >/dev/null 2>&1; then
            MAX_SM=$(nvidia-smi --query-gpu=clocks.max.sm --format=csv,noheader,nounits 2>/dev/null | head -1)
            if [[ -n "$MAX_SM" ]] && nvidia-smi -lgc "$MAX_SM,$MAX_SM" >/dev/null 2>&1; then
                echo "   persistence on, SM clock locked at ${MAX_SM} MHz"
            else
                echo "   persistence on; could not lock SM clock (driver/permissions?)"
            fi
        else
            echo "   could not enable persistence mode (driver/permissions?), leaving GPU clocks alone"
        fi
    else
        nvidia-smi -rgc >/dev/null 2>&1 || true
        nvidia-smi -pm 0 >/dev/null 2>&1 || true
        echo "   clock lock released, persistence off"
    fi
else
    echo "== GPU clocks: no nvidia-smi found, skipping"
fi

# if command -v nmcli >/dev/null 2>&1; then
#     if [[ "$MODE" == "on" ]]; then
#         nmcli radio wifi off 2>/dev/null && echo "== Wi-Fi radio off (cameras/xPC link are wired, unaffected)" \
#             || echo "== Wi-Fi radio: could not turn off (already off, or no Wi-Fi hardware)"
#     else
#         nmcli radio wifi on 2>/dev/null && echo "== Wi-Fi radio back on" || true
#     fi
# else
#     echo "== Wi-Fi radio: nmcli not found, skipping"
# fi

echo "== Background timers"
for t in "${QUIET_TIMERS[@]}"; do
    if systemctl list-unit-files "$t*" 2>/dev/null | grep -q "$t"; then
        if [[ "$MODE" == "on" ]]; then
            systemctl stop "$t" 2>/dev/null && echo "   stopped $t" || true
        else
            systemctl start "$t" 2>/dev/null && echo "   restarted $t" || true
        fi
    fi
done

echo
echo "Done ($MODE). Verify during a run with:"
echo "  grep -E 'MHz' /proc/cpuinfo | sort -u | head"
echo "  grep xhci /proc/interrupts"
echo "  for p in \$(pgrep -f actors.processor); do echo -n \"\$p: \"; taskset -cp \$p; done"
echo "  nvidia-smi --query-gpu=persistence_mode,clocks.sm,clocks.max.sm --format=csv"
echo "  nmcli radio wifi"
echo "  systemctl list-timers --all | head -20"
