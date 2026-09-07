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
# Not done here, deliberately:
#   - isolcpus / nohz_full would need a kernel cmdline change and a reboot.
#   - CUDA MPS is a separate opt-in; see MULTICAM_3D_PLAN.md.

set -euo pipefail

MODE="${1:-on}"
IRQ_TARGET_CPU=31          # an E-core, well away from the pinned P-cores
IRQ_MATCH="xhci_hcd"

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

echo
echo "Done ($MODE). Verify during a run with:"
echo "  grep -E 'MHz' /proc/cpuinfo | sort -u | head"
echo "  grep xhci /proc/interrupts"
echo "  for p in \$(pgrep -f actors.processor); do echo -n \"\$p: \"; taskset -cp \$p; done"
