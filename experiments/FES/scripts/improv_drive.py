#!/usr/bin/env python
"""Drive a headless improv run without the TUI.

`improv run` is just `improv server` in a subprocess plus `improv client` (the
Textual TUI) in the foreground, and the TUI's only job is to send plain command
strings over a ZMQ REQ socket. Nexus.remote_input() reads them straight off the
wire, so the whole interactive protocol is four words -- "setup", "run",
"stop", "quit" -- and can be scripted.

This runs the server directly and sends those four itself, pausing for you at
the two points that need a human: when to start the experiment, and when to end
it. Everything else (waiting for actors to be ready, the 5 s pause after stop
so actors finish flushing their .npy files, reaping the server) is automatic.

    python improv_drive.py --config latency_benchmarking.yaml --logfile <path>

Prefer scripts/fes-run.sh, which sets up the environment and the run id first.
"""

import argparse
import os
import re
import signal
import socket
import subprocess
import sys
import time

import zmq

#: Nexus logs this once every actor has reported ready. There is no message on
#: the output socket for it, so the log is the signal.
READY_MARKER = "Allowing start"

#: Printed by `improv server` (and logged) once its sockets are bound.
STARTED_MARKER = "Server running on"

#: Seconds to wait after "stop" before "quit". Actors flush their recorded
#: arrays in stop(); quitting straight away terminates them mid-write.
STOP_DRAIN_SECONDS = 5.0

REQUEST_TIMEOUT_MS = 10_000


def free_port():
    """An OS-assigned free TCP port, released immediately."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class LogTail:
    """Incremental reader for the improv logfile."""

    def __init__(self, path):
        self.path = path
        self.offset = 0

    def read_new(self):
        try:
            with open(self.path, "r", errors="replace") as fh:
                fh.seek(self.offset)
                text = fh.read()
                self.offset = fh.tell()
                return text
        except FileNotFoundError:
            return ""

    def wait_for(self, marker, timeout, proc=None, echo=False):
        """Block until `marker` appears in the log. True if it did."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            text = self.read_new()
            if echo and text.strip():
                for line in text.splitlines():
                    if line.strip():
                        print(f"  | {line}", flush=True)
            if marker in text:
                return True
            if proc is not None and proc.poll() is not None:
                return False
            time.sleep(0.1)
        return False


class Driver:
    def __init__(self, config, logfile, actor_path=None):
        self.config = config
        self.logfile = logfile
        self.actor_path = actor_path
        self.control_port = free_port()
        self.output_port = free_port()
        self.logging_port = free_port()
        self.tail = LogTail(logfile)
        self.server = None
        self.ctx = None
        self.control = None

    # -- lifecycle ---------------------------------------------------------

    def start_server(self, timeout=60):
        cmd = [
            "improv", "server",
            "-c", str(self.control_port),
            "-o", str(self.output_port),
            "-l", str(self.logging_port),
            "-f", self.logfile,
        ]
        for path in self.actor_path or []:
            cmd += ["-a", path]
        cmd.append(self.config)

        print(f"[drive] starting server on control port {self.control_port}")
        with open(self.logfile, "a+") as fh:
            self.server = subprocess.Popen(cmd, stdout=fh, stderr=fh)

        if not self.tail.wait_for(STARTED_MARKER, timeout, proc=self.server):
            self.dump_log_tail()
            raise RuntimeError(
                f"improv server did not start within {timeout}s "
                f"(see {self.logfile})"
            )

        self.ctx = zmq.Context()
        self.control = self.ctx.socket(zmq.REQ)
        self.control.connect(f"tcp://127.0.0.1:{self.control_port}")
        print("[drive] server up")

    def send(self, command):
        """Send one command string and wait for Nexus's acknowledgement."""
        self.control.send_string(command)
        if self.control.poll(REQUEST_TIMEOUT_MS) & zmq.POLLIN:
            self.control.recv_multipart()
            return True
        # A REQ socket that timed out is stuck in the wrong state; rebuild it
        # so a later command (notably "quit") still has a chance.
        print(f"[drive] no reply to '{command}' after "
              f"{REQUEST_TIMEOUT_MS // 1000}s", file=sys.stderr)
        self.control.setsockopt(zmq.LINGER, 0)
        self.control.close()
        self.control = self.ctx.socket(zmq.REQ)
        self.control.connect(f"tcp://127.0.0.1:{self.control_port}")
        return False

    def dump_log_tail(self, lines=25):
        try:
            with open(self.logfile, "r", errors="replace") as fh:
                tail = fh.readlines()[-lines:]
        except OSError:
            return
        print(f"--- last {len(tail)} lines of {self.logfile} ---", file=sys.stderr)
        for line in tail:
            print("  " + line.rstrip(), file=sys.stderr)

    def shutdown(self, timeout=30):
        """Send quit, reap the server, fall back to improv cleanup."""
        if self.control is not None:
            self.send("quit")
        if self.server is not None:
            try:
                self.server.wait(timeout=timeout)
                print(f"[drive] server exited ({self.server.returncode})")
            except subprocess.TimeoutExpired:
                print("[drive] server did not exit; terminating", file=sys.stderr)
                self.server.terminate()
                try:
                    self.server.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    self.server.kill()
                self.cleanup()
        if self.control is not None:
            self.control.setsockopt(zmq.LINGER, 0)
            self.control.close()
        if self.ctx is not None:
            self.ctx.term()

    @staticmethod
    def cleanup():
        print("[drive] running improv cleanup")
        subprocess.run(["improv", "cleanup"], input="y\n", text=True, timeout=60)


def prompt(message):
    """Wait for ENTER. Ctrl-C and EOF both mean 'stop now', not 'crash'."""
    try:
        input(f"\n>>> {message} ")
        return True
    except (EOFError, KeyboardInterrupt):
        print()
        return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="improv yaml to run")
    parser.add_argument("--logfile", required=True, help="path for global.log")
    parser.add_argument("--actor-path", action="append", default=[])
    parser.add_argument("--setup-timeout", type=float, default=300.0,
                        help="seconds to wait for all actors to report ready")
    parser.add_argument("--run-folder", default=None,
                        help="printed on exit so you know where output landed")
    args = parser.parse_args()

    driver = Driver(args.config, args.logfile, args.actor_path)

    # Ctrl-C at any point should still take the pipeline down cleanly.
    def on_sigint(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, on_sigint)

    try:
        driver.start_server()

        print("[drive] setup: starting actors (this loads the DLC models)")
        driver.send("setup")
        if not driver.tail.wait_for(READY_MARKER, args.setup_timeout,
                                    proc=driver.server, echo=True):
            driver.dump_log_tail()
            raise RuntimeError(
                f"actors were not all ready within {args.setup_timeout}s. "
                f"A camera that failed to open is the usual reason -- check "
                f"the log above and try scripts/usb-camera-rebind.sh."
            )
        print("[drive] all actors ready")

        if prompt("Press ENTER to START the experiment (Ctrl-C to abort)"):
            driver.send("run")
            print("[drive] running")
            if prompt("Press ENTER to STOP the experiment"):
                pass
            driver.send("stop")
            print(f"[drive] stopping; waiting {STOP_DRAIN_SECONDS:.0f}s for "
                  f"actors to finish writing")
            time.sleep(STOP_DRAIN_SECONDS)
        else:
            print("[drive] aborted before run")

    except KeyboardInterrupt:
        print("\n[drive] interrupted", file=sys.stderr)
    except Exception as e:
        print(f"[drive] {e}", file=sys.stderr)
        driver.shutdown()
        return 1

    driver.shutdown()

    if args.run_folder:
        print(f"\n[drive] output: {args.run_folder}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
