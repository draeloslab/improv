import time
import os
import ipaddress
import socket
import json
import numpy as np
import logging
import yaml
from pathlib import Path
from improv.actor import Actor
from . import cpu_affinity

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Create a file handler
log_file = "sender_udp.log"
file_handler = logging.FileHandler(log_file)
file_handler.setLevel(logging.INFO)

# Create a formatter and set it for the handler
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)

# Add the handler to the logger
logger.addHandler(file_handler)

class SenderUDP(Actor):
    """Actor to send data over UDP.

    Reads from preds{0..max_camera_slots-1}_in, whichever of those are actually
    wired in the yaml for this run (1 to 4 cameras -- unwired slots simply never
    yield a message and are skipped every step at negligible cost). Sends a UDP
    packet only when at least one camera produced a fresh prediction this step
    (mirrors the fix applied to the UART Sender's saturation bug).

    Every camera is identified by its real camera_num (from
    actors/config/camera_config.yaml's active_cameras / config.yaml's
    model_path_N), NOT by which preds{N}_in slot it happens to be wired to --
    slot index is just wiring order in the yaml (e.g. Processor1 can have
    camera_num=3 while still being wired to preds1_in), so it is not a stable
    or meaningful identity to hand to a downstream consumer.

    Wire format: a 2-element JSON array `[frame_index, angles]`.
      - frame_index: this sender's own outgoing packet counter (0, 1, 2, ...).
        A single scalar regardless of how many cameras contributed this step,
        so a receiver can detect drops/reordering without needing to reason
        about per-camera frame numbers.
      - angles: a JSON object mapping camera_num (as a string key, e.g. "0",
        "3") to its most recently known angle. Only cameras that have produced
        at least one prediction so far this run appear as keys, so the same
        object shape naturally handles a 1-, 2-, 3-, or 4-camera run with no
        padding/nulls for cameras that aren't part of this experiment.
        Example, 2 active cameras (0 and 3): [1523, {"0": 142.7, "3": 88.1}]
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def setup(self):
        logger.info("Beginning setup for SenderUDP")

        # The sender is the last hop before the wire, but its whole step is a
        # few queue polls, a json.dumps and a sendto -- ~0.1 ms. The ~0.07 ms
        # an E-core adds to that is far less than the 5+ ms a processor loses
        # by being displaced off a P-core, so this belongs in the background
        # pool with the savers and the GUI.
        cpu_affinity.pin_actor(cpu_affinity.BACKGROUND, label="SenderUDP")

        # UDP connection parameters
        self.UDP_IP_send = os.getenv("SENDER_UDP_IP", "192.168.137.201")
        self.UDP_PORT_send = int(os.getenv("SENDER_UDP_PORT", "11115"))

        try:
            resolved_ip = str(ipaddress.ip_address(self.UDP_IP_send))
        except ValueError as exc:
            raise ValueError(
                f"Invalid UDP destination IP {self.UDP_IP_send!r}; use a dotted IPv4/IPv6 address"
            ) from exc

        if not (0 < self.UDP_PORT_send < 65536):
            raise ValueError(f"Invalid UDP destination port {self.UDP_PORT_send}")

        # Create UDP socket
        self.sock_send = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        print(f"UDP socket created for sending to {resolved_ip}:{self.UDP_PORT_send}")

        # Load the configuration file
        source_folder = Path(__file__).resolve().parent.parent
        with open(f'{source_folder}/config.yaml', 'r') as file:
            config = yaml.safe_load(file)

        # How many preds{N}_in slots to poll each runStep. Independent of how
        # many cameras are actually wired in this particular yaml -- an unwired
        # slot's self.links[...].get() just raises every step and is skipped.
        # Bump this if a yaml ever wires more than 4 Processor actors.
        self.max_camera_slots = config.get('max_camera_slots', 4)

        self.packet_n = 0
        self.skipped_steps = 0  # runStep calls where no camera had a fresh prediction

        # --- Per-camera-number state --------------------------------------
        # Everything below is keyed by camera_num, discovered lazily the first
        # time a message from that camera arrives (see _ensure_camera). This
        # is what makes the actor adapt to 1..4 cameras without code changes:
        # nothing here has a fixed size, and there is no cam0/cam2-specific
        # code path left anywhere in this actor.
        self.last_angle = {}       # camera_num -> most recently known angle
        self.last_frame_num = {}   # camera_num -> most recently known frame_num
        self.angle_history = {}    # camera_num -> list, for robust min/max at stop()

        self.sent_angles = {}      # camera_num -> list, one entry per SENT packet
        self.sent_frame_nums = {}  # camera_num -> list, one entry per SENT packet
        self.fresh = {}            # camera_num -> list of bool, one per SENT packet

        # True end-to-end: one entry per FRESH prediction (not per packet)
        self.true_e2e = {}
        self.true_e2e_frame_nums = {}
        self.true_e2e_timestamps = {}

        self.send_timestamps = []
        self.step_latencies = []

        # Percentiles for robust min/max (filters outliers)
        self.min_percentile = 1
        self.max_percentile = 99

        date = time.strftime("%Y%m%d")
        timestamp = time.strftime("%Y%m%d-%H%M")
        string = config['output_path']
        self.out_folder = Path(f"{string}/{date}/{timestamp}")
        self.out_folder.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output folder set to {self.out_folder}")

        logger.info(f"UDP Sender configured to send to {resolved_ip}:{self.UDP_PORT_send}, "
                    f"polling up to {self.max_camera_slots} preds{{N}}_in slots")
        logger.info("Completed setup for SenderUDP")

    def _ensure_camera(self, cam):
        """Lazily create per-camera-number bookkeeping the first time cam is seen."""
        if cam not in self.last_angle:
            self.last_angle[cam] = 0.0
            self.last_frame_num[cam] = -1
            self.angle_history[cam] = []
            self.sent_angles[cam] = []
            self.sent_frame_nums[cam] = []
            self.fresh[cam] = []
            self.true_e2e[cam] = []
            self.true_e2e_frame_nums[cam] = []
            self.true_e2e_timestamps[cam] = []
            logger.info(f"SenderUDP: new camera_num {cam} seen for the first time")

    def runStep(self):
        """Poll every wired preds{N}_in slot; send ONLY if at least one camera
        produced a fresh prediction this step."""
        step_start = time.perf_counter()
        fresh_this_step = {}  # camera_num -> (angle, camera_start, frame_num)

        for slot in range(self.max_camera_slots):
            try:
                element = self.links[f"preds{slot}_in"].get(timeout=0.0001)
            except Exception:
                continue  # slot not wired this run, or nothing new -- either way, skip

            # Current format: [pred, angle, camera_start, frame_num, camera_num].
            # Falls back gracefully for older/shorter messages, using the slot
            # index as a best-effort camera identity if camera_num isn't present.
            if len(element) >= 5:
                _, angle, camera_start, frame_num, cam = element[:5]
            elif len(element) == 4:
                _, angle, camera_start, frame_num = element
                cam = slot
            elif len(element) == 2:
                _, angle = element
                camera_start, frame_num, cam = None, -1, slot
            else:
                continue
            if cam is None:
                cam = slot

            self._ensure_camera(cam)
            fresh_this_step[cam] = (angle, camera_start, frame_num)

        # --- Only send when there is something new. This is the fix: the UART
        # Sender writes unconditionally every step (measured at 99.5% wire
        # utilization, ~1147 pkt/s, of which only ~2.6% carried a fresh angle).
        # UDP has no serial backpressure to hide that behind, so without this
        # gate the pipeline would flood the socket at the actor's full spin
        # rate instead of at the ~30 Hz the predictions actually arrive at.
        if not fresh_this_step:
            self.skipped_steps += 1
            return

        for cam, (angle, _camera_start, frame_num) in fresh_this_step.items():
            self.last_angle[cam] = angle
            self.last_frame_num[cam] = frame_num
            self.angle_history[cam].append(angle)

        if self.packet_n % 10000 == 0:
            logger.info(f"Sending angles: "
                        f"{ {c: self.last_angle[c] for c in sorted(self.last_angle)} }")

        # --- Build and send the packet ---
        # angles keyed by camera_num (string, JSON object keys must be strings);
        # frame_index is this sender's own outgoing packet counter, not any
        # per-camera frame number -- see the class docstring.
        angles_out = {str(cam): float(self.last_angle[cam]) for cam in sorted(self.last_angle)}
        payload = [self.packet_n, angles_out]

        try:
            data_bytes = json.dumps(payload).encode('utf-8')
            self.sock_send.sendto(data_bytes, (self.UDP_IP_send, self.UDP_PORT_send))
            logger.debug(f"Sent {len(data_bytes)} bytes via UDP to "
                         f"{self.UDP_IP_send}:{self.UDP_PORT_send}: {payload}")
        except Exception as e:
            logger.error(f"Error sending UDP data: {e}")
            return

        send_time = time.time()
        self.packet_n += 1

        # --- Log timing data, per camera_num ---
        self.send_timestamps.append(send_time)
        for cam in self.last_angle:
            is_fresh = cam in fresh_this_step
            self.sent_angles[cam].append(self.last_angle[cam])
            self.sent_frame_nums[cam].append(self.last_frame_num[cam])
            self.fresh[cam].append(is_fresh)
            if is_fresh:
                _, camera_start, frame_num = fresh_this_step[cam]
                if camera_start is not None:
                    self.true_e2e[cam].append(send_time - camera_start)
                    self.true_e2e_frame_nums[cam].append(frame_num)
                    self.true_e2e_timestamps[cam].append(send_time)

        self.step_latencies.append(time.perf_counter() - step_start)

    def stop(self):
        logger.info("Stopping SenderUDP")

        for cam in sorted(self.angle_history):
            hist = self.angle_history[cam]
            if len(hist) > 0:
                lo = np.percentile(hist, self.min_percentile)
                hi = np.percentile(hist, self.max_percentile)
                logger.info(f"Camera {cam} angle — robust min: {lo}, robust max: {hi} "
                            f"(from {len(hist)} samples)")
            else:
                logger.info(f"Camera {cam}: no angle data collected")

        try:
            self.sock_send.close()
        except Exception as e:
            logger.error(f"Error closing socket: {e}")

        # --- Save all timing/data logs, one set of files per camera_num ---
        # Filenames are `_cam{N}.npy` where N is the REAL camera_num, so a
        # 1- or 2-camera run using camera_num 0 (and 2) reproduces the exact
        # filenames actors.sender.Sender always wrote -- existing analysis
        # tooling (freebie's loader) keeps working unmodified for those cases.
        # Any other camera_num used (1, 3, ...) gets its own files the same way.
        np.save(self.out_folder / "sender_timestamps.npy", self.send_timestamps)
        np.save(self.out_folder / "sender_step_latencies.npy", self.step_latencies)

        for cam in sorted(self.sent_angles):
            np.save(self.out_folder / f"sender_sent_angles_cam{cam}.npy", self.sent_angles[cam])
            np.save(self.out_folder / f"sender_sent_frame_nums_cam{cam}.npy", self.sent_frame_nums[cam])
            np.save(self.out_folder / f"sender_fresh_cam{cam}.npy", self.fresh[cam])

            np.save(self.out_folder / f"true_e2e_cam{cam}.npy", self.true_e2e[cam])
            np.save(self.out_folder / f"true_e2e_cam{cam}_frame_nums.npy", self.true_e2e_frame_nums[cam])
            np.save(self.out_folder / f"true_e2e_cam{cam}_timestamps.npy", self.true_e2e_timestamps[cam])

            e2e = self.true_e2e[cam]
            if len(e2e) > 0:
                logger.info(f"True E2E cam{cam}: mean={np.mean(e2e)*1000:.1f}ms, "
                            f"median={np.median(e2e)*1000:.1f}ms, "
                            f"max={np.max(e2e)*1000:.1f}ms (from {len(e2e)} frames)")

        # Legacy file names for backward compatibility with tooling that
        # assumed a single primary camera (camera_num 0).
        legacy_e2e = self.true_e2e.get(0, [])
        np.save(self.out_folder / "endtoendLatencies.npy", legacy_e2e)
        np.save(self.out_folder / "senderStartTimes.npy", self.send_timestamps)

        logger.info(f"Total UDP packets sent: {self.packet_n}")
        logger.info(f"Steps skipped (no fresh data from any camera): {self.skipped_steps}")
        for cam in sorted(self.fresh):
            logger.info(f"Fresh cam{cam} predictions: {sum(self.fresh[cam])} / {len(self.fresh[cam])}")

        logger.info("SenderUDP stopped")

if __name__ == "__main__":
    # For testing purposes, you can instantiate and run the actor here.
    sender = SenderUDP('SenderUDP')
    sender.setup()
    try:
        while True:
            sender.runStep()
            time.sleep(0.01)
    except KeyboardInterrupt:
        sender.stop()
