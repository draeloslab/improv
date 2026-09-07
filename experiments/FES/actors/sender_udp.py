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

from .run_paths import get_logger, run_folder

logger = get_logger(__name__, "sender_udp.log")

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

    3D mode (actors/processor_batch3d.ProcessorBatch3D)
    ---------------------------------------------------
    When a `joints_in` link is wired, this actor also polls it for the single
    dict ProcessorBatch3D emits per step, and the keys in `angles` become JOINT
    NAMES rather than camera numbers:

        [1523, {"INDEX_PIP": 41.2, "INDEX_MCP": 12.8, "THUMB_IP": 7.4, ...}]

    The two modes coexist deliberately: the payload shape (a flat
    string-keyed object of angles) is identical, so a receiver that already
    parses the per-camera form needs no change to consume joint angles. If a
    yaml wires BOTH `joints_in` and the per-camera `preds{N}_in` slots, joint
    angles win the payload from the first joint message onward -- the two key
    spaces cannot be merged without ambiguity, and 3D joint angles are the more
    specific signal. Per-camera angles keep being recorded to disk either way.
    Joint angles are in degrees, deviation from
    straight (0 = fully extended), from dlc2kinematics. A joint that could not
    be triangulated this frame (fewer than two confident views) is sent as
    null, NOT dropped or zero-filled -- a receiver must be able to tell "not
    measured" from "measured as 0", which is a perfectly normal extended joint.
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

        # --- 3D joint-angle state (only used when `joints_in` is wired) ----
        self.last_joint_angles = {}   # joint name -> most recently known angle
        self.joint_names = []         # stable order, from the processor
        self.sent_joint_angles = []   # one row per SENT packet, in joint_names order
        self.joint_frame_nums = []
        self.joint_e2e = []
        self.joints_seen = False

        self.send_timestamps = []
        self.step_latencies = []

        # Percentiles for robust min/max (filters outliers)
        self.min_percentile = 1
        self.max_percentile = 99

        self.out_folder = run_folder()
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

    def _poll_joints(self):
        """Drain `joints_in` for the newest ProcessorBatch3D message.

        Returns the message that was consumed, or None. Drains rather than
        taking one per step: if the sender ever falls behind the processor,
        sending the freshest angles matters far more than sending every one --
        this is a control signal, not a recording (the processor already saves
        the full series to disk).
        """
        link = self.links.get("joints_in")
        if link is None:
            return None

        msg = None
        try:
            msg = link.get(timeout=0.0001)
        except Exception:
            return None
        while True:
            try:
                msg = link.get_nowait()
            except Exception:
                break

        if not isinstance(msg, dict):
            logger.warning(f"joints_in delivered a {type(msg).__name__}, expected dict; ignoring")
            return None

        angles = msg.get('joint_angles') or {}
        names = msg.get('joint_names')
        if names and not self.joint_names:
            self.joint_names = list(names)
            logger.info(f"SenderUDP: 3D mode, {len(self.joint_names)} joint angles: "
                        f"{self.joint_names}")
        elif not self.joint_names:
            self.joint_names = sorted(angles)

        self.joints_seen = True
        self.last_joint_angles.update(angles)

        camera_start = msg.get('camera_start')
        if camera_start is not None:
            self._pending_joint_start = camera_start
        self._pending_joint_frame = msg.get('frame_num', -1)
        return msg

    def runStep(self):
        """Poll every wired preds{N}_in slot; send ONLY if at least one camera
        produced a fresh prediction this step."""
        step_start = time.perf_counter()
        fresh_this_step = {}  # camera_num -> (angle, camera_start, frame_num)

        # --- 3D joint angles, if this yaml wires ProcessorBatch3D ---
        fresh_joints = self._poll_joints()

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
        if not fresh_this_step and not fresh_joints:
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
        if self.joints_seen:
            # 3D mode: keys are joint names. NaN is not valid JSON, and it means
            # "this joint had fewer than two confident views this frame" -- a
            # real, actionable state that must not be confused with 0 degrees
            # (a normally extended joint). Send it as null.
            angles_out = {
                name: (float(v) if v is not None and np.isfinite(v) else None)
                for name, v in ((n, self.last_joint_angles.get(n)) for n in self.joint_names)
            }
        else:
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

        if fresh_joints is not None:
            self.sent_joint_angles.append(
                [self.last_joint_angles.get(n, np.nan) for n in self.joint_names])
            self.joint_frame_nums.append(getattr(self, '_pending_joint_frame', -1))
            start = getattr(self, '_pending_joint_start', None)
            if start is not None:
                self.joint_e2e.append(send_time - start)

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

        # --- 3D joint angles (only present when joints_in was wired) ---
        if self.joints_seen:
            np.save(self.out_folder / "sender_joint_names.npy", np.asarray(self.joint_names))
            np.save(self.out_folder / "sender_joint_angles.npy",
                    np.asarray(self.sent_joint_angles, dtype=float))
            np.save(self.out_folder / "sender_joint_frame_nums.npy",
                    np.asarray(self.joint_frame_nums))
            np.save(self.out_folder / "sender_joint_e2e.npy", np.asarray(self.joint_e2e))
            if self.joint_e2e:
                logger.info(f"Joint-angle E2E: mean={np.mean(self.joint_e2e)*1000:.1f}ms, "
                            f"median={np.median(self.joint_e2e)*1000:.1f}ms, "
                            f"max={np.max(self.joint_e2e)*1000:.1f}ms "
                            f"(from {len(self.joint_e2e)} packets)")
            arr = np.asarray(self.sent_joint_angles, dtype=float)
            if arr.size:
                for i, name in enumerate(self.joint_names):
                    col = arr[:, i]
                    good = np.isfinite(col)
                    if good.any():
                        logger.info(f"  {name}: median {np.median(col[good]):6.1f} deg, "
                                    f"measured {good.sum()}/{len(col)} packets")
                    else:
                        logger.info(f"  {name}: never triangulated")

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
