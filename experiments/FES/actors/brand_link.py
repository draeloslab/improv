"""UDP link back from the BRAND machine: what stimulation actually ran, when, and the current target.

Wire protocol (JSON, one object per datagram). This machine -> BRAND, sent by SenderUDP from its
`stim_in` link to (SENDER_UDP_IP, brand_link.stim_port):

    {"type": "stim_request", "id": 17, "electrodes": [2, 4, 16], "amplitude": 3000, "pulse_width": 120,
     "frequency": 50, "duration": 1.5, "t_sent": <this machine's time.time()>}

BRAND -> this machine, to brand_link.feedback_port (this actor):

    {"type": "stim_on",  "id": 17, "electrodes": [...], "amplitude": ..., "pulse_width": ..., "frequency": ...,
                         "duration": ..., "t_brand": <BRAND time when cerestim reported stim=1>}
    {"type": "stim_off", "id": 17, "t_brand": ...}
    {"type": "stim_rejected", "id": 17, "reason": "amplitude over the limit"}
    {"type": "target", "fingers": {"index": 0.2, "middle": 0.8, ...}}      (or "joint_angles": {name: deg})

BRAND should send stim_on/stim_off when the stimulator reports the train starting/ending (cerestim_output
stim = 1 / 0), echoing the parameters it actually delivered, and send the target whenever it changes (and
about once a second, so a restarted improv run picks it up). The joint-angle stream on the existing port
(11115, [frame_index, {joint: deg}]) is unchanged.

Each message is stamped with its arrival time on this machine (`t_rx`, time.time() - the processor's
frame-time clock), forwarded on q_out, and appended to <run folder>/brand_feedback.jsonl.
"""
import json
import logging
import os
import socket
import time
import traceback
from pathlib import Path

import yaml
from improv.actor import Actor

from . import cpu_affinity
from .run_paths import get_logger, run_folder

logger = logging.getLogger(__name__)      # the log FILE is attached in BrandReceiver.setup(): SenderUDP imports this
                                           # module for link_settings(), which must not create an empty log per run


def link_settings():
    """brand_link block of config.yaml over the defaults (ports can also come from the environment)."""
    s = dict(stim_port=11116, feedback_port=11117)
    try:
        with open(Path(__file__).resolve().parent.parent / 'config' / 'config.yaml') as f:
            s.update((yaml.safe_load(f) or {}).get('brand_link') or {})
    except Exception:
        pass
    s['stim_port'] = int(os.getenv('SENDER_STIM_PORT', s['stim_port']))
    s['feedback_port'] = int(os.getenv('BRAND_FEEDBACK_PORT', s['feedback_port']))
    return s


class BrandReceiver(Actor):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def setup(self):
        get_logger(__name__, "brand_link.log")
        cpu_affinity.pin_actor(cpu_affinity.BACKGROUND, label="BrandReceiver")
        self.port = link_settings()['feedback_port']
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            self.sock.bind(('0.0.0.0', self.port))
        except OSError as e:
            logger.error(f"could not bind UDP port {self.port}: {e} (free it with: lsof -ti:{self.port} | xargs -r kill -9)")
            raise
        self.sock.setblocking(False)
        self.out_folder = run_folder()
        self.log = open(self.out_folder / 'brand_feedback.jsonl', 'a')
        self.counts = {}
        self.first_step = None
        self.warned = False
        logger.info(f"BrandReceiver listening on 0.0.0.0:{self.port}")

    def runStep(self):
        if self.first_step is None:
            self.first_step = time.time()
        got = False
        while True:                                   # drain: every stim event matters, none may be skipped
            try:
                data, addr = self.sock.recvfrom(65535)
            except BlockingIOError:
                break
            except Exception:
                logger.error(f"recv failed: {traceback.format_exc()}")
                break
            got = True
            t_rx = time.time()
            try:
                msg = json.loads(data.decode('utf-8'))
                if not isinstance(msg, dict):
                    raise ValueError('not a JSON object')
            except Exception as e:
                logger.warning(f"unparseable packet from {addr}: {e}: {data[:200]!r}")
                continue
            msg['t_rx'] = t_rx
            kind = msg.get('type', '?')
            self.counts[kind] = self.counts.get(kind, 0) + 1
            if kind != 'target':
                logger.info(f"{kind} {msg.get('id', '')}")
            try:
                self.q_out.put(msg)
            except Exception:
                logger.error(f"q_out.put failed: {traceback.format_exc()}")
            self.log.write(json.dumps(msg) + '\n')
        if got:
            self.log.flush()
        elif not self.warned and not self.counts and time.time() - self.first_step > 10:
            self.warned = True
            logger.warning(f"nothing from BRAND on port {self.port} after 10 s: is the BRAND-side node running "
                           f"and sending to this machine's address?")
        else:
            time.sleep(0.001)

    def stop(self):
        logger.info(f"BrandReceiver stopping; messages received: {self.counts}")
        for f in (self.sock, self.log):
            try:
                f.close()
            except Exception:
                pass
