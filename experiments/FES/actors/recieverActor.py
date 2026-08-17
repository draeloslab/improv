import time
# import serial
import socket
import numpy as np
import logging
from pathlib import Path

import yaml
from improv.actor import Actor

from .run_paths import get_logger, run_folder
from . import xpc_link

logger = get_logger(__name__, "reciever.log", level=logging.DEBUG)

#: How long after the first runStep to wait before complaining that nothing has
#: arrived. Long enough to cover actor startup skew, short enough that you find
#: out at the start of a session rather than at the end of it.
SILENCE_WARNING_SECONDS = 5.0

class Receiver(Actor):
    """Actor to receive UDP packets and parse xPC data.
    
    Receives UDP packets containing neural data and other variables from xPC,
    parses them, and passes the data through the actor pipeline.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def setup(self):
        logger.info("Beginning setup for UDPReceiver")

        # UDP connection parameters
        self.UDP_IP_receive = "0.0.0.0"  # Listen on all interfaces
        self.UDP_PORT_receive = 11114

        # Check the spoofed xPC link before binding. A wrong or missing link is
        # the difference between a session's worth of data and an empty
        # fpos_data.npy, and it is invisible at runtime -- so say so loudly now.
        xpc_link.check(logger)

        # Create and bind socket
        self.sock_receive = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # A dirty quit can leave the previous run's socket lingering; without
        # this the next run dies on bind with EADDRINUSE.
        self.sock_receive.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            self.sock_receive.bind((self.UDP_IP_receive, self.UDP_PORT_receive))
        except OSError as e:
            logger.error(
                "Could not bind UDP port %d: %s. Something else is holding it "
                "-- most likely a standalone actors/reciever.py, or an improv "
                "run that did not exit cleanly. Free it with: "
                "lsof -ti:%d | xargs -r kill -9",
                self.UDP_PORT_receive, e, self.UDP_PORT_receive,
            )
            raise
        self.sock_receive.settimeout(0.1)  # Non-blocking with short timeout

        # Silence tracking -- see runStep.
        self.packets_received = 0
        self.first_step_time = None
        self.warned_silent = False

        # Data parsing parameters
        self.data_lengths = [            # should add up to 832 (July 2022)
            4,      # eTime
            1,      # feat
            2,      # dsize
            768,    # neural_data   (96 * 8 bytes)
            20,     # fpos          (5 * 4 bytes)
            20,     # target_pos    (5 * 4 bytes)
            2,      # trial_count   (uint16 - 2 bytes)
            1,      # switch1
            1,      # switch2
            1,      # switch3
            4,      # val1          (single - 4 bytes)
            4,      # val2          (single - 4 bytes)
            2,      # msCount       (uint16 - 2 bytes)
            1,      # xpcBinSize    (uint16, but likely only 1 byte after casting)
            1       # enable
        ]

        logger.info(f"UDP Receiver listening on port {self.UDP_PORT_receive}")
        
        # Initialize data storage lists (similar to processor.py)
        self.timestamps = []
        self.fpos_data = []
        
         # load the configuration file
        source_folder = Path(__file__).resolve().parent.parent

        with open(f'{source_folder}/config.yaml', 'r') as file:
            config = yaml.safe_load(file)
    

        self.out_folder = run_folder()
        
        logger.info("Completed setup for UDPReceiver")

    def parse_any_data(self, packet_in, data_length_list, start=0):
        """Helper function to parse packet data based on length specifications."""
        idx = start
        outputs = []
        for data_len in data_length_list:
            outputs.append(packet_in[idx:idx + data_len])
            idx += data_len
        return outputs

    def parse_packet(self, packet):
        """
        Takes in the UDP packet data and parses into correct variables
        :param packet: input of the udp packet received from xPC
        :return: parsed information from packet
        """
        packet_np = np.array([packet]).view(np.uint8)
        
        eTime, feat, dsize, neural_data, fpos, target_pos, trial_count, switch1, switch2, switch3, val1, val2, msCount, \
            xpcBinSize, enable = self.parse_any_data(packet_np, self.data_lengths)

        # Convert to appropriate data types
        msCount = msCount.view(np.ushort)  # this is xPC's iteration counter
        fpos = fpos.view(np.single)
        target_pos = target_pos.view(np.single)
        trial_count = trial_count.view(np.ushort)
        switches = [switch1, switch2, switch3]
        vals = [val1.view(np.single), val2.view(np.single)]
        xpcBinSize = xpcBinSize[0]
        
        # Create dictionary for easy access
        xpc_dict = {
            'eTime': eTime, 
            'feat': feat, 
            'dsize': dsize, 
            'neural_data': neural_data, 
            'fpos': fpos,
            'msCount': msCount, 
            'xpcBinSize': xpcBinSize, 
            'enable': enable, 
            'trial_count': trial_count,
            'target_pos': target_pos, 
            'xpc_switches': switches, 
            'xpc_vals': vals
        }

        return eTime, feat, dsize, neural_data, fpos, msCount, xpcBinSize, enable, \
               target_pos, trial_count, switches, vals, xpc_dict

    def runStep(self):
        """Main execution step - receive and parse UDP packet."""
        if self.first_step_time is None:
            self.first_step_time = time.time()

        try:

            # Receive UDP packet with timeout
            data = self.sock_receive.recv(1500)
            if self.packets_received == 0:
                logger.info("First UDP packet received from xPC")
            self.packets_received += 1
            logger.debug("Received UDP packet")

            # Parse the packet
            eTime, feat, dsize, neural_data, fpos, msCount, xpcBinSize, enable, \
                target_pos, trial_count, xpc_switches, xpc_vals, xpc_dict = self.parse_packet(data)
            
            # Store timestamp and fPos data (similar to processor.py style)
            current_time = time.time()
            self.timestamps.append(current_time)
            self.fpos_data.append(xpc_dict.copy())  # Use copy() to ensure we store the data properly #TODO need to change to xpc_dict.copy()
            
            # Log key data
            logger.debug(f"Parsed packet - fpos: {fpos}, msCount: {msCount}, timestamp: {current_time}")
            
            
        except socket.timeout:
            # No data received within timeout - this is normal per-step, but a
            # sustained silence is not, and the per-step DEBUG line is far too
            # noisy to notice it in. Say it once, clearly.
            if (not self.warned_silent
                    and self.packets_received == 0
                    and time.time() - self.first_step_time > SILENCE_WARNING_SECONDS):
                self.warned_silent = True
                logger.warning(
                    "No xPC packets on port %d after %.0f s of running. Check "
                    "that the run is started on Morpheous, that the red Vision "
                    "cable is in xPC, and that the link is up (%s).",
                    self.UDP_PORT_receive, SILENCE_WARNING_SECONDS,
                    xpc_link.FIX_COMMAND,
                )
                xpc_link.check(logger)
        except Exception as e:
            logger.error(f"Error receiving/parsing UDP data: {e}")

    def stop(self):
        logger.info("Stopping UDPReceiver")
        if self.packets_received == 0:
            logger.error(
                "Received 0 xPC packets this run -- fpos_data.npy will be "
                "empty. Fix the link with: %s", xpc_link.FIX_COMMAND,
            )
        else:
            logger.info("Received %d xPC packets this run", self.packets_received)
        try:
            self.sock_receive.close()
        except Exception as e:
            logger.error(f"Error closing socket: {e}")
        
        # Save collected data to numpy files (similar to processor.py style)
        try:
            np.save(self.out_folder / "Reciever_timestamps.npy", self.timestamps)
            np.save(self.out_folder / "fpos_data.npy", self.fpos_data)
            logger.info(f"Timestamps and fPos data saved to {self.out_folder}")
        except Exception as e:
            logger.error(f"Error saving data: {e}")
            
        logger.info("UDPReceiver stopped")
