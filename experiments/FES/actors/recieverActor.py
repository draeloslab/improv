import time
# import serial
import socket
import numpy as np
import logging
from pathlib import Path
from improv.actor import Actor

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Create a file handler
log_file = "reciever.log"
file_handler = logging.FileHandler(log_file)
file_handler.setLevel(logging.INFO)

# Create a formatter and set it for the handler
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)

# Add the handler to the logger
logger.addHandler(file_handler)

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
        
        # Create and bind socket
        self.sock_receive = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock_receive.bind((self.UDP_IP_receive, self.UDP_PORT_receive))
        # self.sock_receive.settimeout(0.1)  # Non-blocking with short timeout
        
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


        timestamp = time.strftime("%Y%m%d-%H%M")
        self.out_folder = Path(f"/home/chesteklab/predictions/{timestamp}")
        self.out_folder.mkdir(parents=True, exist_ok=True)
        
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
        try:

            # Receive UDP packet with timeout
            data = self.sock_receive.recv(1500)
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
            # No data received within timeout - this is normal
            logger.debug("Did not get packet")
            pass
        except Exception as e:
            logger.error(f"Error receiving/parsing UDP data: {e}")

    def stop(self):
        logger.info("Stopping UDPReceiver")
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
