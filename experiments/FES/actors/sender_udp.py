import time
import socket
import json
import numpy as np
import logging
from improv.actor import Actor

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
    
    Receives data from the input queue and sends it as UDP packets
    to a specified IP address and port.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def setup(self):
        logger.info("Beginning setup for SenderUDP")

        # UDP connection parameters
        self.UDP_IP_send = "127.0.0.1"  # Default to localhost, can be configured
        self.UDP_PORT_send = 11115  # Default port, can be configured
        
        # Create UDP socket
        self.sock_send = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        
        self.last_angle = 0
        self.last_angle2 = 0
        
        logger.info(f"UDP Sender configured to send to {self.UDP_IP_send}:{self.UDP_PORT_send}")
        logger.info("Completed setup for SenderUDP")

    def runStep(self):
        """Main execution step - get angles from processors and send as JSON over UDP."""
        got_data = False

        try:
            #Processor 0
            element = self.links["preds0_in"].get(timeout=0.0001)
            if len(element) == 4:
                _, angle, _, _ = element
            else:
                _, angle = element
            self.last_angle = angle
            got_data = True
        except Exception:
            pass

        try:
            #Processor 2
            element2 = self.links["preds2_in"].get(timeout=0.0001)
            if len(element2) == 4:
                _, angle2, _, _ = element2
            else:
                _, angle2 = element2
            self.last_angle2 = angle2
            got_data = True
        except Exception:
            pass

        # Drain generator q_in just in case
        try:
            self.q_in.get(timeout=0.0001)
        except Exception:
            pass

        if got_data:
            try:
                # Pack the angles as simple JSON dictionary
                data_dict = {"angle": float(self.last_angle), "angle2": float(self.last_angle2)}
                data_bytes = json.dumps(data_dict).encode('utf-8')
                
                # Send the data via UDP
                self.sock_send.sendto(data_bytes, (self.UDP_IP_send, self.UDP_PORT_send))
                logger.debug(f"Sent {len(data_bytes)} bytes via UDP: {data_dict}")
            except Exception as e:
                logger.error(f"Error sending UDP data: {e}")

    def stop(self):
        logger.info("Stopping SenderUDP")
        try:
            self.sock_send.close()
        except Exception as e:
            logger.error(f"Error closing socket: {e}")
        logger.info("SenderUDP stopped")
