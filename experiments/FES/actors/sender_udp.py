import time
import socket
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
        
        logger.info(f"UDP Sender configured to send to {self.UDP_IP_send}:{self.UDP_PORT_send}")
        logger.info("Completed setup for SenderUDP")

    def runStep(self):
        """Main execution step - get data from queue and send as UDP packet."""
        try:
            # Get data from input queue
            element = self.q_in.get(timeout=0.1)
            logger.debug(f"Received element from queue: {element}")
            
            # Extract angle or data from the element
            # Assuming element format is (timestamp, data, angle) based on sender.py
            _, _, data = element
            logger.info(f'Received data: {data}, type: {type(data)}')
            
            # Convert data to bytes if it's not already
            if isinstance(data, (int, float)):
                # Pack as numpy array then convert to bytes
                data_bytes = np.array([data], dtype=np.float32).tobytes()
            elif isinstance(data, np.ndarray):
                data_bytes = data.tobytes()
            elif isinstance(data, bytes):
                data_bytes = data
            else:
                # Try to convert to string then encode
                data_bytes = str(data).encode('utf-8')
            
            # Send the data via UDP
            self.sock_send.sendto(data_bytes, (self.UDP_IP_send, self.UDP_PORT_send))
            logger.debug(f"Sent {len(data_bytes)} bytes via UDP")
            
        except Exception as e:
            logger.error(f"Error in runStep: {e}")

    def stop(self):
        logger.info("Stopping SenderUDP")
        try:
            self.sock_send.close()
        except Exception as e:
            logger.error(f"Error closing socket: {e}")
        logger.info("SenderUDP stopped")
