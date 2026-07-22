import time
import os
import ipaddress
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
        self.UDP_IP_send = os.getenv("SENDER_UDP_IP", "192.168.137.201")
        self.UDP_PORT_send = int(os.getenv("SENDER_UDP_PORT", "5000"))

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
        
        self.last_angle = 0
        self.last_angle2 = 10
        
        logger.info(f"UDP Sender configured to send to {resolved_ip}:{self.UDP_PORT_send}")
        logger.info("Completed setup for SenderUDP")

    def runStep(self):
        """Main execution step - get angles from processors and send as JSON over UDP."""
        got_data = True

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
                print(f"Sent {len(data_bytes)} bytes via UDP to {self.UDP_IP_send}:{self.UDP_PORT_send}: {data_dict}")
                logger.info(f"Sent {len(data_bytes)} bytes via UDP to {self.UDP_IP_send}:{self.UDP_PORT_send}: {data_dict}")
            except Exception as e:
                logger.error(f"Error sending UDP data: {e}")
                print(f"Error sending UDP data: {e}")

    def stop(self):
        logger.info("Stopping SenderUDP")
        try:
            self.sock_send.close()
        except Exception as e:
            logger.error(f"Error closing socket: {e}")
        logger.info("SenderUDP stopped")

if __name__ == "__main__":
    # For testing purposes, you can instantiate and run the actor here.
    sender = SenderUDP('SenderUDP')
    sender.setup()
    try:
        while True:
            sender.runStep()
            time.sleep(0.01)  # Adjust sleep time as needed
    except KeyboardInterrupt:
        sender.stop()