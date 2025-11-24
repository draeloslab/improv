import time
import serial
import logging
from improv.actor import Actor

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Create a file handler
log_file = "sender.log"
file_handler = logging.FileHandler(log_file)
file_handler.setLevel(logging.INFO)

# Create a formatter and set it for the handler
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)

# Add the handler to the logger
logger.addHandler(file_handler)

class Sender(Actor):
    """Sample actor to generate data to pass into a sample processor.

    Intended for use along with sample_processor.py.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def setup(self):
        logger.info("Beginning setup for Sender")

        # Constants
        self.NUM_SENSORS = 5  # Number of sensors for flex/force: typically 5
        self.CHANS = [2]  # Sensor channels
        self.TIMER_INTERVAL = 0.001  # 1 ms interval

        # Serial connection (for example, use COM port or /dev/ttyUSB0)
        self.ser = serial.Serial("/dev/ttyUSB0", 115200)

        logger.info("Completed setup for Sender")


    def pack_bytes(self, adc_vals):
        """
        Pack 10-bit sensor values into an array of bytes.
        """
        # Create header bytes
        packed_vals = bytearray(10)  # Create a byte array of packet data bytes + header
        packed_vals[0] = ord('-')  # Header byte 1
        packed_vals[1] = ord('>')  # Header byte 2

        # Combine 10-bit values into one 64-bit integer
        temp_data = 0
        for i in range(self.NUM_SENSORS + 1):
            temp_data |= int(adc_vals) << (i * 10)

        # Extract bytes from temp_data and store in packed_vals
        for i in range(8):  # Remaining 8 bytes
            packed_vals[i + 2] = (temp_data >> (i * 8)) & 0xFF
        
        return packed_vals


    def runStep(self):

        #Grab angle from processor
        try:
            element = self.q_in.get(timeout=0.001)  # Non-blocking get with small timeout
            _ ,angle = element
            self.last_angle = angle  # Store the angle for reuse
            # logger.info(f'recieved angle {angle}, and type {type(angle)}')
        except Exception as e:
            # No new data available, use previous angle if it exists
            if not hasattr(self, 'last_angle'):
                logger.debug(f"No element available yet and no previous value: {e}")
                return  # No data to send
            angle = self.last_angle
        

        # Create message packet
        valspack = self.pack_bytes(angle)

        # Send the message through UART
        self.ser.write(valspack)
        # logger.info(f"Sent packet: {valspack.hex()}")



    def stop(self):
        logger.info("Stopping Sender")
        # self.ser.close()  #Not sure if this is necessary
        logger.info("Sender stopped")
