import time
import serial
import logging
from improv.actor import Actor

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Create a file handler
log_file = "processor.log"
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
            element = self.q_in.get()
        except Exception as e:
            logger.error(f"Could not get element! {e}")
            return
        
        _ ,angle = element
        logger.info(f'recieved angle {angle}, and type {type(angle)}')
        

        # Create message packet
        valspack = self.pack_bytes(angle)

        # Send the message through UART
        self.ser.write(valspack)


    def stop(self):
        logger.info("Stopping Sender")
        # self.ser.close()  #Not sure if this is necessary
        logger.info("Sender stopped")
