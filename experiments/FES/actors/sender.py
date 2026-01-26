import time
import serial
import logging
from improv.actor import Actor
from pathlib import Path
import yaml

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

        self.top_min = 120
        self.side_min = 120

        self.last_angle = 0
        self.last_angle2 = 0

        # Load the configuration file
        source_folder = Path(__file__).resolve().parent.parent
        with open(f'{source_folder}/config.yaml', 'r') as file:
            config = yaml.safe_load(file)

        self.resize = config['resize']


    def pack_bytes(self, adc_vals):
        """
        Pack each sensor’s 10-bit value into a continuous stream of bits.
        The packet begins with a 2-byte header ('-' and '>'),
        followed by enough bytes to cover all sensor bits.
        """
        total_sensors = len(adc_vals)  # This is NUM_SENSORS+1 (e.g. 7)
        # Calculate how many bytes are needed to pack all 10-bit values.
        num_data_bytes = (total_sensors * 10 + 7) // 8  # Ceiling division

        # Create bytearray: 2 bytes for header + data bytes.
        packed_vals = bytearray(2 + num_data_bytes)
        packed_vals[0] = ord('-')  # Header byte 1
        packed_vals[1] = ord('>')  # Header byte 2

        # Combine each sensor's 10-bit value into one large integer.
        temp_data = 0
        for i in range(total_sensors):
            temp_data |= int(adc_vals[i]) << (i * 10)

        # Extract bytes from temp_data.
        for i in range(num_data_bytes):
            packed_vals[i + 2] = (temp_data >> (i * 8)) & 0xFF

        return packed_vals
    
    def normalize_to_range(self,value, min_val, max_val):
        """Convert value from [min_val, max_val] to [0, 1023]"""
        normalized = ((value - min_val) / (max_val - min_val)) * 1023
        return max(0, min(1023, int(normalized)))  # Clamp to valid range

    def runStep(self):

        #Grab angle from processor
        try:
            #Processor 0
            element = self.links["preds0_in"].get(timeout=0.0001)
            _ ,angle = element
            angle = self.normalize_to_range(angle, 140,300)
            self.last_angle = angle  # Store the angle for reuse
            # logger.info(f'recieved angle {angle}, from camera 0')
        except Exception as e:
            pass
            # logger.info(f'Error in sender 0: {repr(e)}')
            # # No new data available, use previous angle if it exists
            # if not hasattr(self, 'last_angle'):
            #     logger.debug(f"No element available yet and no previous value: {e}")
            #     return  # No data to send
            # angle = self.last_angle


        try:
            #Processor 2
            element2 = self.links["preds2_in"].get(timeout=0.0001)
            _ ,angle2 = element2
            angle2 = angle2/self.resize
            angle2 = self.normalize_to_range(angle2, 410,580)
            self.last_angle2 = angle2  # Store the angle for reuse
            # logger.info(f'recieved angle {angle2}, from camera 2')
        except Exception as e:
            pass
            # logger.info(f'Error in sender 2: {repr(e)}')
            # # No new data available, use previous angle if it exists
            # if not hasattr(self, 'last_angle2'):
            #     logger.debug(f"No element available yet and no previous value: {e}")
            #     return  # No data to send
            # angle2 = self.last_angle2

        # Create message packet
        # logger.info(f'Sending angles: {angle}, {angle2}')
        valspack = self.pack_bytes([0,0,self.last_angle,0,self.last_angle2,0,0])
        # logger.info(f'Sending packed values: {valspack.hex()}')


        # Send the message through UART
        self.ser.write(valspack)
        # logger.info(f"Sent packet: {valspack.hex()}")



    def stop(self):
        logger.info("Stopping Sender")
        # self.ser.close()  #Not sure if this is necessary
        logger.info("Sender stopped")
