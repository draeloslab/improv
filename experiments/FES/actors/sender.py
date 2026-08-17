import time
import numpy
import serial
import logging
from improv.actor import Actor
from pathlib import Path
import yaml
import numpy as np

from .run_paths import get_logger, run_folder

logger = get_logger(__name__, "sender.log")

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

        self.packet_n = 0

        self.last_angle = 0
        self.last_angle2 = 0

        self.top_angle_history = []
        self.side_angle_history = []

        # Percentiles for robust min/max (filters outliers)
        self.min_percentile = 1  # 1st percentile as "true min"
        self.max_percentile = 99  # 99th percentile as "true max"

        # --- Timing logs ---
        # Wall-clock timestamp for every UART send (time.time())
        self.send_timestamps = []
        # Per-packet: which frame_num from cam0 and cam2 was used for the angle
        self.sent_frame_nums_cam0 = []
        self.sent_frame_nums_cam2 = []
        # Actual angle values sent each packet
        self.sent_angles_cam0 = []
        self.sent_angles_cam2 = []
        # Whether each camera had a fresh prediction this step (vs stale)
        self.fresh_cam0 = []
        self.fresh_cam2 = []
        # True end-to-end latency: generator_timestamp → sender UART write
        self.true_e2e_cam0 = []
        self.true_e2e_cam0_frame_nums = []
        self.true_e2e_cam0_timestamps = []
        self.true_e2e_cam2 = []
        self.true_e2e_cam2_frame_nums = []
        self.true_e2e_cam2_timestamps = []
        # Sender's own step latency
        self.step_latencies = []
        # Track latest frame nums for staleness detection
        self.last_frame_num_cam0 = -1
        self.last_frame_num_cam2 = -1

        # Load the configuration file
        source_folder = Path(__file__).resolve().parent.parent
        with open(f'{source_folder}/config.yaml', 'r') as file:
            config = yaml.safe_load(file)

        self.resize = config['resize']

        self.out_folder = run_folder()
        logger.info(f"Output folder set to {self.out_folder}")


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
        step_start = time.perf_counter()
        got_fresh_cam0 = False
        got_fresh_cam2 = False
        camera_start_cam0 = None
        camera_start_cam2 = None
        frame_num_cam0 = getattr(self, 'last_frame_num_cam0', -1)
        frame_num_cam2 = getattr(self, 'last_frame_num_cam2', -1)

        #Grab angle from processor
        try:
            #Processor 0
            element = self.links["preds0_in"].get(timeout=0.0001)
            # Support [pred, angle], [pred, angle, camera_start, frame_num], and
            # [pred, angle, camera_start, frame_num, camera_num] formats.
            if len(element) >= 4:
                _, angle, camera_start_cam0, frame_num_cam0 = element[:4]
            else:
                _, angle = element
                camera_start_cam0 = None
                frame_num_cam0 = -1
            # self.last_angle = angle
            self.last_angle = self.normalize_to_range(angle, 700, 330)  # Convert from 100-180 to 0-1023
            # self.last_angle = (angle-100) * 10  # Store the angle for reuse # Roughly convert from 120 -180 to 200 to 800
            self.last_frame_num_cam0 = frame_num_cam0
            got_fresh_cam0 = True
        except Exception as e:
            pass
            # # No new data available, use previous angle if it exists
            # if not hasattr(self, 'last_angle'):
            #     logger.debug(f"No element available yet and no previous value: {e}")
            #     return  # No data to send
            # angle = self.last_angle


        try:
            #Processor 2
            element2 = self.links["preds2_in"].get(timeout=0.0001)
            # Support [pred, angle], [pred, angle, camera_start, frame_num], and
            # [pred, angle, camera_start, frame_num, camera_num] formats.
            if len(element2) >= 4:
                _, angle2, camera_start_cam2, frame_num_cam2 = element2[:4]
            else:
                _, angle2 = element2
                camera_start_cam2 = None
                frame_num_cam2 = -1
            # self.last_angle2 = angle2
            self.last_angle2 = self.normalize_to_range(angle2, 1075, 400)  # Convert from 800-1200 to 0-1023
            # self.last_angle2 = (angle2-800) * 2  # roughly convert from 800-1200 to 0-800
            self.last_frame_num_cam2 = frame_num_cam2
            got_fresh_cam2 = True
        except Exception as e:
            pass
            # logger.info(f'Error in sender 2: {repr(e)}')
            # # No new data available, use previous angle if it exists
            # if not hasattr(self, 'last_angle2'):
            #     logger.debug(f"No element available yet and no previous value: {e}")
            #     return  # No data to send
            # angle2 = self.last_angle2

        # --- Drain generator q_in (not needed for timing anymore since
        #     processor now passes camera_start through) ---
        try:
            self.q_in.get(timeout=0.0001)
        except Exception as e:
            pass


        # Create message packet
        if self.packet_n % 10000 == 0:  # Log every 10000 packets
            logger.info(f'Sending angles: {self.last_angle} and {self.last_angle2}')

        # Collect angles for final min/max computation at stop()
        self.top_angle_history.append(self.last_angle)
        self.side_angle_history.append(self.last_angle2)


        valspack = self.pack_bytes([0, 0, self.last_angle, 0, self.last_angle2, 0])

        # Send the message through UART
        self.ser.write(valspack)
        send_time = time.time()
        self.packet_n += 1

        # --- Log timing data ---
        self.send_timestamps.append(send_time)
        self.sent_frame_nums_cam0.append(frame_num_cam0)
        self.sent_frame_nums_cam2.append(frame_num_cam2)
        self.sent_angles_cam0.append(self.last_angle)
        self.sent_angles_cam2.append(self.last_angle2)
        self.fresh_cam0.append(got_fresh_cam0)
        self.fresh_cam2.append(got_fresh_cam2)

        # True end-to-end: only when we got a fresh prediction with a camera_start
        if got_fresh_cam0 and camera_start_cam0 is not None:
            self.true_e2e_cam0.append(send_time - camera_start_cam0)
            self.true_e2e_cam0_frame_nums.append(frame_num_cam0)
            self.true_e2e_cam0_timestamps.append(send_time)

        if got_fresh_cam2 and camera_start_cam2 is not None:
            self.true_e2e_cam2.append(send_time - camera_start_cam2)
            self.true_e2e_cam2_frame_nums.append(frame_num_cam2)
            self.true_e2e_cam2_timestamps.append(send_time)

        self.step_latencies.append(time.perf_counter() - step_start)







    def stop(self):
        logger.info("Stopping Sender")

        # Compute robust min/max once over the entire run
        if len(self.top_angle_history) > 0:
            top_min = np.percentile(self.top_angle_history, self.min_percentile)
            top_max = np.percentile(self.top_angle_history, self.max_percentile)
            logger.info(f"Top angle — robust min: {top_min}, robust max: {top_max} "
                        f"(from {len(self.top_angle_history)} samples)")
        else:
            logger.info("No top angle data collected")

        if len(self.side_angle_history) > 0:
            side_min = np.percentile(self.side_angle_history, self.min_percentile)
            side_max = np.percentile(self.side_angle_history, self.max_percentile)
            logger.info(f"Side angle — robust min: {side_min}, robust max: {side_max} "
                        f"(from {len(self.side_angle_history)} samples)")
        else:
            logger.info("No side angle data collected")

        if hasattr(self, 'ser') and self.ser.is_open:
            self.ser.close()

        # --- Save all timing/data logs ---
        np.save(self.out_folder / "sender_timestamps.npy", self.send_timestamps)
        np.save(self.out_folder / "sender_step_latencies.npy", self.step_latencies)
        np.save(self.out_folder / "sender_sent_angles_cam0.npy", self.sent_angles_cam0)
        np.save(self.out_folder / "sender_sent_angles_cam2.npy", self.sent_angles_cam2)
        np.save(self.out_folder / "sender_sent_frame_nums_cam0.npy", self.sent_frame_nums_cam0)
        np.save(self.out_folder / "sender_sent_frame_nums_cam2.npy", self.sent_frame_nums_cam2)
        np.save(self.out_folder / "sender_fresh_cam0.npy", self.fresh_cam0)
        np.save(self.out_folder / "sender_fresh_cam2.npy", self.fresh_cam2)

        # True end-to-end latencies (one entry per fresh prediction)
        np.save(self.out_folder / "true_e2e_cam0.npy", self.true_e2e_cam0)
        np.save(self.out_folder / "true_e2e_cam0_frame_nums.npy", self.true_e2e_cam0_frame_nums)
        np.save(self.out_folder / "true_e2e_cam0_timestamps.npy", self.true_e2e_cam0_timestamps)
        np.save(self.out_folder / "true_e2e_cam2.npy", self.true_e2e_cam2)
        np.save(self.out_folder / "true_e2e_cam2_frame_nums.npy", self.true_e2e_cam2_frame_nums)
        np.save(self.out_folder / "true_e2e_cam2_timestamps.npy", self.true_e2e_cam2_timestamps)

        # Legacy files for backward compatibility
        np.save(self.out_folder / "endtoendLatencies.npy", self.true_e2e_cam0)
        np.save(self.out_folder / "senderStartTimes.npy", self.send_timestamps)

        if len(self.true_e2e_cam0) > 0:
            logger.info(f"True E2E cam0: mean={np.mean(self.true_e2e_cam0)*1000:.1f}ms, "
                        f"median={np.median(self.true_e2e_cam0)*1000:.1f}ms, "
                        f"max={np.max(self.true_e2e_cam0)*1000:.1f}ms "
                        f"(from {len(self.true_e2e_cam0)} frames)")
        if len(self.true_e2e_cam2) > 0:
            logger.info(f"True E2E cam2: mean={np.mean(self.true_e2e_cam2)*1000:.1f}ms, "
                        f"median={np.median(self.true_e2e_cam2)*1000:.1f}ms, "
                        f"max={np.max(self.true_e2e_cam2)*1000:.1f}ms "
                        f"(from {len(self.true_e2e_cam2)} frames)")
        logger.info(f"Total UART packets sent: {self.packet_n}")
        logger.info(f"Fresh cam0 predictions: {sum(self.fresh_cam0)} / {len(self.fresh_cam0)}")
        logger.info(f"Fresh cam2 predictions: {sum(self.fresh_cam2)} / {len(self.fresh_cam2)}")
        
        logger.info("Sender stopped")
