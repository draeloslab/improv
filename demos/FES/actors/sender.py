import time
import serial
import struct

# Constants
NUM_SENSORS = 5  # Number of sensors for flex/force: typically 5
PINS = ["A0", "A1", "A3"]  # Example pin names for sensors
CHANS = [2, 4, 5]  # Sensor channels
TIMER_INTERVAL = 0.001  # 1 ms interval

# Serial connection (for example, use COM port or /dev/ttyUSB0)
ser = serial.Serial("COM6", 115200)

# Timer flag
timer_ready = False


def one_ms_passed():
    global timer_ready
    timer_ready = True


def read_sensors():
    """
    Read sensor values and return a list of readings.
    """
    # Initialize return values with zeros
    vals = [0] * (NUM_SENSORS + 1)
    vals[0] = 0  # First value corresponds to the outdated proximity sensor

    for i in range(len(PINS)):
        # Analog reading would be replaced by appropriate sensor reading
        vals[CHANS[i]] = analog_read(PINS[i])
        time.sleep(0.0001)  # Delay for 100 microseconds
    
    return vals


def analog_read(pin):
    """
    Dummy function to simulate analog reading from a pin.
    """
    return 255  # Replace with actual analog read logic if needed


def pack_bytes(adc_vals):
    """
    Pack 10-bit sensor values into an array of bytes.
    """
    # Create header bytes
    packed_vals = bytearray(10)  # Create a byte array of packet data bytes + header
    packed_vals[0] = ord('-')  # Header byte 1
    packed_vals[1] = ord('>')  # Header byte 2

    # Combine 10-bit values into one 64-bit integer
    temp_data = 0
    for i in range(NUM_SENSORS + 1):
        temp_data |= adc_vals[i] << (i * 10)

    # Extract bytes from temp_data and store in packed_vals
    for i in range(8):  # Remaining 8 bytes
        packed_vals[i + 2] = (temp_data >> (i * 8)) & 0xFF
    
    return packed_vals


def main_loop():
    global timer_ready
    
    while True:
        # Run the loop continuously
        if timer_ready:
            # Read from the sensors
            sensor_vals = read_sensors()

            # Create message packet
            valspack = pack_bytes(sensor_vals)

            # Send the message through UART
            ser.write(valspack)

            # Reset the timer flag
            timer_ready = False
        
        # Simulate 1ms timer interval
        time.sleep(TIMER_INTERVAL)
        one_ms_passed()


if __name__ == "__main__":
    # Start main loop
    try:
        main_loop()
    except KeyboardInterrupt:
        # Close serial port when the script is interrupted
        ser.close()