import time
import serial
import struct

# Constants
# There are 5 flex/force sensors plus an extra frame sensor.
# Additionally, index 0 is reserved for an outdated proximity sensor.
NUM_SENSORS = 6  # This will create a list of NUM_SENSORS+1 = 7 sensor values.
PINS = ["A0", "A1", "A3"]  # Example pin names for analog sensors.
CHANS = [2, 4, 5]         # The analog sensor readings will be stored at these indices.
TIMER_INTERVAL = 0.001    # 1 ms interval

# Serial connection (for example, use COM port or /dev/ttyUSB0)
ser = serial.Serial("/dev/ttyUSB0", 115200)

# Timer flag
timer_ready = False

def one_ms_passed():
    global timer_ready
    timer_ready = True

def read_sensors():
    """
    Read sensor values and return a list of readings.
    The list has NUM_SENSORS+1 elements:
      - Index 0: Outdated proximity sensor (left at 0)
      - Indices given by CHANS: Updated from analog reads
      - The last element (index -1) will be used for the extra frame sensor.
    """
    # Initialize sensor readings to zero.
    vals = [0] * (NUM_SENSORS + 1)
    vals[0] = 0  # Outdated proximity sensor.

    # Update analog sensors.
    for i in range(len(PINS)):
        # Replace analog_read() with the actual sensor reading logic as needed.
        vals[CHANS[i]] = analog_read(PINS[i])
        time.sleep(0.0001)  # 100 microseconds delay
    
    # Add extra sensor value (frame sensor) at the end.
    # For now, we use a constant (123); later, replace this with your frame number.
    vals[-1] = 123
    
    return vals

def analog_read(pin):
    """
    Dummy function to simulate analog reading from a pin.
    Replace with actual analog read logic if needed.
    """
    return 255

def pack_bytes(adc_vals):
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
        temp_data |= adc_vals[i] << (i * 10)

    # Extract bytes from temp_data.
    for i in range(num_data_bytes):
        packed_vals[i + 2] = (temp_data >> (i * 8)) & 0xFF
    
    return packed_vals

def main_loop():
    global timer_ready
    
    while True:
        # Check timer flag.
        if timer_ready:
            # Read sensor values.
            sensor_vals = read_sensors()

            # Create the message packet.
            valspack = pack_bytes(sensor_vals)

            # Send the packet over UART.
            ser.write(valspack)
            print(f"Sent packet: {valspack.hex()}")

            # Reset the timer flag.
            timer_ready = False
        
        # Simulate a 1 ms timer interval.
        time.sleep(TIMER_INTERVAL)
        one_ms_passed()

if __name__ == "__main__":
    try:
        main_loop()
    except KeyboardInterrupt:
        # Close serial port when the script is interrupted.
        ser.close()
