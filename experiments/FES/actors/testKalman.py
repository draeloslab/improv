import time
from kalmanfilter import KalmanFilterPredictor
import numpy as np
import matplotlib.pyplot as plt

def calculateAngle(predictions):

        p2, p3, p4 = predictions[:,0, :2], predictions[:,2, :2], predictions[:,3, :2]
        #  DIP=0, PIP=1, MCP=2, Wrist=3, currently getting angle at MCP
        # Define vectors from point 3 to points 2 and 4
        v3_to_2 = p2 - p3
        v3_to_4 = p4 - p3

        # Calculate dot product and determinant for each frame
        dot_product = np.sum(v3_to_2 * v3_to_4, axis=1)
        determinant = v3_to_2[:, 0] * v3_to_4[:, 1] - v3_to_2[:, 1] * v3_to_4[:, 0]

        # Calculate angle in degrees at point 3
        angle = np.degrees(np.arctan2(determinant, dot_product)) % 360
        return angle

kalman_filter = KalmanFilterPredictor(
                adapt=False,
                forward=0.002,
                fps=30,  
                nderiv=2,
                priors=[1, 1],
                initial_var=10,    
                process_var=1,     
                dlc_var=10,        
                lik_thresh=0.2     
            )
folder = '/home/chesteklab/predictions/20260226/20260226-1220'
predictions = np.load(f"{folder}/predictions_cam0.npy")

# Initialize array to store smoothed predictions with same shape as input
smoothed_predictions = np.zeros_like(predictions)
latencies = []
# Process each frame through the Kalman filter
for i, prediction in enumerate(predictions):
    start_time = time.perf_counter()
    smoothed = kalman_filter.process(prediction)
    smoothed_predictions[i] = smoothed
    end_time = time.perf_counter()
    latencies.append(end_time - start_time)

# Plot the first keypoint (index 0) x and y coordinates for all 800 time points
plt.figure(figsize=(12, 6))
keypoint = 3
plt.subplot(2, 1, 1)
plt.plot(predictions[:, keypoint, 0], label='Raw Keypoint 0 X', alpha=0.7)
plt.plot(smoothed_predictions[:, keypoint, 0], label='Smoothed Keypoint 0 X', linewidth=2)
plt.legend()
plt.ylabel('X Position')
plt.title('Kalman Filter Smoothing - First Keypoint')
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(predictions[:, keypoint, 1], label='Raw Keypoint 0 Y', alpha=0.7)
plt.plot(smoothed_predictions[:, keypoint, 1], label='Smoothed Keypoint 0 Y', linewidth=2)
plt.legend()
plt.xlabel('Frame')
plt.ylabel('Y Position')
plt.grid(True)

plt.tight_layout()
plt.show()

angles = calculateAngle(predictions)
smoothed_angles = calculateAngle(smoothed_predictions)
plt.figure(figsize=(10, 5))
plt.plot(angles, label='Raw Angle at MCP', alpha=0.7)
plt.plot(smoothed_angles, label='Smoothed Angle at MCP', linewidth=2)
plt.legend()
plt.xlabel('Frame')
plt.ylabel('Angle (degrees)')
plt.title('Angle at MCP Joint - Raw vs Smoothed Predictions')
plt.grid(True)
plt.show()

print(f"Average latency per frame: {np.mean(latencies)*1000:.2f} ms")
plt.plot(latencies)
plt.xlabel('Frame')
plt.ylabel('Latency (seconds)')
plt.title('Latency per Frame for Kalman Filter Processing')
plt.grid(True)
plt.show()


