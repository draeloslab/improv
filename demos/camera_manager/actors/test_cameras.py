import yaml
from multiprocessing import Process, RawArray
import threading
import zmq
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
import cv2

from pathlib import Path
from TIS import *

# Needed packages:
# pyhton-gst-1.0
# python-opencv
# tiscamera (+ pip install pycairo PyGObject)

# @profile
def read_frames(camera_name, socket, shared_frame, video_out):
    global stop_program

    time_start = time.perf_counter()
    frame_count = 0
    total_delay = 0
    max_delay = 0

    while not stop_program:
        # process item        
        frame_time = socket.recv_pyobj()

        frame = np.frombuffer(shared_frame).reshape(1080, 1920, 4).astype(np.uint8)

        # video_out.write(frame[:,:,:3])
        
        received_time = time.perf_counter() 

        frame_count += 1
        total_delay += received_time - frame_time

        if received_time - frame_time > max_delay:
            max_delay = received_time - frame_time

        if frame_count % 240 == 0:
            time_end = time.perf_counter()
            total_time = time_end - time_start

            print(f"[Camera {camera_name}] - General FPS: {round(frame_count / total_time,2)} - avg delay: {total_delay/frame_count:.3f} - max delay: {max_delay:.3f}")

            # frame_size = round(frame[:,:,:3].nbytes/(1024**2),4)
            # print(f"Frame shape {frame.shape} - size on memory: {frame_size}MB")

            frame_count = 0
            total_time = 0
            total_delay = 0
            max_delay = 0

            time_start = time.perf_counter()

# load the configuration file
source_folder = Path(__file__).resolve().parent.parent

with open(f'{source_folder}/config/camera_config.yaml', 'r') as file:
    config = yaml.safe_load(file)

camera_params = config['camera_params']

frame_w = camera_params['resolution']['width'] # frame width
frame_h = camera_params['resolution']['height'] # frame height
fps = camera_params['fps'] # FPS
num_devices = len(config['active_cameras'])

stop_program = False

# video saving initializing (this will be another actor)
video_codec = cv2.VideoWriter_fourcc(*'XVID')

# zmq intialization
zmq_frames_ip = config['zmq_config']['ip']
zmq_context = zmq.Context()

# camera setup
camera_sockets = []
camera_interfaces = []
camera_names = []
shared_frames = []
videos_out = []

for i,c in enumerate(config['active_cameras']):
    camera = c['camera']

    print(f'Initializing camera {camera["name"]}')

    # initializing the video output
    out_file_name = f'output_{camera["name"]}.avi'
    video_out = cv2.VideoWriter(f'/home/matteo/camera_video/{out_file_name}', video_codec, 60, (frame_w, frame_h), True)
    videos_out.append(video_out)

    # initializing the camera frame and socket for communication
    shared_frame = RawArray('d', frame_w * frame_h * 4)
    zmq_port = camera['zmq_port']

    print(f'\tinit camera zmq socket on {zmq_frames_ip}:{zmq_port}')

    socket = zmq_context.socket(zmq.PULL)
    socket.connect(f"tcp://{zmq_frames_ip}:{zmq_port}")

    # initializing each camera TIS interface
    print(f'\topening device: {camera}')

    tis_camera = TIS(zmq_port, shared_frame)
    tis_camera.open_device(camera['serial_id'], frame_w, frame_h, fps, SinkFormats.BGR, showvideo=False)

    camera_interfaces.append(tis_camera)
    camera_names.append(camera['name'])
    
    print(f'\tdevice {camera} opened')

    # starting the camera pipeline
    tis_camera.start_pipeline()

    thread = threading.Thread(target=read_frames, args=(camera_names[i], socket, shared_frame, videos_out[i]))
    thread.start()

    print(f'\tdevice {camera} pipeline started')

input("Press Enter to start the program...\n")

for tis_camera in camera_interfaces:
    tis_camera.start_sharing()

time_start_recording = time.perf_counter()

input("Press Enter to stop the program...\n")
time_end_recording = time.perf_counter()

print(f"Recording time: {time_end_recording - time_start_recording:.0f} s")

## stop function code
stop_program = True

for tis_camera in camera_interfaces:
    tis_camera.stop_pipeline()

stop_time = time.perf_counter()

print('Stop time:', stop_time)

logger.info("Camera reading setup completed")