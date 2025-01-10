import subprocess
import struct
import numpy as np
import cv2
import threading
import os
from pathlib import Path
from queue import Queue
from skvideo.io import FFmpegWriter

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Create a file handler
log_file = "camera_video_converter.log"
file_handler = logging.FileHandler(log_file)
file_handler.setLevel(logging.INFO)

# Create a formatter and set it for the handler
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)

# Add the handler to the logger
logger.addHandler(file_handler)

class VideoConverter:
    """
    A class to handle video conversion from buffer files to video using FFmpeg.
    """
    def __init__(self, compression_quality, video_params, output_video, out_folder_buffer,
                log_progress=False, msg_out=None, stop_program_event=threading.Event()):
        """
        Initializes the VideoConverter.

        Args:
            compression_quality (int): Quality level for video compression (0-100).
            video_params (dict): containing frame_w, frame_h, fps.
            output_video (str): Path to the output video file.
            out_folder_buffer (str): Directory containing buffer binary files.
            log_progress (bool): Whether to log progress messages.
            msg_out (Queue): Queue to send progress messages.
            stop_program_event (threading.Event(), optional): threading event for handling the force quit of the application
        """
        self.compression_quality = compression_quality
        self.frame_w = video_params['frame_w']
        self.frame_h = video_params['frame_h']
        self.fps = video_params['fps']
        self.output_video = output_video
        self.out_folder_buffer = Path(out_folder_buffer)
        self.log_progress = log_progress
        self.msg_out = msg_out
        self.stop_program_event = stop_program_event

        self.video_conv_queue = Queue(maxsize=1500)
        self.video_proc = None
        self.saving_error = False

    def __map_cv_quality_to_ffmpeg_q(self, imwrite_quality):
        """
        Maps OpenCV imwrite quality (0-100) to FFmpeg quality scale (2-31).

        Args:
            imwrite_quality (int): OpenCV imwrite quality.

        Returns:
            int: Corresponding FFmpeg quality value.
        """
        return max(2, min(31, int(31 - (imwrite_quality * 29 / 100))))

    def save_video_process(self):
        """
        Saves frames from the video_conv_queue to the output video using FFmpegWriter.
        Deletes buffer files upon successful saving.
        """

        if self.log_progress:
            logger.info(f"VideoConverter: save_video_process started for video: {self.output_video}")

        try:
            input_dict = {
                '-pix_fmt': 'rgb24',
                '-r': str(self.fps)
            }

            video_compression_quality = self.__map_cv_quality_to_ffmpeg_q(self.compression_quality)

            output_dict = {
                '-c:v': 'mjpeg',                          # Use MJPEG codec
                '-q:v': str(video_compression_quality),
                '-pix_fmt': 'yuvj420p',
                '-r': str(self.fps),
                '-threads': '0'
            }

            # video saving using ffmpeg
            # video_save_command = [
            #     'ffmpeg', '-y', '-f', 'rawvideo', '-vcodec', 'rawvideo',
            #     '-s', f'{self.frame_w}x{self.frame_h}', '-pix_fmt', 'rgb24', '-r', str(self.fps),
            #     '-i', '-', '-an', '-vcodec', 'libx264', '-pix_fmt', 'yuv420p', self.output_video,
            #     '-crf', str(video_compression_quality),  
            #     '-preset', 'slow',
            #     '-loglevel', 'error'  # Suppress all output except for errors
            # ]            

            # self.video_proc = subprocess.Popen(video_save_command, stdin=subprocess.PIPE)

            frame_received = False

            try:
                self.video_proc = FFmpegWriter(self.output_video, inputdict=input_dict, outputdict=output_dict)

                while True:
                    try:
                        frame = self.video_conv_queue.get(timeout=10)  # Set timeout to 10 seconds
                    except Exception as e:
                        logger.error("VideoConverter: Timeout while waiting for frame.")
                        self.saving_error = True
                        break

                    if frame is None:
                        logger.info("VideoConverter: Received termination signal.")
                        break      
                    else:
                        self.video_proc.writeFrame(frame)
                        # self.video_proc.stdin.write(frame.tobytes())

                        if not frame_received:
                            frame_received = True                
            except Exception as e:
                logger.error(f"VideoConverter: Error writing frame to video | {e}")
                self.saving_error = True
            finally:
                if frame_received:
                    self.video_proc.close()
                    # self.video_proc.stdin.close()
                    # self.video_proc.wait()
                    logger.info("VideoConverter: Video process closed.")

            if not self.saving_error and not self.stop_program_event.is_set():
                # Delete binary files after successful video creation
                buffer_files = sorted(self.out_folder_buffer.glob('buffer_*.bin'))
                logger.info(f"VideoConverter: Deleting buffer files: {len(buffer_files)} in folder {self.out_folder_buffer}")

                for buffer_file in buffer_files:
                    try:
                        os.remove(buffer_file)
                    except Exception as e:
                        logger.error(f"VideoConverter: Failed to delete {buffer_file} | {e}")
        except Exception as e:
            logger.error(f"VideoConverter: Error during video saving process | {e}")

    def convert_saved_frames(self):
        """
        Converts saved binary frame files into video frames and queues them for saving.
        Sends progress messages if log_progress is True.
        """
        # Gather and sort all binary files
        buffer_files = sorted(self.out_folder_buffer.glob('buffer_*.bin'))

        msg = {'type': 'num_buffer_files', 'value': len(buffer_files)}
        self.msg_out.put(msg)
        
        if self.log_progress:
            logger.info(f"VideoConverter: Number of buffer files to process: {len(buffer_files)}")

        if len(buffer_files) > 0:
            self.writer_video_proc = threading.Thread(target=self.save_video_process)
            self.writer_video_proc.start()

            for idx, buffer_file in enumerate(buffer_files):
                logger.info(f"VideoConverter - Processing {idx+1}/{len(buffer_files)} - {buffer_file}")

                try:
                    with open(buffer_file, 'rb') as f:      
                        while True:
                            # Read the length of the compressed frame (4 bytes)
                            length_bytes = f.read(4)

                            if not length_bytes:
                                break

                            frame_length = struct.unpack('I', length_bytes)[0]

                            # Read the compressed frame data
                            frame_data = f.read(frame_length)

                            if len(frame_data) != frame_length:
                                logger.warning(f"VideoConverter: Unexpected frame length in {buffer_file}")
                                break

                            # Decompress the frame (assuming PNG compression)
                            nparr = np.frombuffer(frame_data, np.uint8)
                            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                            if frame is not None:
                                self.video_conv_queue.put(frame)
                            else:
                                logger.warning(f"VideoConverter: Failed to decode frame in {buffer_file}")
                except Exception as e:
                    logger.error(f"VideoConverter: Error processing buffer file {buffer_file} | {e}")
                    break

                msg = {'type': 'buffer_conv_progress', 'value': idx+1}
                self.msg_out.put(msg)

                # check if the stop_program has been set - if yes stop the conversion
                if self.stop_program_event.is_set():
                    self.video_conv_queue.put(None)
                    break
                
                if self.log_progress:
                    logger.info(f"VideoConverter: Completed processing buffer file {buffer_file}")

            # Signal the end of the video conversion
            self.video_conv_queue.put(None)

            self.writer_video_proc.join()
        else:
            logger.warning("VideoConverter: No buffer files to process.")

        logger.info("VideoConverter: Conversion process completed.")

        msg = {'type': 'video_conversion_done'}
        self.msg_out.put(msg)