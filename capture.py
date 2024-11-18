import numpy as np
import socket
import pickle
import struct
import cv2

client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect(("localhost", 7080))
connection = client_socket.makefile('wb')

def socket_handler(frame):

    data = pickle.dumps(frame)
    message = struct.pack("Q", len(data)) + data
    client_socket.sendall(message)

def capture_stream(camera_type:str, filepath:str=None):
    if camera_type == 'rs':
        import pyrealsense2 as rs
        pipeline = rs.pipeline()
        config = rs.config()

        # Camera being used: Intel RealSense Depth Camera D435i
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

        pipeline.start(config)
        
        try:
            while True:
                frames = pipeline.wait_for_frames()
                color_frame = frames.get_color_frame()
                if not color_frame:
                    continue

                color_image = np.asanyarray(color_frame.get_data())
                socket_handler(color_image)
                cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
                cv2.imshow('RealSense', color_image)
                if cv2.waitKey(1) == ord('q'):
                    break

        finally:
            pipeline.stop()

    elif camera_type == 'cv':
        source = filepath if filepath != None else 0
        cap = cv2.VideoCapture(source)

        while True:
            ret, frame = cap.read()
            socket_handler(frame)
            # cv2.imshow('opencvOutput', frame)
            if cv2.waitKey(1) == ord('q'):
                break
        cap.release()
    
    cv2.destroyAllWindows()

if __name__ == "__main__":
    camera_type = 'rs'
    filename = 'samples/pot.mp4'

    print("Reading Video Capture...")
    capture_stream(camera_type, filename)
    print("Done!")