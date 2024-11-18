import numpy as np
import cv2
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

        cv2.imshow('RealSense', color_image)
        if cv2.waitKey(1) == ord('q'):
            break
finally:
    pipeline.stop()
    cv2.destroyAllWindows()