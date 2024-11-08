import torch
import json
import numpy as np
import cv2

class YoloDetector:
    def __init__(self) -> None:
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

    def get_bounding_box(self, frame:np.ndarray) -> dict:
        res = self.model(frame)
        data = json.loads(res.pandas().xyxy[0].to_json(orient="records"))
        data = data[0] if len(data) > 0 else {'confidence':0}
        return data

# model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
# cap = cv2.VideoCapture('samples/tape2.mp4')

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break
#     cpy = frame.copy()

#     res = model(frame)
#     data = json.loads(res.pandas().xyxy[0].to_json(orient="records"))
#     data = data if len(data) > 0 else {'confidence':0}
#     print(data)
#     # if data['confidence'] > 0.1:
#     #     cv2.rectangle(cpy, (int(data['xmin']), int(data['ymin'])), (int(data['xmax']), int(data['ymax'])), (255,0,255), 3)

#     cv2.imshow('copy frame', cpy)
#     if cv2.waitKey(1) == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()