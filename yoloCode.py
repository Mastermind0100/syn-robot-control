from yolov6.core.inferer import Inferer
import cv2

def test():
    cap = cv2.VideoCapture('samples/sample3.mp4')
    inferer = Inferer(source='samples/sample3.mp4', 
                    webcam=False, 
                    webcam_addr=0, 
                    weights='yolov6m.pt', 
                    device='0', 
                    yaml='data/coco.yaml', 
                    img_size=640, 
                    half=False)

    while True:
        ret, frame = cap.read()
        pred_frame = inferer.image_infer(frame)

        cv2.imshow('sample', pred_frame)
        if cv2.waitKey(1) == ord('q'):
            break
        elif cv2.waitKey(1) == ord('p'):
            cv2.waitKey(-1)

    cv2.destroyAllWindows()
    cap.release()


def yolo_get_frame(frame):
    inferer = Inferer(source='samples/sample3.mp4', 
                  webcam=False, 
                  webcam_addr=0, 
                  weights='yolov6m.pt', 
                  device='0', 
                  yaml='data/coco.yaml', 
                  img_size=640, 
                  half=False)
    
    pred_img = inferer.image_infer(frame)

    return pred_img