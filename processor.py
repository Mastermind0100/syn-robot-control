from utilCodes import SynSinModel, YoloDetector, RobotController
import socket
import struct
import pickle
import cv2

def run_server():
    yolo_model = YoloDetector()
    model_path = 'modelcheckpoints/realestate/zbufferpts.pth'
    synsin = SynSinModel(model_path, focal_length=26) # change focal length here as per needed
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind(("localhost", 7080))
    server_socket.listen(2)
    conn, addr = server_socket.accept()

    data = b""
    payload_size = struct.calcsize("Q")

    while True:
        while len(data) < payload_size:
            packet = conn.recv(4096)
            if not packet:
                break

            data += packet

        packed_msg_size = data[:payload_size]
        data = data[payload_size:]
        msg_size = struct.unpack("Q", packed_msg_size)[0]

        while len(data) < msg_size:
            data += conn.recv(4096)

        frame_data = data[:msg_size]
        data = data[msg_size:]

        # Deserialize frame
        frame = pickle.loads(frame_data)
        pred_frame = synsin.get_pred_frame(frame)
        mid_point_bounding_box = []

        # init_coordinates = [0, frame.shape[1]//2] # need to change these for robot control as needed

        res_frame = yolo_model.get_bounding_box(frame)
        if res_frame['confidence'] > 0.1:
            mid_point_bounding_box.append((res_frame['xmin'] + res_frame['xmax'])//2)
            mid_point_bounding_box.append((res_frame['ymin'] + res_frame['ymax'])//2)
            cv2.rectangle(frame, (int(res_frame['xmin']), int(res_frame['ymin'])), (int(res_frame['xmax']), int(res_frame['ymax'])), (255,0,255), 2)
        
        res_pred_frame = yolo_model.get_bounding_box(pred_frame)
        if res_pred_frame['confidence'] > 0.1:
            cv2.rectangle(pred_frame, (int(res_pred_frame['xmin']), int(res_pred_frame['ymin'])), (int(res_pred_frame['xmax']), int(res_pred_frame['ymax'])), (255,0,255), 2)

        cv2.imshow('received frame', frame)
        cv2.imshow('predicted frame', pred_frame)
        if cv2.waitKey(1) == ord('q'):
            break
    
    cv2.destroyAllWindows()
    conn.close()
    server_socket.close()

if __name__ == "__main__":
    run_server()