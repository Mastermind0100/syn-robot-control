from utilCodes import SynSinModel, YoloDetector, RobotController, RTDEHandler
import socket
import struct
import pickle
import cv2

def run_server():
    focal_length = 644
    host = "localhost" # "10.149.230.20"
    port = 30004
    record_config_file = "rdte_config_files/record_config.xml"
    control_config_file = "rdte_config_files/control_config.xml"
    init_coordinates = [480//2, 640//2]
    controller = RobotController(init_coordinates,focal_length)
    yolo_model = YoloDetector()
    rdte_handler = RTDEHandler(host, port, record_config_file, control_config_file)
    model_path = 'modelcheckpoints/realestate/zbufferpts.pth'
    synsin = SynSinModel(model_path, focal_length) # change focal length here as per needed
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind(("localhost", 7080))
    server_socket.listen(2)
    conn, addr = server_socket.accept()

    data = b""
    payload_size = struct.calcsize("Q")

    temp_depth = 0.15

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

         # need to change these for robot control as needed
        print(f"frame shape{frame.shape}")

        '''
        IMPORTANT NOTE: WE HAVE TO CHECK THE COORDINATES WHETHER THEY ARE WITH RESPECT TO
        ORIGINAL FRAME SIZE OR THE RESIZED ONE AFTER THE SYNSIN MODEL TRANSFORMS IT
        '''

        res_frame = yolo_model.get_bounding_box(frame)
        if res_frame['confidence'] > 0.1:
            mid_point_bounding_box.append((res_frame['xmin'] + res_frame['xmax'])//2)
            mid_point_bounding_box.append((res_frame['ymin'] + res_frame['ymax'])//2)
            cv2.rectangle(frame, (int(res_frame['xmin']), int(res_frame['ymin'])), (int(res_frame['xmax']), int(res_frame['ymax'])), (255,0,255), 2)
        
        print(mid_point_bounding_box)
        if len(mid_point_bounding_box) == 0:
            continue

        # controller.calcualte_image_jacobian(synsin.average_depth)
        controller.calcualte_image_jacobian(temp_depth)
        controller.calculate_jacobian(rdte_handler.get_data()["actual_q"])
        controller.calculate_error_matrix(mid_point_bounding_box[0], mid_point_bounding_box[1])
        q_dot, r_dot = controller.get_r_dot_matrix(lam=0.3)

        res = rdte_handler.send_control_position(q_dot)
        if res["status"] != 200:
            print(res)
            break
        else:
            temp_depth -= 0.015

        # cv2.imshow('received frame', frame)
        # cv2.imshow('predicted frame', pred_frame)
        if cv2.waitKey(1) == ord('q'):
            break
    
    cv2.destroyAllWindows()
    conn.close()
    server_socket.close()

if __name__ == "__main__":
    run_server()