from utilCodes import SynSinModel, YoloDetector, RobotController, RTDEHandler
import socket
import struct
import pickle
import cv2

def run_server():
    focal_length = 644
    host = "10.149.230.20"
    # host = "localhost"
    port = 30004
    config_file = "rdte_config_files/main_config.xml"
    init_coordinates = [640//2, 480//2]
    controller = RobotController(init_coordinates,focal_length)
    yolo_model = YoloDetector()
    rdte_handler = RTDEHandler(host, port, config_file)
    model_path = 'modelcheckpoints/realestate/zbufferpts.pth'
    synsin = SynSinModel(model_path, focal_length) # change focal length here as per needed
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind(("localhost", 7080))
    server_socket.listen(2)
    conn, addr = server_socket.accept()

    data = b""
    payload_size = struct.calcsize("Q")

    temp_depth = 0.15
    flag = False
    init_q = [0,0,0,0,0,0]
    while True: #and temp_depth > 0:
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

        '''
        IMPORTANT NOTE: WE HAVE TO CHECK THE COORDINATES WHETHER THEY ARE WITH RESPECT TO
        ORIGINAL FRAME SIZE OR THE RESIZED ONE AFTER THE SYNSIN MODEL TRANSFORMS IT
        '''
        # if not flag:
        #     accept_inp = input("Should we proceed? (Y/N): ")
        #     if accept_inp == 'Y' or accept_inp == 'y':
        #         flag = True                

        res_frame = yolo_model.get_bounding_box(frame)
        if res_frame['confidence'] > 0.1:
            mid_point_bounding_box.append((res_frame['xmin'] + res_frame['xmax'])//2)
            mid_point_bounding_box.append((res_frame['ymin'] + res_frame['ymax'])//2)
            cv2.rectangle(frame, (int(res_frame['xmin']), int(res_frame['ymin'])), (int(res_frame['xmax']), int(res_frame['ymax'])), (255,0,255), 2)
            bounding_area = (res_frame["xmax"] - res_frame["xmin"])*(res_frame["ymax"] - res_frame["ymin"])
            print(f"Bounding Area: {bounding_area}, Threshold: {0.6*(640*480)}, Total: {640*480}")
            if bounding_area > 0.6*(640*480): #stopping condition for arm close to object
                break

        cv2.imshow('received frame', frame)
        
        # print(mid_point_bounding_box)
        if len(mid_point_bounding_box) == 0:
            continue
        
        actual_q = rdte_handler.get_data()["actual_q"]
        
        if round(actual_q[0],3) != round(init_q[0],3) and round(actual_q[1],3) != round(init_q[1],3) and round(actual_q[2],3) != round(init_q[2],3):
            init_q = actual_q
            temp_depth -= 0.015

        print(f"Actual Q: {actual_q}, Datatype: {type(actual_q)}")
        controller.calcualte_image_jacobian(synsin.average_depth)
        # controller.calcualte_image_jacobian(-temp_depth)
        # controller.calculate_jacobian(actual_q)
        controller.old_calculate_jacobian(actual_q)
        controller.calculate_error_matrix(mid_point_bounding_box[0], mid_point_bounding_box[1])
        print(f"X Diff: {init_coordinates[0]}, {mid_point_bounding_box[0]}")
        print(f"Y Diff: {init_coordinates[1]}, {mid_point_bounding_box[1]}")
        q_dot, r_dot = controller.get_r_dot_matrix(lam=0.03)
        res = rdte_handler.send_control_position(q_dot)
        print(res)
        print(f"Q dot: {q_dot}")
        print(f"R dot: {r_dot}")
        print(f"Detected Depth: {synsin.average_depth}")
        if res["status"] != 200:    
            break
        
        rdte_handler.con.send(rdte_handler.watchdog)

        cv2.imshow('predicted frame', pred_frame)
        if cv2.waitKey(1) == ord('q'):
            break
    
    cv2.destroyAllWindows()
    conn.close()
    server_socket.close()
    rdte_handler.watchdog.input_int_register_0 = 1

if __name__ == "__main__":
    run_server()