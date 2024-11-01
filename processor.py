from synsin import SynSinModel
import socket
import struct
import pickle
import cv2

def run_server():
    model_path = 'modelcheckpoints/realestate/zbufferpts.pth'
    synsin = SynSinModel(model_path)
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
        og_frame, pred_frame = synsin.get_pred_frame(frame)

        cv2.imshow('received frame', frame)
        cv2.imshow('predicted frame', pred_frame)
        if cv2.waitKey(1) == ord('q'):
            break
    
    cv2.destroyAllWindows()
    conn.close()
    server_socket.close()

if __name__ == "__main__":
    run_server()