import quaternion
import numpy as np
import os
import cv2
import json 
import math
import rtde.rtde as rtde
import rtde.rtde_config as rtde_config

os.environ['DEBUG'] = ''

import torch
import torch.nn as nn
import torchvision.transforms as transforms

from Models.networks.sync_batchnorm import convert_model
from Models.base_model import BaseModel

from options.options import get_model

from PIL import Image
import time

torch.backends.cudnn.enabled = True

class SynSinModel:
    def __init__(self, model_path:str, focal_length:int) -> None:
        self.focal_length = focal_length
        self.baseline = 0.15
        self.averages = []
        self.average_depth = 0
        self.current_depth = 0
        opts = torch.load(model_path)['opts']
        opts.render_ids = [1]
        model = get_model(opts)
        torch_devices = [int(gpu_id.strip()) for gpu_id in opts.gpu_ids.split(",")]

        if 'sync' in opts.norm_G:
            model = convert_model(model)
            model = nn.DataParallel(model, torch_devices[0:1]).cuda()
        else:
            model = nn.DataParallel(model, torch_devices[0:1]).cuda()

        #  Load the original model to be tested
        self.model_to_test = BaseModel(model, opts)
        self.model_to_test.load_state_dict(torch.load(model_path)['state_dict'])
        self.model_to_test.eval()

        print("Loaded model")

        self.transform = transforms.Compose([
            transforms.Resize((256,256)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        
        # theta = -0.1
        # phi = 0.1
        tx = 0
        ty = -0.15
        tz = 0

        angle_degrees = 10
        angle_radians = np.deg2rad(angle_degrees)
        rotation_matrix = np.array([
            [np.cos(angle_radians), 0, np.sin(angle_radians)],
            [0, 1, 0],
            [-np.sin(angle_radians), 0, np.cos(angle_radians)]
        ])

        self.RT = torch.eye(4).unsqueeze(0)
        # RT[0,0:3,0:3] = torch.Tensor(quaternion.as_rotation_matrix(quaternion.from_rotation_vector([phi, theta, 0])))
        self.RT[0,0:3,0:3] = torch.Tensor(quaternion.as_rotation_matrix(quaternion.from_rotation_matrix(rotation_matrix)))
        self.RT[0,0:3,3] = torch.Tensor([tx, ty, tz])
        self.batch = {
            'cameras' : [{
                'K' : torch.eye(4).unsqueeze(0),
                'Kinv' : torch.eye(4).unsqueeze(0)
            }]
        }
        
    def get_pred_frame(self, frame:np.ndarray):
        im = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        temp = frame.copy()
        temp = cv2.resize(temp, (256,256))
        im = self.transform(im)
        self.batch['images'] = [im.unsqueeze(0)]
        with torch.no_grad():
            pred_imgs = self.model_to_test.model.module.forward_angle(self.batch, [self.RT])
        pred_frame = pred_imgs[0].squeeze().cpu().permute(1,2,0).numpy() * 0.5 + 0.5
        pred_frame = np.array(pred_frame)
        pred_frame = cv2.normalize(pred_frame, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U)
        pred_frame = cv2.cvtColor(pred_frame, cv2.COLOR_RGB2BGR)

        current_depth, self.averages = self.compute_depth(temp, pred_frame, self.averages.copy())
        # print(f"Average Depth: {sum(self.averages)/len(self.averages)}, Current Depth: {current_depth}")

        return pred_frame
    
    def run_pred_video(self, cap:cv2.VideoCapture, save_output:bool=False):
        if save_output:
            fwidth  = int(cap.get(3))
            fheight = int(cap.get(4))
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            output_filename = 'results_generated.mp4'
            res_cap = cv2.VideoWriter(output_filename, fourcc, 60.0, (fwidth, fheight))

        print("video loaded")
        frame_counter = 0
        averages = []
        start = time.time()
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            im = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            temp = frame.copy()
            temp = cv2.resize(temp, (256,256))
            im = self.transform(im)
            self.batch['images'] = [im.unsqueeze(0)]

            with torch.no_grad():
                pred_imgs = self.model_to_test.model.module.forward_angle(self.batch, [self.RT])

            pred_frame = pred_imgs[0].squeeze().cpu().permute(1,2,0).numpy() * 0.5 + 0.5
            pred_frame = np.array(pred_frame)
            pred_frame = cv2.normalize(pred_frame, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U)
            pred_frame = cv2.cvtColor(pred_frame, cv2.COLOR_RGB2BGR)
            
            frame_counter += 1

            current_depth, averages = self.compute_depth(temp, pred_frame, averages.copy())
            # print(f"Average Depth: {sum(averages)/len(averages)}, Current Depth: {current_depth}")
            cv2.imshow("original", temp)
            cv2.imshow("generated", pred_frame)
            
            if save_output:
                res_cap.write(pred_frame)

            if cv2.waitKey(1) == ord('q'):
                break
            elif cv2.waitKey(1) == ord('p'):
                cv2.waitKey(-1)

        curr = time.time() - start
        print(f"Frame Counter: {frame_counter}, Time Elapsed: {curr}, FPS: {frame_counter//curr}")
        if save_output:
            res_cap.release()
        cv2.destroyAllWindows()

    def compute_depth(self, frame1, frame2, averages):
        average_depth = 0
        img1 = frame1
        img2 = frame2
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        stereo = cv2.StereoBM_create(numDisparities=32, blockSize=15)
        disparity = stereo.compute(gray1, gray2)
        # disp_norm = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        depth_data = (self.focal_length * self.baseline) / (disparity + 0.0001)

        total_depth = 0
        count = 0
        # print("Depth:", depth_data)
        for depthrow in depth_data:
            max_depth_in_row = max(depthrow)
            if max_depth_in_row > 0 and max_depth_in_row < self.baseline*5:
                total_depth += max_depth_in_row
                count += 1

        try:    
            average_depth = total_depth/count
            averages.append(average_depth)
        except:
            average_depth = sum(averages)/len(averages)
        finally:
            self.average_depth = average_depth
            return average_depth, averages

    def get_bounded_frame(self, frame: np.ndarray) -> np.ndarray:
        res = frame.copy()
        gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 115, 255, cv2.THRESH_BINARY_INV)

        cv2.imshow('thresh', gray)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        contours, heirarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        x = y = 50000
        w = h = 0

        for contour in contours:
            x_t,y_t,w_t,h_t = cv2.boundingRect(contour)
            x = x_t if x_t < x else x
            w = w_t if w_t > w else w
            y = y_t if y_t < y else y
            h = h_t if h_t > h else h

        cv2.rectangle(res, (x,y), (x+w, y+h), (0,255,0), 5)

        return res

    def get_focal_length(self) -> int:
        return self.focal_length

class YoloDetector:
    def __init__(self) -> None:
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

    def get_bounding_box(self, frame:np.ndarray) -> dict:
        res = self.model(frame)
        data = json.loads(res.pandas().xyxy[0].to_json(orient="records"))
        data = data[0] if len(data) > 0 else {'confidence':0}
        return data

class RobotController:
    def __init__(self, coords:list, f:int) -> None:
        self.u = coords[0]
        self.v = coords[1]
        self.f = f
        self.Z = 0
        self.L = np.zeros([2, 6])
        self.error_matrix = np.zeros([1,2])
        self.jacobian = np.zeros([6,6])
        self.DH_PARAMS = [{"theta": 0, "a": 0, "d": 0.1625, "alpha": np.pi / 2},
                          {"theta": 0, "a": -0.425, "d": 0, "alpha": 0},
                          {"theta": 0, "a": -0.3922, "d": 0, "alpha": 0},
                          {"theta": 0, "a": 0, "d": 0.1333, "alpha": np.pi / 2},
                          {"theta": 0, "a": 0, "d": 0.0997, "alpha": -np.pi / 2},
                          {"theta": 0, "a": 0, "d": 0.0996, "alpha": 0}]

    def transformation_matrix(self, theta, a, d, alpha):    
        """Compute the individual transformation matrix using DH parameters."""    
        ct, st = np.cos(theta), np.sin(theta)    
        ca, sa = np.cos(alpha), np.sin(alpha)    
        return np.array([[ct, -st * ca, st * sa, a * ct],
                        [st, ct * ca, -ct * sa, a * st],
                        [0, sa, ca, d],
                        [0, 0, 0, 1]])
    
    def calculate_jacobian(self, q):
        """    Compute the Jacobian matrix for UR5e given joint angles q.    :param q: List or array of joint angles [q1, q2, q3, q4, q5, q6]    :return: 6x6 Jacobian matrix    """    
        # Initialize transformation matrices    
        T = np.eye(4)    
        transforms = []    
        # Compute the transformation matrix from base to each joint    
        for i in range(6):        
            theta = q[i] + self.DH_PARAMS[i]["theta"]  
            # Add joint angle to theta        
            a = self.DH_PARAMS[i]["a"]        
            d = self.DH_PARAMS[i]["d"]        
            alpha = self.DH_PARAMS[i]["alpha"]        
            T_i = self.transformation_matrix(theta, a, d, alpha)        
            T = T @ T_i  
            # Cumulative transformation from base to joint i        
            transforms.append(T)    
        # End effector position    
        o_n = T[:3, 3]    
        # Initialize Jacobian matrix    
        J = np.zeros((6, 6))    
        for i in range(6):        
            o_i = transforms[i][:3, 3]        
            z_i = transforms[i][:3, 2]        
            # Linear velocity part        
            J[:3, i] = np.cross(z_i, o_n - o_i)        
            # Angular velocity part        
            J[3:, i] = z_i       
        
        self.jacobian = J

    def calculate_error_matrix(self, u_detected, v_detected):
        u_delta = self.u - u_detected
        v_delta = self.v - v_detected
        self.error_matrix = np.array([u_delta, v_delta])

    def calcualte_image_jacobian(self, Z):
        self.Z = Z
        Z=-Z
        self.L[0][0] = -self.f/Z
        self.L[0][2] = self.u/Z
        self.L[0][3] = self.u*self.v/self.f
        self.L[0][4] = -((self.f**2)+(self.u**2))/self.f
        self.L[0][5] = self.v
        self.L[1][1] = -self.f/Z
        self.L[1][2] = self.v/Z
        self.L[1][3] = ((self.f**2)+(self.v**2))/self.f
        self.L[1][4] = -self.u*self.v/self.f
        self.L[1][5] = -self.u

    def old_calculate_jacobian(self, q):
        pi = 3.1415926
        d4=0.1333
        d5=0.0997
        d6=0.0996
        a2=-0.425
        a3=-0.3922
        
        self.jacobian[0][0] = (d4 * math.cos(q[0]))  + d6 * math.cos(q[0])*math.cos(q[4])  + (-a2) * math.cos(q[1])*math.sin(q[0]) - (-a3 * math.sin(q[0])*math.sin(q[1])*math.sin(q[2])) + (d6 * math.cos(q[1] + q[2] + q[3])*math.sin(q[0])*math.sin(q[4])) - (d5 * math.cos(q[1] + q[2])*math.sin(q[0])*math.sin(q[3]))  - (d5 * math.sin(q[1] + q[2])*math.cos(q[3])*math.sin(q[0])) + (-a3 * math.cos(q[1])*math.cos(q[2])*math.sin(q[0])) 

        self.jacobian[0][1] = (math.cos(q[0])*(d5 * math.cos(q[1] + q[2] + q[3]) - d6/2 * math.cos(q[1] + q[2] + q[3] + q[4]) + -a3 * math.sin(q[1] + q[2]) + -a2 * math.sin(q[1]) + d6/2 * math.cos(q[1] + q[2] + q[3] - q[4]))) 
        self.jacobian[0][2] = (math.cos(q[0])*(d5 * math.cos(q[1] + q[2] + q[3]) - d6/2 * math.cos(q[1] + q[2] + q[3] + q[4]) + -a3 * math.sin(q[1] + q[2]) + d6/2 * math.cos(q[1] + q[2] + q[3] - q[4])))

        self.jacobian[0][3] = (math.cos(q[0])*(d6 * math.sin(q[1] + q[2] + q[3])*math.sin(q[4]) + d5 * math.cos(q[1] + q[2])*math.cos(q[3]) - d5 * math.sin(q[1] + q[2])*math.sin(q[3]))) 

        self.jacobian[0][4] = -(d6 * math.sin(q[0])*math.sin(q[4]))  - (d6 * math.cos(q[1] + q[2] + q[3])*math.cos(q[0])*math.cos(q[4])) 
        self.jacobian[0][5] = 0

        self.jacobian[1][0] = (d4 * math.sin(q[0]))  - (-a2 * math.cos(q[0])*math.cos(q[1])) + (d6 * math.cos(q[4])*math.sin(q[0]))  - (d6 * math.cos(q[1] + q[2] + q[3])*math.cos(q[0])*math.sin(q[4]))  + (d5 * math.cos(q[1] + q[2])*math.cos(q[0])*math.sin(q[3]))  + (d5 * math.sin(q[1] + q[2])*math.cos(q[0])*math.cos(q[3]))  - (-a3 * math.cos(q[0])*math.cos(q[1])*math.cos(q[2]))  + (-a3 * math.cos(q[0])*math.sin(q[1])*math.sin(q[2])) 

        self.jacobian[1][1] = (math.sin(q[0])*(d5 * math.cos(q[1] + q[2] + q[3]) - d6/2 * math.cos(q[1] + q[2] + q[3] + q[4]) + -a3 * math.sin(q[1] + q[2]) + -a2 * math.sin(q[1]) + d6/2 * math.cos(q[1] + q[2] + q[3] - q[4]))) 

        self.jacobian[1][2] = (math.sin(q[0])*(d5 * math.cos(q[1] + q[2] + q[3]) - d6/2 * math.cos(q[1] + q[2] + q[3] + q[4]) + -a3 * math.sin(q[1] + q[2]) + d6/2 * math.cos(q[1] + q[2] + q[3] - q[4])))

        self.jacobian[1][3] = (math.sin(q[0])*(d6 * math.sin(q[1] + q[2] + q[3])*math.sin(q[4]) + d5 * math.cos(q[1] + q[2])*math.cos(q[3]) - d5 * math.sin(q[1] + q[2])*math.sin(q[3])))

        self.jacobian[1][4] = (d6 * math.cos(q[0])*math.sin(q[4]))  - (d6 * math.cos(q[1] + q[2] + q[3])*math.cos(q[4])*math.sin(q[0]))
        self.jacobian[1][5] = 0

        self.jacobian[2][0] = 0
        self.jacobian[2][1] = (d5 * math.cos(q[1] + q[2])*math.sin(q[3]))  - (-a2 * math.cos(q[1]))  - math.sin(q[4])*((d6 * math.cos(q[1] + q[2])*math.cos(q[3]))  - (d6 * math.sin(q[1] + q[2])*math.sin(q[3])) ) - (-a3 * math.cos(q[1] + q[2]))  + (d5 * math.sin(q[1] + q[2])*math.cos(q[3])) 

        self.jacobian[2][2] = (d5 * math.sin(q[1] + q[2] + q[3]))  - (-a3 * math.cos(q[1] + q[2]))  - (d6 * math.cos(q[1] + q[2] + q[3])*math.sin(q[4])) 
        self.jacobian[2][3] = (d5 * math.sin(q[1] + q[2] + q[3]))  - (d6 * math.cos(q[1] + q[2] + q[3])*math.sin(q[4]))
        self.jacobian[2][4] = -(d6 * math.sin(q[1] + q[2] + q[3])*math.cos(q[4]))
        self.jacobian[2][5] = 0

        self.jacobian[3][0] = math.exp(q[5] / 3.1415926)*(math.cos(q[0])*math.cos(q[4]) + math.cos(q[1] + q[2] + q[3])*math.sin(q[0])*math.sin(q[4]))
        self.jacobian[3][1] = math.sin(q[1] + q[2] + q[3])*math.exp(q[5] / 3.1415926)*math.cos(q[0])*math.sin(q[4])
        self.jacobian[3][2] = math.sin(q[1] + q[2] + q[3])*math.exp(q[5] / 3.1415926)*math.cos(q[0])*math.sin(q[4])
        self.jacobian[3][3] = math.sin(q[1] + q[2] + q[3])*math.exp(q[5] / pi)*math.cos(q[0])*math.sin(q[4])
        self.jacobian[3][4] = -math.exp(q[5] / pi)*(math.sin(q[0])*math.sin(q[4]) + math.cos(q[1] + q[2] + q[3])*math.cos(q[0])*math.cos(q[4]))
        self.jacobian[3][5] = (math.exp(q[5] / pi)*(math.cos(q[4])*math.sin(q[0]) - math.cos(q[1] + q[2] + q[3])*math.cos(q[0])*math.sin(q[4]))) / pi

        self.jacobian[4][0] = math.exp(q[5] / pi)*(math.cos(q[4])*math.sin(q[0]) - math.cos(q[1] + q[2] + q[3])*math.cos(q[0])*math.sin(q[4]))
        self.jacobian[4][1] = math.sin(q[1] + q[2] + q[3])*math.exp(q[5] / 3.1415926)*math.sin(q[0])*math.sin(q[4])
        self.jacobian[4][2] = math.sin(q[1] + q[2] + q[3])*math.exp(q[5] / 3.1415926)*math.sin(q[0])*math.sin(q[4])
        self.jacobian[4][3] = math.sin(q[1] + q[2] + q[3])*math.exp(q[5] / pi)*math.sin(q[0])*math.sin(q[4])
        self.jacobian[4][4] = math.exp(q[5] / pi)*(math.cos(q[0])*math.sin(q[4]) - math.cos(q[1] + q[2] + q[3])*math.cos(q[4])*math.sin(q[0]))
        self.jacobian[4][5] = -(math.exp(q[5] / pi)*(math.cos(q[0])*math.cos(q[4]) + math.cos(q[1] + q[2] + q[3])*math.sin(q[0])*math.sin(q[4]))) / pi

        self.jacobian[5][0] = 0
        self.jacobian[5][1] = -math.cos(q[1] + q[2] + q[3])*math.exp(q[5] / pi)*math.sin(q[4])
        self.jacobian[5][2] = -math.cos(q[1] + q[2] + q[3])*math.exp(q[5] / pi)*math.sin(q[4])
        self.jacobian[5][3] = -math.cos(q[1] + q[2] + q[3])*math.exp(q[5] / pi)*math.sin(q[4])
        self.jacobian[5][4] = -math.sin(q[1] + q[2] + q[3])*math.exp(q[5] / pi)*math.cos(q[4])
        self.jacobian[5][5] = -(math.sin(q[1] + q[2] + q[3])*math.exp(q[5] / pi)*math.sin(q[4])) / pi

    def get_r_dot_matrix(self, lam):
        # J_inv = np.linalg.inv(self.jacobian)
        # print(J_inv)
        # L_ps_inv = np.linalg.pinv(self.L)
        L_transpose = np.transpose(self.L)
        J_transpose = np.transpose(self.jacobian)
        # L_transpose = L_ps_inv
        # J_transpose = J_inv
        r_dot = np.dot((-lam*L_transpose),((self.Z/self.f)*self.error_matrix))
        r_dot = r_dot.round(4)
        # r_dot[0] = -r_dot[0]
        # r_dot[1] = -r_dot[1]
        q_dot = np.dot(J_transpose,r_dot)
        q_dot = q_dot.round(4)

        # Rm = np.array([
        #         [-1,  0,  0],
        #         [ 0, -1,  0],
        #         [ 0,  0,  1]
        #     ])
        
        # v_qdot = q_dot[:3]
        # w_qdot = q_dot[3:]

        # new_v_qdot = np.dot(Rm, v_qdot)
        # new_w_qdot = np.dot(Rm, w_qdot)

        # q_dot = np.hstack((new_v_qdot, new_w_qdot))

        return q_dot, r_dot

class RTDEHandler:
    def __init__(self, host:str, port:int, config_file:str) -> None:
        self.host = host
        self.port = port

        # configurations
        self.conf = rtde_config.ConfigFile(config_file)
        state_names, state_types = self.conf.get_recipe("state")
        setp_names, setp_types = self.conf.get_recipe("setp")
        watchdog_names, watchdog_types = self.conf.get_recipe("watchdog")

        self.con = rtde.RTDE(host, port)
        self.con.connect()

        self.con.get_controller_version()

        self.setp = self.con.send_input_setup(setp_names, setp_types)
        self.setp.input_double_register_0 = 0
        self.setp.input_double_register_1 = 0
        self.setp.input_double_register_2 = 0
        self.setp.input_double_register_3 = 0
        self.setp.input_double_register_4 = 0
        self.setp.input_double_register_5 = 0
        self.watchdog = self.con.send_input_setup(watchdog_names, watchdog_types)
        self.watchdog.input_int_register_0 = 0

        if not self.con.send_output_setup(state_names, state_types):
            print("Unable to configure states")

        if not self.con.send_start():
            print("Unable to start synchronization")

    def list_to_setp(self, sp, list):
        for i in range(0, 6):
            sp.__dict__["input_double_register_%i" % i] = list[i]
        return sp
    
    def get_data(self) -> dict:
        try:
            state = self.con.receive().__dict__
            state["status"] = 200
            return state

        except rtde.RTDEException:
            print("Session error!")
            self.con.disconnect()
            return {"status": 500}
        
    def send_control_position(self, new_setp) -> dict:
        res = {
            "status": 100 # initial condition
        }
        try:
            self.list_to_setp(self.setp, new_setp)
            # print(f"NEW SETP: {new_setp}")
            self.con.send(self.setp)
            res["status"] = 200 # Success
        except rtde.RTDEException:
            res["status"] = 400 # Failure

        finally:
            return res

if __name__ == "__main__":
    host = "10.149.230.20"
    port = 30004
    init_coords = [480//2, 640//2]
    f = 644
    final_setp = [0.72, 0.51, 0.21, 0, 3.11, 0.04]
    actual_q = [1.998331547, -2.044362684, -1.160939217, -1.470036821, 1.388783932, 1.47571516]
    # config_file = "rdte_config_files/main_config.xml"
    # handler = RTDEHandler(host, port, config_file)
    import time
    start = time.time()
    # print(handler.get_data())
    # res = handler.send_control_position(final_setp)
    # print(res)
    controller = RobotController(init_coords, f)
    controller.calcualte_image_jacobian(0.24)
    # controller.calcualte_image_jacobian(temp_depth)
    controller.calculate_jacobian(actual_q)
    # controller.old_calculate_jacobian(actual_q)
    controller.calculate_error_matrix(init_coords[0], init_coords[1])
    q_dot, r_dot = controller.get_r_dot_matrix(lam=0.3)
    end = time.time()
    print(q_dot)
    print("time diff: ", end-start)