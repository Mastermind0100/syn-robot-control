import matplotlib.pyplot as plt
import quaternion
import numpy as np
import os
import cv2
from yoloCode import yolo_get_frame

os.environ['DEBUG'] = ''

import torch
import torch.nn as nn
import torchvision.transforms as transforms

from models.networks.sync_batchnorm import convert_model
from models.base_model import BaseModel

from options.options import get_model

from PIL import Image
import time

torch.backends.cudnn.enabled = True

# REALESTATE
MODEL_PATH = 'modelcheckpoints/realestate/zbufferpts.pth'

BATCH_SIZE = 1

opts = torch.load(MODEL_PATH)['opts']
opts.render_ids = [1]

model = get_model(opts)

torch_devices = [int(gpu_id.strip()) for gpu_id in opts.gpu_ids.split(",")]
device = 'cuda:' + str(torch_devices[0])

if 'sync' in opts.norm_G:
    model = convert_model(model)
    model = nn.DataParallel(model, torch_devices[0:1]).cuda()
else:
    model = nn.DataParallel(model, torch_devices[0:1]).cuda()


#  Load the original model to be tested
model_to_test = BaseModel(model, opts)
model_to_test.load_state_dict(torch.load(MODEL_PATH)['state_dict'])
model_to_test.eval()

print("Loaded model")

# Load the image
transform = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

cap = cv2.VideoCapture('samples/sample3.mp4')
print("video loaded")
frame_counter = 0
start = time.time()
while True:
    # im = Image.open('./demos/im.jpg')
    ret, frame = cap.read()
    if not ret:
        break
    im = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    temp = frame.copy()
    temp = cv2.resize(temp, (256,256))
    im = transform(im)

    # Parameters for the transformation
    theta = -0.15
    phi = -0.2
    tx = 0
    ty = 0.15
    tz = 0

    RT = torch.eye(4).unsqueeze(0)
    # Set up rotation
    # RT[0,0:3,0:3] = torch.Tensor(quaternion.as_rotation_matrix(quaternion.from_rotation_vector([phi, theta, 0])))
    RT[0,0:3,0:3] = torch.Tensor(quaternion.as_rotation_matrix(quaternion.from_euler_angles(0, 0.2, 0)))
    # Set up translation
    RT[0,0:3,3] = torch.Tensor([tx, ty, tz])

    batch = {
        'images' : [im.unsqueeze(0)],
        'cameras' : [{
            'K' : torch.eye(4).unsqueeze(0),
            'Kinv' : torch.eye(4).unsqueeze(0)
        }]
    }

    # Generate a new view at the new transformation
    with torch.no_grad():
        pred_imgs = model_to_test.model.module.forward_angle(batch, [RT])
        # depth = nn.Sigmoid()(model_to_test.model.module.pts_regressor(batch['images'][0].cuda()))

    pred_frame = pred_imgs[0].squeeze().cpu().permute(1,2,0).numpy() * 0.5 + 0.5
    pred_frame = np.array(pred_frame)

    yolo_orig = yolo_get_frame(temp)
    yolo_new = yolo_get_frame(pred_frame)

    cv2.imshow("original", yolo_orig)
    cv2.imshow("generated", yolo_new)
    frame_counter += 1
    curr = time.time() - start
    print(f"Frame Counter: {frame_counter}, Time Elapsed: {curr}, FPS: {frame_counter//curr}")

    if cv2.waitKey(1) == ord('q'):
        break
    elif cv2.waitKey(1) == ord('p'):
        cv2.waitKey(-1)

cap.release()
cv2.destroyAllWindows()