
import _init_paths

from config import cfg
from config import update_config

import cv2
import sys
import argparse
import models
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from core.inference import get_final_preds

def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    # general
    parser.add_argument('--cfg', default="w48_384x288_adam_lr1e-3.yaml", help='experiment configure file name', type=str)

    parser.add_argument('opts', help="Modify config options using the command-line", default=None, nargs=argparse.REMAINDER)

    parser.add_argument('--modelDir', help='model directory', type=str, default='')
    parser.add_argument('--logDir', help='log directory', type=str, default='')
    parser.add_argument('--dataDir', help='data directory', type=str, default='')
    parser.add_argument('--prevModelDir', help='prev Model directory', type=str, default='')

    args = parser.parse_args()
    return args

args = parse_args()
update_config(cfg, args)

class TrtHrnet(torch.nn.Module):
    def __init__(self):
        super(TrtHrnet, self).__init__()
        self.mean = torch.tensor((- 0.485 * 255,- 0.456*255,- 0.406*255))
        self.std = torch.tensor((1.0/(0.229 * 255), 1.0/(0.224*255), 1.0/(0.225*255)))
        self.model = models.pose_hrnet.get_pose_net(cfg, False)
        self.model.load_state_dict(torch.load("pose_hrnet_w48_384x288.pth",map_location=torch.device('cpu')), strict=False)

    def forward(self, x):
        #x = x.float()
        x = x + self.mean
        x = x * self.std
        x = x.permute(0,3,1,2)
        x = self.model(x)
        x = x.reshape(-1, 17, cfg.MODEL.HEATMAP_SIZE[0]*cfg.MODEL.HEATMAP_SIZE[1])
        val,idx = torch.topk(x, 1,dim = 2)
        idx = idx.float()
        return val,idx

#print(cfg)
#read model
#model = models.pose_hrnet.get_pose_net(cfg, False)
#model.load_state_dict(torch.load("pose_hrnet_w48_384x288.pth",map_location=torch.device('cpu')), strict=False)

model = TrtHrnet()
src_img = Image.open("test.jpg")
input_img = src_img.resize((288,384))
#print(src_img.size,input_img.size)
transform0 = transforms.Compose([transforms.ToTensor()])

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

input = transform0(input_img).unsqueeze(0)
input = input.permute(0,2,3,1) * 255

input.detach().numpy().tofile("input.txt")

#x = transform(input_img)
#print(x)
#print(model)
'''
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

input = transform(input_img)
input = input.unsqueeze(0)

print(input.shape)
#sys.exit("")
#load
model.load_state_dict(torch.load("pose_hrnet_w48_384x288.pth",map_location=torch.device('cpu')), strict=False)

#inference
model.eval()


#model_.load_state_dict(torch.load("pose_hrnet_w48_384x288.pth",map_location=torch.device('cpu')), strict=False)

model_.eval()
#input_np = np.random.randn(1, 3, 384, 288).astype(np.float32)
#input_np.tofile("input.txt")
#input_tensor = torch.from_numpy(input_np)
#dummy_input1 = torch.randn(1,3, 384, 288)

output = model_(input)
print(output)
'''
#input_np = np.random.randint(-128, 127, size = (1,3, 384, 288)).astype(np.float32)/128
#input_np.tofile("input.txt")
#input_tensor = torch.from_numpy(input_np)

val,idx = model.forward(input)
val = val.detach().numpy()
idx = idx.detach().numpy()
#print(x)
#print(hotmap.reshape(-1)[:10])


#print(model,model_)
'''
coords, _ = get_final_preds(
    cfg,
    hotmap.cpu().detach().numpy(),
    np.asarray([]),
    np.asarray([]))

print(cfg.MODEL.HEATMAP_SIZE)
'''
image = cv2.imread("test.jpg");
print (idx)
for id in idx[0]:
    #print(id[0])
    height = (int)(id[0] / cfg.MODEL.HEATMAP_SIZE[0]) * 4
    width = ((int)(id[0] + 0.5) % cfg.MODEL.HEATMAP_SIZE[0]) * 4
    print(height, width)
    cv2.circle(image,(width, height), 3, (255,0,255), -1)
#for coor in coords[0]:
#    print(coor)
#    cv2.circle(image,(int(coor[0]) * 4,int(coor[1]) * 4), 3, (255,0,255), -1)

cv2.imwrite("xx.jpg",image)

#print (coords)
#print (val,idx)

#input_names = ["input"]
#output_names = ["output"]

#torch.onnx.export(model, dummy_input1, "pose_hrnet_w48_384x288.onnx", verbose=True, input_names=input_names, output_names=output_names)
