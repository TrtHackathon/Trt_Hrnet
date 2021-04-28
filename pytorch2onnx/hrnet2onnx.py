
import _init_paths

from config import cfg
from config import update_config

import numpy as np
import argparse
import models
import torch
import sys
import os

def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    # general
    parser.add_argument('--cfg', help='experiment configure file name', type=str, required=True)

    parser.add_argument('opts', help="Modify config options using the command-line", default=None, nargs=argparse.REMAINDER)

    parser.add_argument('--modelDir', help='model directory', type=str, required=True)
    parser.add_argument('--logDir', help='log directory', type=str, default='')
    parser.add_argument('--dataDir', help='data directory', type=str, default='')
    parser.add_argument('--prevModelDir', help='prev Model directory', type=str, default='')
    parser.add_argument('--outputDir', help='prev Model directory', type=str, default='./')

    args = parser.parse_args()
    return args

args = parse_args()
update_config(cfg, args)

#print(cfg)
#print(cfg.MODEL.HEATMAP_SIZE)
#sys.exit("")

class TrtHrnet(torch.nn.Module):
    def __init__(self):
        super(TrtHrnet, self).__init__()
        self.mean = torch.tensor((- 0.485 * 255,- 0.456*255,- 0.406*255))
        self.std = torch.tensor((1.0/(0.229 * 255), 1.0/(0.224*255), 1.0/(0.225*255)))
        self.model = models.pose_hrnet.get_pose_net(cfg, False)
        self.model.load_state_dict(torch.load(args.modelDir, map_location=torch.device('cpu')), strict=False)

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
#load
#model.load_state_dict(torch.load("pose_hrnet_w48_384x288.pth",map_location=torch.device('cpu')), strict=False)

model = TrtHrnet()
#print(model)

#inference
model.eval()

data  = np.random.randint(0,255,size=(1, 384, 288, 3)).astype(np.float32)
#dummy_input1 = torch.rand(1, 384, 288, 3, dtype = torch.int8)
dummy_input1 = torch.from_numpy(data)

input_names = ["input"]
output_names = ["prob","indices"]

torch.onnx.export(model, dummy_input1, os.path.join(args.outputDir,"pose_hrnet_w48_384x288.onnx"), verbose=True, input_names=input_names, output_names=output_names)
