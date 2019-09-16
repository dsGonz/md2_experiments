# Import packages
from __future__ import absolute_import, division, print_function

import os
import cv2
import numpy as np
import PIL.Image as pil
import matplotlib.pyplot as plt
import time
import datasets
from os.path import abspath, join
from evaluate_depth import compute_errors

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from torch.autograd import Variable
from layers import disp_to_depth
import networks
from utils import download_model_if_doesnt_exist, readlines, normalize_image


# Config
gt_w = 960
gt_h = 540
model_name = 'vpl_3'
dset = 'vpl'
scene = 'vpl_packers_1'
clip = 'clip_0001'
epoch_num = 19
num_scales = 4
fps=22

# Set up network and load weights
if model_name.startswith('office_trim'):
    models_path = abspath('./logs/office')
else:
    models_path = abspath('./logs')
weights_path = join(models_path, model_name, 'models', 'weights_{}'.format(epoch_num))


# Load pretrained model
print('Loading... \nMODEL {} \nSCENE {} {}'.format(model_name, scene, clip))
encoder = networks.ResnetEncoder(18, False)
depth_decoder = networks.DepthDecoder(num_ch_enc=encoder.num_ch_enc, scales=range(num_scales))
encoder_path = join(weights_path, 'encoder.pth')
depth_decoder_path = join(weights_path, 'depth.pth')

# Load encoder network with weights. Verify encoder architecture
loaded_dict_enc = torch.load(encoder_path)
filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in encoder.state_dict()}
encoder.load_state_dict(filtered_dict_enc)

# Load depth decoder network with weights
loaded_dict = torch.load(depth_decoder_path)
depth_decoder.load_state_dict(loaded_dict)

# Set to eval mode on GPU
encoder.cuda()
depth_decoder.cuda()
encoder.eval()
depth_decoder.eval()


# Create dirs for model outputs
dest_path = join(abspath('./videos'), scene)
if not os.path.isdir(dest_path):
    os.makedirs(dest_path)


# Time the duration
start = time.time()

print('Evaluating...')
with torch.no_grad():

    src_path = join(abspath('../data'), dset, 'raw', scene, clip, 'color') 
    disp_frames = []
    rgb_frames = []
    model_h = loaded_dict_enc['height']
    model_w = loaded_dict_enc['width']

    for frame_name in sorted(os.listdir(src_path))[:100]:
        # Get frame and convert for the model
        frame_path = join(src_path, frame_name)
        frame_orig = np.array(pil.open(frame_path).convert('RGB'))
        frame = cv2.resize(frame_orig, (model_w, model_h))
        frame = transforms.ToTensor()(frame).unsqueeze(0).cuda()

        # Run inference and convert back
        output = depth_decoder(encoder(frame))
        pred_disp = normalize_image(output[("disp",0)].squeeze(0)).cpu().numpy()[0]
        pred_disp = cv2.resize(pred_disp, (gt_w, gt_h))

        # Save frames
        disp_frames.append(pred_disp)
        rgb_frames.append(frame_orig)

    # Make video
    vid_path = join(dest_path, '{}_{}.avi'.format(scene, clip))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(vid_path, fourcc, fps, (gt_w*2, gt_h))
    for i in range(len(rgb_frames)):
        dframe = cv2.cvtColor(disp_frames[i]*255, cv2.COLOR_GRAY2BGR)
        cframe = cv2.cvtColor(rgb_frames[i], cv2.COLOR_RGB2BGR)
        cat_frame = np.concatenate((cframe, dframe), 1)
        cat_frame = np.uint8(cat_frame)
        out.write(cat_frame)

    out.release()
    dur = round(time.time() - start, 4)
    print("\n-> Done! Time: {} sec".format(dur))


