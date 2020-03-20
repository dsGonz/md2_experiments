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
MIN_DEPTH = 0.5
MAX_DEPTH = 8.0
gt_w = 960
gt_h = 540
model_name = 'oa_clean_2'
dset = 'office'
scene = 'kitchen'
clip = 'clip_0002'
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

    c_src_path = join(abspath('../data'), dset, 'raw', scene, clip, 'color') 
    d_src_path = join(abspath('../data'), dset, 'raw', scene, clip, 'depth') 
    disp_frames = []
    rgb_frames = []
    gt_frames = []
    diff_frames = []
    model_h = loaded_dict_enc['height']
    model_w = loaded_dict_enc['width']

    for frame_name in sorted(os.listdir(c_src_path)):
        # Get frame and convert for the model
        frame_path = join(c_src_path, frame_name)
        frame_orig = np.array(pil.open(frame_path).convert('RGB'))
        frame = cv2.resize(frame_orig, (model_w, model_h))
        frame = transforms.ToTensor()(frame).unsqueeze(0).cuda()

        # Get the GT depth frame
        d_frame_name = '{:5}.npy'.format(frame_name[:-4])
        d_frame_path = join(d_src_path, d_frame_name)
        gt_depth = np.load(d_frame_path) / 1000

        # Run inference and convert back
        output = depth_decoder(encoder(frame))
        pred_disp = normalize_image(output[("disp",0)].squeeze(0)).cpu().numpy()[0]
        pred_disp = cv2.resize(pred_disp, (gt_w, gt_h))

        # Median Scaling
        mask = gt_depth > 0
        not_mask = gt_depth == 0
        pred_depth = np.nan_to_num(1 /pred_disp)
        pred_depth_reg = pred_depth[mask]
        gt_depth_reg = gt_depth[mask]
        ratio = np.median(gt_depth_reg) /np.median(pred_depth_reg)

        # Convert predicted depth
        pred_depth_reg *= ratio
        pred_depth_reg[pred_depth_reg < MIN_DEPTH] = MIN_DEPTH
        pred_depth_reg[pred_depth_reg > MAX_DEPTH] = MAX_DEPTH

        # Calculate depth difference
        dmap_gt = np.zeros(gt_depth.shape) 
        dmap_pred = np.zeros(gt_depth.shape) 
        
        dmap_gt[mask] = (gt_depth_reg - MIN_DEPTH) / (MAX_DEPTH - MIN_DEPTH)
        dmap_pred[mask] = (pred_depth_reg - MIN_DEPTH) / (MAX_DEPTH - MIN_DEPTH)

        dmap_diff = np.abs(dmap_gt - dmap_pred)

        # Save frames
        disp_frames.append(pred_disp)
        rgb_frames.append(frame_orig)
        gt_frames.append(dmap_gt)
        diff_frames.append(dmap_diff)

    # Make video
    vid_path = join(dest_path, '{}_{}.avi'.format(scene, clip))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(vid_path, fourcc, fps, (gt_w*2, gt_h*2))
    for i in range(len(rgb_frames)):
        # Convert frames to the same format
        rgb_frame = cv2.cvtColor(rgb_frames[i], cv2.COLOR_RGB2BGR)
        disp_frame = cv2.cvtColor(disp_frames[i]*255, cv2.COLOR_GRAY2BGR)
        gt_frame = cv2.cvtColor(np.uint8(gt_frames[i]*255), cv2.COLOR_GRAY2BGR)
        diff_frame = cv2.cvtColor(np.uint8(diff_frames[i]*255), cv2.COLOR_GRAY2BGR)

        # Label each frame 
        offset = (15,25)
        color = (0, 255, 0)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_size = 0.75
        cv2.putText(rgb_frame, 'RGB', offset, font, font_size, color, 2)
        cv2.putText(disp_frame, 'Predicted Disparity', offset, font, font_size, color, 2)
        cv2.putText(gt_frame, 'GT Depth', offset, font, font_size, color, 2)
        cv2.putText(diff_frame, 'Abs Diff.', offset, font, font_size, color, 2)

        # RGB + disparity frames on top, GT + abs diff frames on bottom
        top_frame = np.concatenate((rgb_frame, disp_frame), 1)
        bottom_frame = np.concatenate((gt_frame, diff_frame), 1)
        vid_frame = np.concatenate((top_frame, bottom_frame), 0)
        vid_frame = np.uint8(vid_frame)
        out.write(vid_frame)

    out.release()
    dur = round(time.time() - start, 4)
    print("\n-> Done! Time: {} sec".format(dur))


