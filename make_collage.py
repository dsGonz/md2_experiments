# Import packages
from __future__ import absolute_import, division, print_function

import os
import cv2
import numpy as np
import PIL.Image as pil
import matplotlib.pyplot as plt
import time
from os.path import abspath, join
# from evaluate_depth import compute_errors

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from torch.autograd import Variable
# from layers import disp_to_depth
import networks
from utils import download_model_if_doesnt_exist, readlines, normalize_image


# Config
gt_w = 960
gt_h = 540
en_w = 480
en_h = 256
model_name = 'oa_clean_2'
clip = 'clip_0002'
epoch_num = 19
num_scales = 4
fps=30
num_frames = 5000

# Videos - Format: [model_name, dset, scene, clip]
vid_list = []
vid_list.append(['oa_clean_2', 'office', 'misc', 'clip_0004'])
vid_list.append(['oa_clean_2', 'office', 'misc', 'clip_0002'])
vid_list.append(['vpl_static_3', 'vpl', 'vpl_2', 'clip_0002'])
vid_list.append(['oa_clean_2', 'office', 'kitchen', 'clip_0003'])
vid_list.append(['oa_clean_2', 'turtle/office', 'run2', 'clip_0003'])
vid_list.append(['vpl_static_3', 'vpl', 'vpl_2', 'clip_0000'])
vid_list.append(['oa_clean_2', 'office', 'kitchen', 'clip_0004'])
vid_list.append(['oa_clean_2', 'turtle/office', 'run5', 'clip_0001'])


def load_model(model_name):
    # Set up network and load weights
    if model_name.startswith('office_trim'):
        models_path = abspath('./logs/office')
    else:
        models_path = abspath('./logs')
    weights_path = join(models_path, model_name, 'models', 'weights_{}'.format(epoch_num))
    
    
    # Load pretrained model
    print('Loading... \nMODEL {}'.format(model_name))
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

    return depth_decoder, encoder


# Create dirs for model outputs
dest_path = abspath('./videos')
if not os.path.isdir(dest_path):
    os.makedirs(dest_path)


# Time the duration
start = time.time()

print('Evaluating...')
with torch.no_grad():

    collage_vids = []
    for vid_info in vid_list:
        vid_frames = []
        model_name, dset, scene, clip = vid_info
        # dset = vid_info[1]
        # scene = vid_info[2]
        # clip = vid_info[3]

        depth_decoder, encoder = load_model(model_name)
        src_path = join(abspath('../data'), dset, 'raw', scene, clip, 'color') 
        print('Estimating depth for SCENE {} CLIP {}.'.format(scene, clip))

        for frame_name in sorted(os.listdir(src_path)):
            # Get frame and convert for the model
            frame_path = join(src_path, frame_name)
            frame_orig = np.array(pil.open(frame_path).convert('RGB'))
            frame = cv2.resize(frame_orig, (en_w, en_h))
            frame = transforms.ToTensor()(frame).unsqueeze(0).cuda()

            # Run inference and convert back
            output = depth_decoder(encoder(frame))
            pred_disp = normalize_image(output[("disp",0)].squeeze(0)).cpu().numpy()[0]
            pred_disp = cv2.resize(pred_disp, (gt_w, gt_h))

            # Convert frames for video. Concatenate
            dframe = cv2.cvtColor(pred_disp*255, cv2.COLOR_GRAY2BGR)
            cframe = cv2.cvtColor(frame_orig, cv2.COLOR_RGB2BGR)
            dframe = cv2.resize(dframe, (480, 270))
            cframe = cv2.resize(cframe, (480, 270))
            cat_frame = np.concatenate((cframe, dframe), 1)
            cat_frame = np.uint8(cat_frame)

            # Save frame
            vid_frames.append(cat_frame)

        collage_vids.append(vid_frames)

    # Save frames 
    # np.save('cvids.npy', collage_vids)

    # collage_vids = np.load('cvids.npy', allow_pickle=True)
    # Interpolate the low fps videos
    low_fps_ids = [2,5]
    for i in range(len(collage_vids)):
        # Skip if not in list
        if i not in low_fps_ids:
            continue

        # Low fps vids are ~22fps. U/D sample by 3/2 to get ~30fps
        U = 3
        D = 2
        vid = np.stack(collage_vids[i])
        print(np.shape(vid))
        vid = np.repeat(vid, U, 0)
        print(np.shape(vid))
        vid = vid[::D]
        print(np.shape(vid))
        vid_frames = [np.squeeze(x) for x in np.split(vid,len(vid))]
        collage_vids[i] = vid_frames
        print(np.shape(collage_vids[i]))


    # Loop all vids to the desired length
    loop_vids = []
    for vid_frames in collage_vids:
        factor = int(num_frames / len(vid_frames))+1

        # Repeat video to meet desired frame length then cut off extra
        loop_frames = factor * vid_frames
        loop_frames = loop_frames[:num_frames]
        loop_vids.append(loop_frames)

    # Concatenate all frames into a collage frame
    for vid in loop_vids:
        print(len(vid))
    collage_frames = []
    for i in range(num_frames):
        row_frames = []
        for j in range(4):
            rframe = np.concatenate((loop_vids[2*j][i], loop_vids[2*j + 1][i]), 1)
            row_frames.append(rframe)
        collage_frame = np.concatenate(row_frames,0)
        collage_frames.append(collage_frame)

    # Write frames to video
    vid_path = join(dest_path, 'collage_vid.avi')
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(vid_path, fourcc, fps, (1920, 1080))
    for frame in collage_frames:
        out.write(frame)

    out.release()
    dur = round(time.time() - start, 4)
    print("\n-> Done! Time: {} sec".format(dur))


