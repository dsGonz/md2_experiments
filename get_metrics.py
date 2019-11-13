# get_metrics.py
#
# DESCRIPTION
# This script evaluates the val set of a given split and also has
# the ability to output visualizations such as the errors. Keep in 
# mind that printing these visualizations are very time-consuming.
# and predictions are named according to their a1 accuracy
#
# INPUTS (in Config)
# * MIN/MAX_DEPTH: depth range of the model and depth data
# * dset_type: the type of data you're evaluating on
# * split: the split that contains the validation set you're evaluating
# * model_name: the name of the model that's being evaluated. located in logs
# * epoch_num: weights from this epoch ID will be used
# * num_scales: The number of scales being used in the model
# * visualize: If true, stores visualizations of each sample in the outputs folder
# * write_depths: If true, saves the predicted depths as .npy files

# Import packages
from __future__ import absolute_import, division, print_function

import os
import cv2
import numpy as np
import PIL.Image as pil
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
import time
import datasets
from matplotlib import cm
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
MIN_DEPTH = 0.1
MAX_DEPTH = 100.0
dpath_root = '/mnt/data0-nfs/shared-datasets/'
dset_type = 'kitti_data'
split = 'eigen_zhou'
model_name = 'kitti_test_run'
epoch_num = 19
num_scales = 4
visualize = True
write_depths = False

# Set up network and load weights
models_path = abspath('./logs')
weights_path = join(models_path, model_name, 'models', 'weights_{}'.format(epoch_num))


# Load pretrained model
print('Loading... \nMODEL {} \nWEIGHTS {}'.format(model_name, epoch_num))
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


# Load validation data
print('Loading data...')
data_path = join(dpath_root, dset_type)
filenames = readlines(join('splits', split, 'val_files.txt'))
dataset = datasets.KITTIRAWDataset(data_path, filenames, loaded_dict_enc['height'], loaded_dict_enc['width'], [0], num_scales, is_train=False, img_ext='.png')
dataloader = DataLoader(dataset, 1, shuffle=False, num_workers=1, pin_memory=True, drop_last=False)
print('Loaded {} validation images from SPLIT: {}  DATASET: {}'.format(len(dataloader), split, dset_type))


# Create dirs for model outputs
dest_path = join(abspath('./outputs'), model_name)
if not os.path.isdir(dest_path):
    os.makedirs(dest_path)
    if write_depths:
        os.makedirs(join(dest_path, 'dense_depth'))
        os.makedirs(join(dest_path, 'registered_depth'))
    if visualize:
        os.makedirs(join(dest_path, 'viz'))

# Get predictions. Time the duration
start = time.time()

print('Evaluating...')
with torch.no_grad():

    errors = []
    error_table = {}
    ratios = []
    derrors = {90:[],80:[],70:[]}
    dmin = {90:100,80:100,70:100}
    dmax = {90:0,80:0,70:0}

    fid = 0
    for data in dataloader:
        # Get GT depth
        gt_depth = np.squeeze(data["depth_gt"].data.numpy())
        gt_h, gt_w = gt_depth.shape

        # Get disparity map predictions
        input_color = data[("color", 0, 0)].cuda()
        output = depth_decoder(encoder(input_color))
        pred_disp, _ = disp_to_depth(output[("disp",0)], MIN_DEPTH, MAX_DEPTH)
        pred_disp = pred_disp.squeeze().detach().cpu().numpy()

        # Get errors
        pred_disp = cv2.resize(pred_disp, (gt_w, gt_h))
        pred_depth = 1 / pred_disp 

        mask = gt_depth > 0
        not_mask = gt_depth == 0
        
        # Skip image if depth map has no registered pixels
        if np.sum(mask) == 0:
            fid += 1
            continue

        pred_depth = pred_depth[mask]
        gt_depth = gt_depth[mask]
        
        # Median scaling
        ratio = np.median(gt_depth) / np.median(pred_depth)
        ratios.append(ratio)
        pred_depth *= ratio

        pred_depth[pred_depth < MIN_DEPTH] = MIN_DEPTH
        pred_depth[pred_depth > MAX_DEPTH] = MAX_DEPTH
        
        # Save the error for each sample
        err = compute_errors(gt_depth, pred_depth)
        errors.append(err)
        error_table[fid] = err

        # Get depth stats
        # diff_depth = np.abs(gt_depth - pred_depth)
        # diff_avg = np.mean(diff_depth)
        # diff_min = np.min(diff_depth)
        # diff_max = np.max(diff_depth)
        # if err[4] >= 0.9:
        #     derrors[90].append(diff_avg)
        #     if diff_min < dmin[90]: 
        #         dmin[90] = diff_min
        #     if diff_max > dmax[90]:
        #         dmax[90] = diff_max
        # elif err[4] >= 0.8:
        #     derrors[80].append(diff_avg)
        #     if diff_min < dmin[80]: 
        #         dmin[80] = diff_min
        #     if diff_max > dmax[80]:
        #         dmax[80] = diff_max
        # elif err[4] >= 0.7:
        #     derrors[70].append(diff_avg)
        #     if diff_min < dmin[70]: 
        #         dmin[70] = diff_min
        #     if diff_max > dmax[70]:
        #         dmax[70] = diff_max

        if write_depths:
            dmap_pred = 1 / pred_disp
            dmap_pred *= ratio
            # dmap_pred[dmap_pred < MIN_DEPTH] = MIN_DEPTH
            # dmap_pred[dmap_pred > MAX_DEPTH] = MAX_DEPTH
            np.save('outputs/{}/dense_depth/{:05}_pred.npy'.format(model_name, fid), dmap_pred)
            dmap_pred[not_mask] = 0
            np.save('outputs/{}/registered_depth/{:05}_pred_reg.npy'.format(model_name, fid), dmap_pred)
            

        
        if visualize:
            # Get Registered depth maps
            dmap_gt = np.zeros(mask.shape)
            dmap_pred = np.zeros(mask.shape)

            # Normalize depth maps to range 0-1
            dmap_gt[mask] = (gt_depth - MIN_DEPTH) / (MAX_DEPTH - MIN_DEPTH)
            dmap_pred[mask] = (pred_depth - MIN_DEPTH) / (MAX_DEPTH - MIN_DEPTH)
            dmap_diff = np.abs(dmap_gt - dmap_pred)

            # Get the pixels that are outside of the threshold and display them
            thresh = np.maximum((gt_depth / pred_depth), (pred_depth / gt_depth))
            a1_mask = np.where(thresh > 1.25)
            dwrong = np.zeros(gt_depth.shape)
            dwrong[a1_mask] = dmap_diff[mask][a1_mask]
            dmap_wrong = np.zeros(dmap_diff.shape)
            dmap_wrong[mask] = dwrong
            
            # Save depths as imgs
            viz_dmap = np.uint8(cm.viridis(dmap_gt)*255)
            viz_dmap[not_mask] = 0
            img = pil.fromarray(viz_dmap).convert('RGB')
            img.save('outputs/{}/viz/{:05}_dmap_0.jpg'.format(model_name, fid))
            img = pil.fromarray(dmap_pred*255).convert('RGB')
            img.save('outputs/{}/viz/{:05}_dmap_1.jpg'.format(model_name, fid))
            img = pil.fromarray(dmap_diff*255).convert('RGB')
            img.save('outputs/{}/viz/{:05}_dmap_2.jpg'.format(model_name, fid))
            img = pil.fromarray(dmap_wrong*255).convert('RGB')
            img.save('outputs/{}/viz/{:05}_dmap_3.jpg'.format(model_name, fid))

            # Get normalized disp map and save as img
            norm_pred_disp = normalize_image(output[("disp",0)]).cpu()[:,0].numpy()[0]
            norm_pred_disp = cv2.resize(norm_pred_disp, (gt_w, gt_h))
            img = pil.fromarray(norm_pred_disp*255).convert('RGB')
            img.save('outputs/{}/viz/{:05}_disp.jpg'.format(model_name, fid))

            cimg = data[("color", 0, 0)].cpu().numpy()[0]
            cimg = np.moveaxis(cimg, 0, -1)
            cimg = cv2.resize(cimg, (gt_w, gt_h))
            cimg = np.uint8(cimg*255)
            img = pil.fromarray(np.uint8(cimg))#.convert('RGB')
            img.save('outputs/{}/viz/{:05}_color.jpg'.format(model_name, fid))

        fid += 1
        

    mean_errors = np.array(errors).mean(0)
    ratios = np.array(ratios)

    # Save errors to a file
    error_path = join(dest_path,'error_table.npy')
    np.save(error_path, error_table)

    print('\nEvaluation Metrics')
    print("  " + ("{:>8} | " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
    print(("&{: 8.3f}  " * 7).format(*mean_errors.tolist()) + "\\\\")

    # Get stats on the median scales
    print('\nMedian Scale Metrics')
    print("  " + ("{:>8} | " * 5).format("min", "max", "mean", "std", "median"))
    print(("&{: 8.3f}  " * 5).format(np.min(ratios), np.max(ratios), np.mean(ratios), np.std(ratios), np.median(ratios)) + "\\\\")

    # Get stats on the depth errors 
    # print('\n0.9 Depth Error Metrics')
    # print("  " + ("{:>8} | " * 3).format("min", "max", "mean"))
    # print(("&{: 8.3f}  " * 3).format(dmin[90], dmax[90], np.mean(derrors[90])) + "\\\\")

    # print('\n0.8 Depth Error Metrics')
    # print("  " + ("{:>8} | " * 3).format("min", "max", "mean"))
    # print(("&{: 8.3f}  " * 3).format(dmin[80], dmax[80], np.mean(derrors[80])) + "\\\\")

    # print('\n0.7 Depth Error Metrics')
    # print("  " + ("{:>8} | " * 3).format("min", "max", "mean"))
    # print(("&{: 8.3f}  " * 3).format(dmin[70], dmax[70], np.mean(derrors[70])) + "\\\\")
    dur = round(time.time() - start, 4)
    print("\n-> Done! Time: {} sec".format(dur))


