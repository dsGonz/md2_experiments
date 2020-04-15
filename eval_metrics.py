# Import packages
from __future__ import absolute_import, division, print_function

import json
import cv2
import torch
import argparse
import os
import sys
import time
import datasets
import numpy as np
import PIL.Image as pil
from matplotlib import cm
from os.path import abspath, join
from torch.utils.data import DataLoader

import networks
from layers import disp_to_depth
from evaluate_depth import compute_errors
from utils import readlines

# CONSTANTS
epoch_num = 19
data_path_root= '/mnt/data0-nfs/shared-datasets/'


def colormap(tensor):
    viridis = cm.ScalarMappable(cmap='viridis')
    viridis.set_clim(vmin=0, vmax=1)
    viridis = viridis.get_cmap()

    image = np.uint8(viridis(tensor)*255)

    return image


def loadModel(model_name):
    # Set up network and load weights
    model_path = join(abspath('./logs'), model_name)
    opts_path = join(model_path, 'models/opt.json')
    weights_path = join(model_path, 'models', 'weights_{}'.format(epoch_num))

    # Load pretrained model options
    with open(opts_path, 'r') as f:
        opts = json.load(f)
    encoder = networks.ResnetEncoder(opts['num_layers'], False)
    depth_decoder = networks.DepthDecoder(num_ch_enc=encoder.num_ch_enc, scales=opts['scales'])
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

    return encoder, depth_decoder, opts


def load_data(args, opts):
    datasets_dict = {"kitti": datasets.KITTIRAWDataset,
                     "kitti_odom": datasets.KITTIOdomDataset,
                     "office": datasets.OfficeRAWDataset,
                     "presil": datasets.PreSILRAWDataset,
                     "gtav": datasets.GTAVRAWDataset}
    dataset = datasets_dict[args.dataset]

    data_path = join(data_path_root, args.data_path)
    filenames = readlines(join('splits', args.split, 'val_files.txt'))
    num_scales = len(opts['scales'])
    dataset = dataset(data_path, filenames, opts['height'], opts['width'], [0], num_scales, is_train=False, img_ext='.png')
    dataloader = DataLoader(dataset, 1, shuffle=False, num_workers=1, pin_memory=True, drop_last=False)

    return dataloader


def make_eval_dirs(dest_path, args):
    if not os.path.isdir(dest_path):
        os.makedirs(dest_path)
    if args.write_depths:
        if not os.path.isdir(join(dest_path, 'dense_depth')):
            os.makedirs(join(dest_path, 'dense_depth'))
            os.makedirs(join(dest_path, 'registered_depth'))
    if args.visualize:
        if not os.path.isdir(join(dest_path, 'viz')):
            os.makedirs(join(dest_path, 'viz'))

def print_log(f, statement):
    f.write(statement + '\n')
    print(statement)

    return

if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('-m', '--model', default='kitti_test_run')
    parser.add_argument('-n', '--name', default='temp')
    parser.add_argument('--dataset', default='kitti')
    parser.add_argument('--data_path', default='kitti_data')
    parser.add_argument('--split', default='eigen_zhou')
    parser.add_argument('--write_depths', action='store_true')
    parser.add_argument('--visualize', action='store_true')

    args = parser.parse_args()
    
    # Create dirs for model outputs
    dest_path = join(abspath('./outputs'), args.name)
    make_eval_dirs(dest_path, args)

    # Print and write all outputs to log file
    f = open(join(dest_path, 'metrics.log'), 'w')

    encoder, depth_decoder, opts = loadModel(args.model)
    print_log(f, "Loaded MODEL: {}".format(opts["model_name"]))

    print_log(f, "Loading data")
    dataloader = load_data(args, opts)
    print_log(f, 'Loaded {} validation images from SPLIT: {}  DATASET: {}'.format
          (len(dataloader), args.split, args.dataset))


    # Get predictions. Time the duration
    start = time.time()
    print_log(f, 'Evaluating...')

    # gt_depths = []
    # pred_disps = []
    # colors = []
    mind = opts['min_depth']
    maxd = opts['max_depth']

    errors = []
    error_table = {}
    ratios = []
    with torch.no_grad():
        fid = 0
        for data in dataloader:
            # Get GT depth and disp map predictions
            color = data[("color", 0, 0)].cuda()
            pred_disp = depth_decoder(encoder(color))[("disp", 0)]
            pred_disp = pred_disp.squeeze().detach().cpu().numpy()
            gt_depth = np.squeeze(data["depth_gt"].data.numpy())
            color = np.squeeze(data[("color", 0, 0)].cpu().numpy())

            # Append to list
            # colors.append(color)
            # pred_disps.append(pred_disp)
            # gt_depths.append(gt_depth)

            gt_h, gt_w = gt_depth.shape


            # Get errors
            pred_disp = cv2.resize(pred_disp, (gt_w, gt_h))
            _, pred_depth = disp_to_depth(pred_disp, mind, maxd)
    
            # mask = gt_depth > 0 and gt_depth <= 80
            # not_mask = gt_depth == 0 and gt_depth > 80
            mask = np.all([gt_depth > 0, gt_depth <= 80], axis=0)
            not_mask = np.any([gt_depth == 0, gt_depth > 80], axis=0)
    
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
    
            # Truncate depths
            pred_depth[pred_depth < mind] = mind
            pred_depth[pred_depth > maxd] = maxd
    
            # Save the error for each sample
            err = compute_errors(gt_depth, pred_depth)
            errors.append(err)
            error_table[fid] = err
    
            if args.write_depths:
                dmap_pred = 1 / pred_disp
                dmap_pred *= ratio
                np.save('outputs/{}/dense_depth/{:05}_pred.npy'.format(args.name, fid), dmap_pred)
                dmap_pred[not_mask] = 0
                np.save('outputs/{}/registered_depth/{:05}_pred_reg.npy'.format(args.name, fid), dmap_pred)
    
            if args.visualize:
                # Get Registered depth maps
                dmap_gt = np.zeros(mask.shape)
                dmap_pred = np.zeros(mask.shape)
    
                # Normalize depth maps to range 0-1
                dmap_gt[mask] = (gt_depth - mind) / (maxd - mind)
                dmap_pred[mask] = (pred_depth - mind) / (maxd - mind)
                dmap_diff = np.abs(dmap_gt - dmap_pred)
    
                # Get the pixels that are outside of the threshold and display them
                thresh = np.maximum((gt_depth / pred_depth), (pred_depth / gt_depth))
                a1_mask = np.where(thresh > 1.25)
                dwrong = np.zeros(gt_depth.shape)
                dwrong[a1_mask] = 1
                dmap_wrong = np.zeros(dmap_diff.shape)
                dmap_wrong[mask] = dwrong
    
                # Save depth visuals as images
                depth_maps = [dmap_gt, dmap_pred, dmap_diff, dmap_wrong]
                for i in range(len(depth_maps)):
                    # Turn into color map and remove invalid pixels
                    viz_dimage = colormap(depth_maps[i])
                    viz_dimage[not_mask] = 0
    
                    # Save each visual
                    img = pil.fromarray(viz_dimage).convert('RGB')
                    img.save('outputs/{}/viz/{:05}_dmap_{}.jpg'.format(args.name, fid, i))
    
                # Save disparity map
                img = pil.fromarray(colormap(pred_disp)).convert('RGB')
                img.save('outputs/{}/viz/{:05}_disp.jpg'.format(args.name, fid))
    
                # Save color image
                color = np.moveaxis(color, 0, -1)
                img = pil.fromarray(np.uint8(color*255))
                img = img.resize((gt_w, gt_h))
                img.save('outputs/{}/viz/{:05}_color.jpg'.format(args.name, fid))
    
            print("Processed {}".format(fid), end="\r")
            fid += 1

    # Average all errors
    mean_errors = np.array(errors).mean(0)
    ratios = np.array(ratios)

    # Save errors to a file
    error_path = join(dest_path, 'error_table.npy')
    mean_error_path = join(dest_path, 'mean_error.npy')
    np.save(error_path, error_table)
    np.save(mean_error_path, mean_errors)

    print_log(f, '\nEvaluation Metrics')
    print_log(f, "  " + ("{:>8} | " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
    print_log(f, ("&{: 8.3f}  " * 7).format(*mean_errors.tolist()) + "\\\\")

    # Get stats on the median scales
    print_log(f, '\nMedian Scale Metrics')
    print_log(f, "  " + ("{:>8} | " * 5).format("min", "max", "mean", "std", "median"))
    print_log(f, ("&{: 8.3f}  " * 5).format(np.min(ratios), np.max(ratios), np.mean(ratios), np.std(ratios), np.median(ratios)) + "\\\\")

    # Get duration
    dur = round(time.time() - start, 4)
    print_log(f, "\n-> Done! Time: {} sec".format(dur))
