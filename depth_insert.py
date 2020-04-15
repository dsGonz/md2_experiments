import argparse
import os
import sys
import torch
import shutil
import json
import numpy as np
from os.path import join
from PIL import Image, ImageDraw
from PIL.ImageColor import getcolor
from get_depth import loadModel, pilToTensor, normalize_disp, tensorToPil
from layers import disp_to_depth

# COLORS
RED = (255, 0, 0, 255)
GREEN = (0, 255, 0, 255)

# CONSTANTS
mask_path = "depth_samples/insert/masks"
COLOR = RED
default_L = 128


def save_config(args):
    """Save options to disk so we know what we ran this experiment with
    """
    dest_path = join(args.dest, args.name)
    if not os.path.exists(dest_path):
        os.makedirs(dest_path)
    to_save = args.__dict__.copy()

    with open(os.path.join(dest_path, 'config.json'), 'w') as f:
        json.dump(to_save, f, indent=2)


def get_pos(width, height, num_x, num_y, offset):
    center_x = np.linspace(offset, width - offset, num_x)
    center_y = np.linspace(offset, height - offset, num_y)

    center_x = np.rint(center_x).astype(int)
    center_y = np.rint(center_y).astype(int)

    return center_x, center_y


def get_binary_mask(obj_img, bg_img, bbox, new_w, new_h):

    bg_mask = Image.new('RGBA', bg_img.size, getcolor('rgba(0,0,0,0)', 'RGBA'))
    bg_mask.paste(obj_img, bbox, obj_img)
    bg_mask = bg_mask.resize((new_w, new_h))

    mask = np.array(bg_mask)[:, :, -1]/255
    mask = np.rint(mask).astype(int)

    return mask


def get_object_depth(depth_map, object_mask):
    depth_map = depth_map.cpu()[:, 0].numpy()[0]
    mask = object_mask == 1
    object_depths = depth_map[mask]

    depth = np.mean(object_depths)

    return depth


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('-m', '--model', default='kitti_test_run')
    parser.add_argument('-i', '--input', default=None)
    parser.add_argument('-d', '--dest', default="depth_samples/insert")
    parser.add_argument('-n', '--name', default="shapes")
    parser.add_argument('--mask', default=None)
    parser.add_argument('--num_sizes', type=int, default=10)
    parser.add_argument('--min_L', type=int, default=16)
    parser.add_argument('--max_L', type=int, default=128)
    parser.add_argument('--offset', type=int, default=60)
    parser.add_argument('--num_x', type=int, default=15)
    parser.add_argument('--num_y', default=3)
    parser.add_argument('--ss_ratio', default=29.288708)
    parser.add_argument('--source_path', default=os.path.expanduser("./depth_samples/inputs"))
    parser.add_argument('--multi_size', dest='multi_size', action='store_true')

    args = parser.parse_args()

    # Make folders. Clear contents from a previous run
    color_path = join(args.dest, args.name, 'inputs')
    disp_path = join(args.dest, args.name, 'outputs')
    if os.path.exists(color_path):
        shutil.rmtree(color_path)
    if os.path.exists(disp_path):
        shutil.rmtree(disp_path)
    os.makedirs(color_path)
    os.makedirs(disp_path)

    # Save config
    save_config(args)

    # Load background image and model parameters
    img = Image.open(join(args.source_path, args.input))
    encoder, depth_decoder, opts = loadModel(args.model)
    in_w, in_h = (opts['width'], opts['height'])
    min_d, max_d = (opts['min_depth'], opts['max_depth'])

    # Predict depth of original background image
    orig_input = pilToTensor(img.resize((in_w, in_h)))
    with torch.no_grad():
        orig_disp = depth_decoder(encoder(orig_input))[("disp", 0)]
    _, orig_depth = disp_to_depth(orig_disp, min_d, max_d)
    orig_depth *= args.ss_ratio

    if args.multi_size:
        lengths = np.linspace(args.min_L, args.max_L, args.num_sizes)
        lengths = np.rint(lengths).astype(int).tolist()
    else:
        lengths = [default_L]

    depth_table = np.zeros((args.num_y, args.num_x, len(lengths), 2))

    w, h = img.size
    cx, cy = get_pos(w, h, args.num_x, args.num_y, args.offset)
    for r in range(args.num_y):
        for c in range(args.num_x):
            for i in range(len(lengths)):
                L = lengths[i]
                img = Image.open(join(args.source_path, args.input))
                hL = L/2
                bbox = (int(cx[c] - hL), int(cy[r] - hL), int(cx[c] + hL), int(cy[r] + hL))

                # Place mask or shape depending on args
                if args.mask is not None:
                    mask = Image.open(join(mask_path, args.mask))
                    mask = mask.resize((L, L))
                    img.paste(mask, bbox, mask)
                else:
                    mask = Image.new('RGBA', (L, L), getcolor('rgba(0,0,0,0)', 'RGBA'))
                    draw = ImageDraw.Draw(mask)
                    draw.ellipse((0,0,L,L), fill=COLOR)
                    img.paste(mask, bbox, mask)

                # Get binary mask of object for depth calculations
                bmask = get_binary_mask(mask, img, bbox, in_w, in_h)

                # Save the color image
                img.save(join(color_path, 'color_{}_{}_{}.png'.format(r, c, i)))

                # Convert to tensor
                orig_shape = img.size
                img = img.resize((opts['width'], opts['height']))
                input = pilToTensor(img)

                with torch.no_grad():
                    disp = depth_decoder(encoder(input))[("disp", 0)]

                # Get depth of the object
                _, depth = disp_to_depth(disp, min_d, max_d)
                depth *= args.ss_ratio
                obj_depth = get_object_depth(depth, bmask)
                bg_depth = get_object_depth(orig_depth, bmask)
                depth_table[r][c][i][0] = bg_depth
                depth_table[r][c][i][1] = obj_depth

                # Reformat tensor to image for visualization
                disp_img = normalize_disp(disp)
                img = tensorToPil(disp_img)
                img = img.resize(orig_shape)

                # Save image
                img.save(join(disp_path, 'disp_{}_{}_{}.png'.format(r, c, i)))

    np.save(join(args.dest, args.name, 'depths.npy'), depth_table)
