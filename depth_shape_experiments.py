import argparse
import os
import torch
import numpy as np
from os.path import join
from PIL import Image, ImageDraw
from get_depth import loadModel, pilToTensor, normalize_disp, tensorToPil

# CONSTANTS
source_path = "depth_samples/inputs"
RED = (255, 0, 0)
GREEN = (0, 255, 0)
R = 32
offset = 60
num_x = 9
num_y = 3

def get_pos(width, height, num_x, num_y):
    center_x = np.linspace(offset, width - offset, num_x)
    center_y = np.linspace(offset, height - offset, num_y)

    center_x = np.rint(center_x).astype(int)
    center_y = np.rint(center_y).astype(int)

    return center_x, center_y

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('-m', '--model', default='kitti_test_run')
    parser.add_argument('-i', '--input', default=None)
    parser.add_argument('-d', '--dest', default="depth_samples")
    parser.add_argument('-n', '--name', default="shapes")
    args = parser.parse_args()

    encoder, depth_decoder, opts = loadModel(args.model)

    img = Image.open(join(source_path, args.input))
    w, h = img.size
    cx, cy = get_pos(w, h, num_x, num_y)
    for r in range(num_y):
        for c in range(num_x):
            # Draw the ellipse for the given position
            img = Image.open(join(source_path, args.input))
            draw = ImageDraw.Draw(img)
            draw.ellipse((cx[c] - R, cy[r] - R, cx[c] + R, cy[r] + R), fill=GREEN)

            # Save the color image 
            color_path = join(args.dest, args.name, 'inputs')
            if not os.path.exists(color_path):
                os.makedirs(color_path)
            img.save(join(color_path, 'color_{}_{}.png'.format(r, c)))

            # Convert to tensor
            orig_shape = img.size
            img = img.resize((opts['width'], opts['height']))
            input = pilToTensor(img)

            with torch.no_grad():
                output = depth_decoder(encoder(input))

            # Reformat tensor to image for visualization
            disp = normalize_disp(output[("disp", 0)])
            img = tensorToPil(disp)
            img = img.resize(orig_shape)

            # Save image
            disp_path = join(args.dest, args.name, 'outputs')
            if not os.path.exists(disp_path):
                os.makedirs(disp_path)
            img.save(join(disp_path, 'disp_{}_{}.png'.format(r, c)))


