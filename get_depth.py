# Import packages
from __future__ import absolute_import, division, print_function

import json
import torch
import argparse
import os
import numpy as np
import PIL.Image as pil
from matplotlib import cm
from os.path import abspath, join

import networks
from utils import normalize_image

# CONSTANTS
epoch_num = 19
source_path = "depth_samples/inputs"
dest_path = "depth_samples/outputs"


def pilToTensor(image):
    # Convert to NumPy and move RGB channel to front
    tensor = np.array(image)
    tensor = np.moveaxis(tensor, 2, 0)
    # Normalize values and append new axis for batch size
    tensor = np.expand_dims(tensor, axis=0)/255
    # Convert to CUDA tensor in float32
    tensor = torch.as_tensor(tensor, dtype=torch.float32).cuda()

    return tensor


def normalize_disp(tensor):
    image = normalize_image(tensor).cpu()[:, 0].numpy()[0]
    image = np.uint8(cm.viridis(image)*255)

    return image


def tensorToPil(tensor):
    return pil.fromarray(tensor).convert('RGB')


def loadModel(model_name):
    # Set up network and load weights
    model_path = join(abspath('./logs'), model_name)
    opts_path = join(model_path, 'models/opt.json')
    weights_path = join(model_path, 'models', 'weights_{}'.format(epoch_num))

    # Load pretrained model options
    with open(opts_path, 'rb') as f:
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


if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('-m', '--model', default='kitti_test_run', help='Pretrained model to use')
    parser.add_argument('-i', '--input', default=None, help='Image to use. If None, then generate depths for all images in source_path')
    args = parser.parse_args()

    encoder, depth_decoder, opts = loadModel(args.model)

    # Get image paths according to input
    if args.input is None:
        image_names = os.listdir(source_path)
    else:
        image_names = [args.input]
    image_paths = [join(source_path, x) for x in image_names]

    for i in range(len(image_names)):
        # Retrieve Image
        image = pil.open(image_paths[i]).convert('RGB')

        # Reformat image to tensor for network
        orig_shape = image.size
        image = image.resize((opts['width'], opts['height']))
        input = pilToTensor(image)

        with torch.no_grad():
            output = depth_decoder(encoder(input))

        # Reformat tensor to image for visualization
        disp = normalize_disp(output[("disp", 0)])
        img = tensorToPil(disp)
        img = img.resize(orig_shape)
        img.save(join(dest_path, image_names[i]))
