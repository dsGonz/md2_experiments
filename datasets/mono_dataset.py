# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import os
import random
import numpy as np
import copy
from PIL import Image  # using pillow-simd for increased speed
import math

import torch
import torch.utils.data as data
from torchvision import transforms
import torchvision.transforms.functional as F

def pil_loader(path):
    # open path as file to avoid ResourceWarning
    # (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


# Construct a rotation matrix in the same way as PIL
def R_matrix(deg, center=None):
    R = np.identity(4, dtype=np.float32)

    # If no rotation, return identity
    if deg == 0:
        return R

    # Convert to radians
    rad = -math.radians(deg)

    # Fill in entries of R matrix
    R[[0,1],[0,1]] = round(math.cos(rad), 15)
    R[0,1] = round(-math.sin(rad), 15)
    R[1,0] = round(math.sin(rad), 15)

    # If center is true, Construct R so it rotates pixels around the center of img
    if center is not None:
        T = np.identity(4)
        T[0,2] = center[0]
        T[1,2] = center[1]
        Tp = T.copy()
        Tp[0,2] *= -1
        Tp[1,2] *= -1

        R = T @ R @ Tp

    return R

# Construct a scale matrix
def scale_matrix(x_scale, y_scale, center=None):
    S = np.identity(4)

    if x_scale == 1 and y_scale == 1:
        return S

    # Fill in entries of R matrix
    S[0,0] = x_scale
    S[1,1] = y_scale

    # If center is true, Construct R so it rotates pixels around the center of img
    if center is not None:
        T = np.identity(4)
        T[0,2] = center[0]
        T[1,2] = center[1]
        Tp = T.copy()
        Tp[0,2] *= -1
        Tp[1,2] *= -1

        S = T @ S @ Tp

    return S

def resize_crop(img, center, h, w, x_scale, y_scale):
    hs = int(h/y_scale)
    ws = int(w/x_scale)
    i = int((h - hs)/2)
    j = int((w - ws)/2)
    resized_img = F.resized_crop(img, i, j, hs, ws, center)

    return resized_img


class MonoDataset(data.Dataset):
    """Superclass for monocular dataloaders

    Args:
        data_path
        filenames
        height
        width
        frame_idxs
        num_scales
        is_train
        img_ext
    """
    def __init__(self,
                 data_path,
                 filenames,
                 height,
                 width,
                 frame_idxs,
                 num_scales,
                 is_train=False,
                 img_ext='.jpg'):
        super(MonoDataset, self).__init__()

        self.data_path = data_path
        self.filenames = filenames
        self.height = height
        self.width = width
        self.num_scales = num_scales
        self.interp = Image.ANTIALIAS

        self.frame_idxs = frame_idxs

        self.is_train = is_train
        self.img_ext = img_ext

        self.loader = pil_loader
        self.to_tensor = transforms.ToTensor()

        # We need to specify augmentations differently in newer versions of torchvision.
        # We first try the newer tuple version; if this fails we fall back to scalars
        try:
            self.brightness = (0.8, 1.2)
            self.contrast = (0.8, 1.2)
            self.saturation = (0.8, 1.2)
            self.hue = (-0.1, 0.1)
            transforms.ColorJitter.get_params(
                self.brightness, self.contrast, self.saturation, self.hue)

        except TypeError:
            self.brightness = 0.2
            self.contrast = 0.2
            self.saturation = 0.2
            self.hue = 0.1

        self.rot_degrees = (-5, 5)
        self.resize_range = (0.95, 1.05)

        self.resize = {}
        for i in range(self.num_scales):
            s = 2 ** i
            self.resize[i] = transforms.Resize((self.height // s, self.width // s),
                                               interpolation=self.interp)

        self.load_depth = self.check_depth()

    def preprocess(self, inputs, color_aug, rot, resize):
        """Resize colour images to the required scales and augment if required

        We create the color_aug object in advance and apply the same augmentation to all
        images in this item. This ensures that all images input to the pose network receive the
        same augmentation.
        """
        for k in list(inputs):
            frame = inputs[k]
            if "color" in k:
                n, im, i = k
                for i in range(self.num_scales):
                    inputs[(n, im, i)] = self.resize[i](inputs[(n, im, i - 1)])

        for k in list(inputs):
            f = inputs[k]
            if "color" in k:
                n, im, i = k
                inputs[(n, im, i)] = self.to_tensor(f)
                center_x = self.K[0,2]*(self.width // (2 ** i))
                center_y = self.K[1,2]*(self.height // (2 ** i))
                center = (center_x, center_y)
                inputs[(n + "_aug", im, i)] = self.to_tensor(color_aug(resize(rot(f,center), center)))

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        """Returns a single training item from the dataset as a dictionary.

        Values correspond to torch tensors.
        Keys in the dictionary are either strings or tuples:

            ("color", <frame_id>, <scale>)          for raw colour images,
            ("color_aug", <frame_id>, <scale>)      for augmented colour images,
            ("K", scale) or ("inv_K", scale)        for camera intrinsics,
            "stereo_T"                              for camera extrinsics, and
            "depth_gt"                              for ground truth depth maps.

        <frame_id> is either:
            an integer (e.g. 0, -1, or 1) representing the temporal step relative to 'index',
        or
            "s" for the opposite image in the stereo pair.

        <scale> is an integer representing the scale of the image relative to the fullsize image:
            -1      images at native resolution as loaded from disk
            0       images resized to (self.width,      self.height     )
            1       images resized to (self.width // 2, self.height // 2)
            2       images resized to (self.width // 4, self.height // 4)
            3       images resized to (self.width // 8, self.height // 8)
        """
        inputs = {}

        do_color_aug = self.is_train and random.random() > 0.5
        do_flip = self.is_train and random.random() > 0.5
        do_rot = self.is_train and random.random() > 0.5
        # do_resize = self.is_train and random.random() > 0.5
        do_resize = False

        line = self.filenames[index].split()
        folder = line[0]

        if len(line) == 3:
            frame_index = int(line[1])
        else:
            frame_index = 0

        if len(line) == 3:
            side = line[2]
        else:
            side = None

        for i in self.frame_idxs:
            if i == "s":
                other_side = {"r": "l", "l": "r"}[side]
                inputs[("color", i, -1)] = self.get_color(folder, frame_index, other_side, do_flip)
            else:
                inputs[("color", i, -1)] = self.get_color(folder, frame_index + i, side, do_flip)

        if do_rot:
            deg = random.uniform(self.rot_degrees[0], self.rot_degrees[1])
            rot = (lambda img, center: F.rotate(img, deg, center=center))
        else:
            deg = 0
            rot = (lambda img, center: img)

        if do_resize:
            x_scale = random.uniform(self.resize_range[0], self.resize_range[1])
            y_scale = random.uniform(self.resize_range[0], self.resize_range[1])
            resize = (lambda img, center: resize_crop(img, center, self.height, self.width, x_scale, y_scale))
        else:
            x_scale = 1
            y_scale = 1
            resize = (lambda img, center: img)

        # adjusting intrinsics to match each scale in the pyramid
        for scale in range(self.num_scales):
            K = self.K.copy()

            K[0, :] *= self.width // (2 ** scale)
            K[1, :] *= self.height // (2 ** scale)

            # Apply rotation
            center_pt = (K[0,2], K[1,2])
            R = R_matrix(deg, center_pt)
            S = scale_matrix(x_scale, y_scale, center_pt)
            K = np.float32(S @ R @ K)

            inv_K = np.linalg.pinv(K)

            inputs[("K", scale)] = torch.from_numpy(K)
            inputs[("inv_K", scale)] = torch.from_numpy(inv_K)

        if do_color_aug:
            color_aug = transforms.ColorJitter.get_params(
                self.brightness, self.contrast, self.saturation, self.hue)
        else:
            color_aug = (lambda x: x)

        self.preprocess(inputs, color_aug, rot, resize)

        for i in self.frame_idxs:
            del inputs[("color", i, -1)]
            del inputs[("color_aug", i, -1)]

        if self.load_depth:
            depth_gt = self.get_depth(folder, frame_index, side, do_flip)
            inputs["depth_gt"] = np.expand_dims(depth_gt, 0)
            inputs["depth_gt"] = torch.from_numpy(inputs["depth_gt"].astype(np.float32))

        if "s" in self.frame_idxs:
            stereo_T = np.eye(4, dtype=np.float32)
            baseline_sign = -1 if do_flip else 1
            side_sign = -1 if side == "l" else 1
            stereo_T[0, 3] = side_sign * baseline_sign * 0.1

            inputs["stereo_T"] = torch.from_numpy(stereo_T)

        return inputs

    def get_color(self, folder, frame_index, side, do_flip):
        raise NotImplementedError

    def check_depth(self):
        raise NotImplementedError

    def get_depth(self, folder, frame_index, side, do_flip):
        raise NotImplementedError
