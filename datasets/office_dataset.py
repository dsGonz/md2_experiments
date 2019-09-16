# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import os
import skimage.transform
import numpy as np
import PIL.Image as pil

from kitti_utils import generate_depth_map
from .mono_dataset import MonoDataset


class OfficeDataset(MonoDataset):
    """Superclass for different types of Office dataset loaders
    """
    def __init__(self, *args, **kwargs):
        super(OfficeDataset, self).__init__(*args, **kwargs)

        self.K = np.array([[526.82, 0, 461.87, 0],
                           [0, 527.67, 277.13, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=np.float32)

        self.full_res_shape = (960, 540)
        img_w, img_h = self.full_res_shape

        self.K[0,:] /= img_w
        self.K[1,:] /= img_h

    def check_depth(self):
        line = self.filenames[0].split()
        scene_name = line[0]
        frame_index = int(line[1])

        depth_filename = os.path.join(
                self.data_path,
                scene_name.replace('color', 'depth'),
                '{:0>5}.npy'.format(frame_index))

        return os.path.isfile(depth_filename)

    def get_color(self, folder, frame_index, side, do_flip):
        color = self.loader(self.get_image_path(folder, frame_index, side))

        if do_flip:
            color = color.transpose(pil.FLIP_LEFT_RIGHT)

        return color


class OfficeRAWDataset(OfficeDataset):
    """Office dataset which loads the original velodyne depth maps for ground truth
    """
    def __init__(self, *args, **kwargs):
        super(OfficeRAWDataset, self).__init__(*args, **kwargs)

    def get_image_path(self, folder, frame_index, side):
        f_str = "{:010d}{}".format(frame_index, self.img_ext)
        image_path = os.path.join(
                # self.data_path, folder, '{:0>5}.png'.format(frame_index))
                self.data_path, folder, '{:0>5}{}'.format(frame_index, self.img_ext))
        return image_path

    def get_depth(self, folder, frame_index, side, do_flip):
        depth_filename = os.path.join(
              self.data_path,
              folder.replace('color', 'depth'),
              '{:0>5}.npy'.format(frame_index))

        depth_gt = np.load(depth_filename)

        if do_flip:
            depth_gt = np.fliplr(depth_gt)

        return depth_gt



class OfficeDepthDataset(OfficeDataset):
    """Office dataset which uses the updated ground truth depth maps
    """
    def __init__(self, *args, **kwargs):
        super(OfficeDepthDataset, self).__init__(*args, **kwargs)

    def get_image_path(self, folder, frame_index, side):
        f_str = "{:010d}{}".format(frame_index, self.img_ext)
        image_path = os.path.join(
            self.data_path,
            folder,
            "image_0{}/data".format(self.side_map[side]),
            f_str)
        return image_path

    def get_depth(self, folder, frame_index, side, do_flip):
        f_str = "{:010d}.png".format(frame_index)
        depth_path = os.path.join(
            self.data_path,
            folder,
            "proj_depth/groundtruth/image_0{}".format(self.side_map[side]),
            f_str)

        depth_gt = pil.open(depth_path)
        depth_gt = depth_gt.resize(self.full_res_shape, pil.NEAREST)
        depth_gt = np.array(depth_gt).astype(np.float32) / 256

        if do_flip:
            depth_gt = np.fliplr(depth_gt)

        return depth_gt
