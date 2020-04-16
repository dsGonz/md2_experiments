# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import os
import math
import skimage.transform
import numpy as np
import PIL.Image as pil

from kitti_utils import generate_depth_map
from .mono_dataset import MonoDataset


class GTAVDataset(MonoDataset):
    """Superclass for different types of GTAV dataset loaders
    """
    def __init__(self, *args, **kwargs):
        super(GTAVDataset, self).__init__(*args, **kwargs)

        self.K = np.array([[640, 0, 622, 0],
                           [0, 640, 187.5, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=np.float32)

        self.full_res_shape = (1242, 375)
        img_w, img_h = self.full_res_shape

        self.K[0,:] /= img_w
        self.K[1,:] /= img_h

    def check_depth(self):
        line = self.filenames[0].split()
        scene_name = line[0]
        frame_index = int(line[1])

        depth_filename = os.path.join(
                self.data_path,
                'crop',
                scene_name,
                'depth',
                '{:06d}.npy'.format(frame_index))

        return os.path.isfile(depth_filename)

    def get_color(self, folder, frame_index, side, do_flip):
        color = self.loader(self.get_image_path(folder, frame_index, side))

        if do_flip:
            color = color.transpose(pil.FLIP_LEFT_RIGHT)

        return color


class GTAVRAWDataset(GTAVDataset):
    """GTAV dataset which loads the original velodyne depth maps for ground truth
    """
    def __init__(self, *args, **kwargs):
        super(GTAVRAWDataset, self).__init__(*args, **kwargs)

    def get_image_path(self, folder, frame_index, side):
        f_str = "{:06d}{}".format(frame_index, self.img_ext)
        image_path = os.path.join(
                self.data_path,
                'crop',
                folder,
                'image_2',
                f_str)
        return image_path

    def get_depth(self, folder, frame_index, side, do_flip):
        f_str = "{:06d}.npy".format(frame_index)
        depth_filename = os.path.join(
              self.data_path,
              'crop',
              folder,
              'depth',
              f_str)

        depth_gt = np.load(depth_filename)
        depth_gt[depth_gt > 80] = 0 # Max valid depth of LiDAR is 80

        if do_flip:
            depth_gt = np.fliplr(depth_gt)

        return depth_gt

    def ndcToDepth(self, depth_filename):
        img_w, img_h = self.full_res_shape
        nc_z = 0.15
        fc_z = 600
        fov_v = 59 #degrees
        nc_h = 2 * nc_z * math.tan(fov_v / 2.0)
        nc_w = 1242 / 375.0 * nc_h

        depth = np.zeros((img_h,img_w))


        fd = open(depth_filename, 'rb')
        f = np.fromfile(fd, dtype=np.float32, count=img_h*img_w)
        ndc = f.reshape((img_h, img_w))

        # Vectorized approach 
        nc_x = np.abs(2 * np.arange(img_w) / (img_w - 1) - 1) * nc_w / 2.0
        nc_y = np.abs(2 * np.arange(img_h) / (img_h - 1) - 1) * nc_h / 2.0

        nc_xx = np.tile(nc_x, (img_h, 1))
        nc_yy = np.tile(nc_y, (img_w, 1)).T

        d_nc = np.sqrt(np.power(nc_xx,2) + np.power(nc_yy,2) + np.power(nc_z,2))
        depth = d_nc / (ndc + (nc_z * d_nc / (2 * fc_z)))

        # Original 
        # for j in range(0,img_h):
            # for i in range(0,img_w):
                # nc_x = abs(((2 * i) / (img_w - 1.0)) - 1) * nc_w / 2.0
                # nc_y = abs(((2 * j) / (img_h - 1.0)) - 1) * nc_h / 2.0
    
                # d_nc = math.sqrt(pow(nc_x,2) + pow(nc_y,2) + pow(nc_z,2))
                # depth[j,i] = d_nc / (ndc[j,i] + (nc_z * d_nc / (2 * fc_z)))
                # if ndc[j,i] == 0.0:
                    # depth[j,i] = fc_z
    
        return depth



# This aint right. Uses dense GT depths in the form of .PNG instead of .BIN
class GTAVDepthDataset(GTAVDataset):
    """GTAV dataset which uses the updated ground truth depth maps
    """
    def __init__(self, *args, **kwargs):
        super(GTAVDepthDataset, self).__init__(*args, **kwargs)

    def get_image_path(self, folder, frame_index, side):
        f_str = "{:06d}{}".format(frame_index, self.img_ext)
        image_path = os.path.join(
            self.data_path,
            "crop".format(self.side_map[side]),
            f_str)
        return image_path

    def get_depth(self, folder, frame_index, side, do_flip):
        f_str = "{:010d}.png".format(frame_index)
        depth_path = os.path.join(
            self.data_path,
            "proj_depth/groundtruth/image_0{}".format(self.side_map[side]),
            f_str)

        depth_gt = pil.open(depth_path)
        depth_gt = depth_gt.resize(self.full_res_shape, pil.NEAREST)
        depth_gt = np.array(depth_gt).astype(np.float32) / 256

        if do_flip:
            depth_gt = np.fliplr(depth_gt)

        return depth_gt
