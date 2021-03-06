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


class GTAVKITTIDataset(MonoDataset):
    """Superclass for different types of GTAVKITTI dataset loaders
    """
    def __init__(self, *args, **kwargs):
        super(GTAVKITTIDataset, self).__init__(*args, **kwargs)

        self.K = np.array([[0.58, 0, 0.5, 0],
                           [0, 1.92, 0.5, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=np.float32)

        self.full_res_shape = (1242, 375)
        img_w, img_h = self.full_res_shape

        self.side_map = {"2": 2, "3": 3, "l": 2, "r": 3}
        # self.K[0,:] /= img_w
        # self.K[1,:] /= img_h

    def check_depth(self):
        line = self.filenames[0].split()
        scene_name = line[0]
        frame_index = int(line[1])

        if scene_name.startswith('gtav_data'):
            depth_filename = os.path.join(
                    self.data_path,
                    scene_name,
                    'depth',
                    '{:06d}.npy'.format(frame_index))
        else:
            depth_filename = os.path.join(
                self.data_path,
                scene_name,
                "velodyne_points/data/{:010d}.bin".format(int(frame_index)))


        return os.path.isfile(depth_filename)

    def get_color(self, folder, frame_index, side, do_flip):
        color = self.loader(self.get_image_path(folder, frame_index, side))

        if do_flip:
            color = color.transpose(pil.FLIP_LEFT_RIGHT)

        return color


class GTAVKITTIRAWDataset(GTAVKITTIDataset):
    """GTAVKITTI dataset which loads the original velodyne depth maps for ground truth
    """
    def __init__(self, *args, **kwargs):
        super(GTAVKITTIRAWDataset, self).__init__(*args, **kwargs)

    def get_image_path(self, folder, frame_index, side):
        if folder.startswith('gtav_data'):
            f_str = "{:06d}{}".format(frame_index, self.img_ext)
            folder = os.path.join(folder, "image_2")
        else:
            f_str = "{:010d}{}".format(frame_index, self.img_ext)
            folder = os.path.join(folder, "image_0{}/data".format(self.side_map[side]))

        image_path = os.path.join(
                self.data_path,
                folder,
                f_str)
        return image_path

    def get_depth(self, folder, frame_index, side, do_flip):
        if folder.startswith('gtav_data'):
            f_str = "{:06d}.npy".format(frame_index)
            depth_filename = os.path.join(
                self.data_path,
                folder,
                'depth',
                f_str)

            depth_gt = np.load(depth_filename)
            depth_gt[depth_gt > 80] = 0 # Max valid depth of LiDAR is 80
        else:
            calib_path = os.path.join(self.data_path, folder.split("/")[0], folder.split("/")[1])

            velo_filename = os.path.join(
                self.data_path,
                folder,
                "velodyne_points/data/{:010d}.bin".format(int(frame_index)))

            depth_gt = generate_depth_map(calib_path, velo_filename, self.side_map[side])
            depth_gt = skimage.transform.resize(
                depth_gt, self.full_res_shape[::-1], order=0, preserve_range=True, mode='constant')

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
class GTAVKITTIDepthDataset(GTAVKITTIDataset):
    """GTAVKITTI dataset which uses the updated ground truth depth maps
    """
    def __init__(self, *args, **kwargs):
        super(GTAVKITTIDepthDataset, self).__init__(*args, **kwargs)

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
