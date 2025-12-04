import os

import numpy as np
import rasterio
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


class BRIGHTDataset(Dataset):
    def __init__(self, root_dir, image_names, channel_mask, target_size, augmentations=None, mode='train'):
        self.root_dir = root_dir
        self.augmentations = augmentations
        self.mode = mode
        self.target_size = target_size
        self.channel_mask = channel_mask

        if len(channel_mask) != 6:
            raise ValueError("channel_mask must be length 6")

        self.indices = [i for i, x in enumerate(channel_mask) if x == '1']
        self.use_pre = '1' in channel_mask[:3]
        self.use_post = '1' in channel_mask[3:]

        self.pre_dir = os.path.join(root_dir, 'pre-event')
        self.post_dir = os.path.join(root_dir, 'post-event')
        self.target_dir = os.path.join(root_dir, 'target')

        self.image_names = []

        for name in image_names:
            pre_name = name.replace('_post_disaster.tif', '_pre_disaster.tif')
            target_name = name.replace('_post_disaster.tif', '_building_damage.tif')

            pre_path = os.path.join(self.pre_dir, pre_name)
            post_path = os.path.join(self.post_dir, name)
            target_path = os.path.join(self.target_dir, target_name)

            if not os.path.exists(target_path):
                continue
            if self.use_pre and not os.path.exists(pre_path):
                continue
            if self.use_post and not os.path.exists(post_path):
                continue

            self.image_names.append(name)

    def __len__(self):
        return len(self.image_names)

    def resize_tensor(self, tensor, mode='bilinear'):
        return F.interpolate(
            tensor.unsqueeze(0),
            size=(self.target_size, self.target_size),
            mode=mode,
            align_corners=False if mode == 'bilinear' else None
        ).squeeze(0)

    def load_image(self, path):
        with rasterio.open(path) as src:
            img = src.read()

        c, h, w = img.shape
        if c == 1:
            img = np.repeat(img, 3, axis=0)
        elif c > 3:
            img = img[:3]

        return torch.from_numpy(img).float()

    def __getitem__(self, idx):
        name = self.image_names[idx]
        pre_name = name.replace('_post_disaster.tif', '_pre_disaster.tif')
        target_name = name.replace('_post_disaster.tif', '_building_damage.tif')

        pre_path = os.path.join(self.pre_dir, pre_name)
        post_path = os.path.join(self.post_dir, name)
        target_path = os.path.join(self.target_dir, target_name)

        try:
            parts = []

            if self.use_pre:
                img = self.load_image(pre_path)
                parts.append(self.resize_tensor(img))
            else:
                parts.append(torch.zeros(3, self.target_size, self.target_size))

            if self.use_post:
                img = self.load_image(post_path)
                parts.append(self.resize_tensor(img))
            else:
                parts.append(torch.zeros(3, self.target_size, self.target_size))

            full_tensor = torch.cat(parts, dim=0)
            image_tensor = full_tensor[self.indices]

            with rasterio.open(target_path) as src:
                mask = src.read(1)

            mask = torch.from_numpy(mask).long()
            mask_tensor = self.resize_tensor(mask.unsqueeze(0).float(), mode='nearest').squeeze(0).long()

            return image_tensor, mask_tensor, name

        except (Exception,):
            return torch.zeros(len(self.indices), self.target_size, self.target_size), \
                torch.zeros(self.target_size, self.target_size).long(), \
                name
