import random

import math
import numpy as np
from torch.utils.data import Dataset
import os
import nibabel as nib
import torch


class Getfile(Dataset):

    def __init__(self, base_dir=None, split='train'):
        self._base_dir = base_dir
        self.split = split
        train_path = self._base_dir + '/train.txt'
        val_path = self._base_dir + '/val.txt'
        test_path = self._base_dir + '/test.txt'
        if split == 'train':
            with open(train_path, 'r') as f:
                self.image_list = f.readlines()
        elif split == 'val':
            with open(val_path, 'r') as f:
                self.image_list = f.readlines()
        elif split == 'test':
            with open(test_path, 'r') as f:
                self.image_list = f.readlines()

        self.image_list = [item.replace('\n', '').split(",")[0] for item in self.image_list]
        print("total {} samples".format(len(self.image_list)))

        # 在初始化时一次性加载所有数据
        self.data = self._load_all_data(split)

    def _load_all_data(self, split):
        data = []

        for file_name in self.image_list:
            if split == 'test':
                train_path = os.path.join(self._base_dir, 'ct_test')
                file_path = os.path.join(train_path, file_name)
                label_file_name = file_name.replace('image', 'label_encrypt_1mm')
                label_file_path = os.path.join(train_path, label_file_name)
            else:
                train_path = os.path.join(self._base_dir, 'ct_train')
                file_path = os.path.join(train_path, file_name)
                label_file_name = file_name.replace('image', 'label')
                label_file_path = os.path.join(train_path, label_file_name)
            img = torch.from_numpy(nib.load(file_path).get_fdata()).float()
            img = img.reshape(1, img.shape[0], img.shape[1], img.shape[2])
            label = torch.from_numpy(nib.load(label_file_path).get_fdata()).float()
            data.append({'image': img, 'label': label})
        return data

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        return self.data[idx]

    def get_filename(self, idx):
        return self.image_list[idx]


class SlideWindowTrainDataset(Dataset):
    def __init__(self, base_dataset, patch_size, stride_xy, stride_z, num_classes, num_random_patches):
        self.base_dataset = base_dataset
        self.patch_size = patch_size
        self.stride_xy = stride_xy
        self.stride_z = stride_z
        self.num_classes = num_classes
        self.total_patches = self.calculate_total_patches()
        self.random_patches = []  # 用于存储随机切割的 patch
        self.num_random_patches = num_random_patches
        self.generate_random_patches()

    def calculate_total_patches(self):
        total_patches = 0
        for i in range(len(self.base_dataset)):
            sample = self.base_dataset[i]
            image = sample['image']
            w, h, d = image.shape[1], image.shape[2], image.shape[3]
            sx = math.ceil((w - self.patch_size[0]) / self.stride_xy) + 1
            sy = math.ceil((h - self.patch_size[1]) / self.stride_xy) + 1
            sz = math.ceil((d - self.patch_size[2]) / self.stride_z) + 1
            total_patches += sx * sy * sz
        return total_patches

    def generate_random_patches(self):
        for i in range(len(self.base_dataset)):
            sample = self.base_dataset[i]
            image = sample['image']
            label = sample['label']
            ran_nums = self.num_random_patches // len(self.base_dataset)
            for _ in range(ran_nums):
                random_patch = self.add_random_patch(image, label)
                self.random_patches.append(random_patch)

    def add_random_patch(self, image, label):
        # 随机选择起始坐标
        w, h, d = image.shape[1], image.shape[2], image.shape[3]
        start_x = random.randint(0, w - self.patch_size[0])
        start_y = random.randint(0, h - self.patch_size[1])
        start_z = random.randint(0, d - self.patch_size[2])

        # 确保所选坐标不会导致 patch 越界
        if start_x + self.patch_size[0] > w:
            start_x = w - self.patch_size[0]
        if start_y + self.patch_size[1] > h:
            start_y = h - self.patch_size[1]
        if start_z + self.patch_size[2] > d:
            start_z = d - self.patch_size[2]

        # 根据所选参数提取随机 patch
        patch_image = image[:, start_x:start_x + self.patch_size[0],
                      start_y:start_y + self.patch_size[1],
                      start_z:start_z + self.patch_size[2]]

        patch_label = label[start_x:start_x + self.patch_size[0],
                      start_y:start_y + self.patch_size[1],
                      start_z:start_z + self.patch_size[2]]

        sample = {'image': patch_image, 'label': patch_label}
        return sample

    def __len__(self):
        return self.total_patches + len(self.random_patches)

    def __getitem__(self, idx):
        if idx < self.total_patches:
            current_idx = 0
            for i in range(len(self.base_dataset)):
                sample = self.base_dataset[i]
                image = sample['image']
                label = sample['label']
                w, h, d = image.shape[1], image.shape[2], image.shape[3]
                sx = math.ceil((w - self.patch_size[0]) / self.stride_xy) + 1
                sy = math.ceil((h - self.patch_size[1]) / self.stride_xy) + 1
                sz = math.ceil((d - self.patch_size[2]) / self.stride_z) + 1
                num_patches = sx * sy * sz

                if current_idx + num_patches > idx:
                    patch_idx = idx - current_idx
                    patch_x = patch_idx % sx
                    patch_y = (patch_idx // sx) % sy
                    patch_z = patch_idx // (sx * sy)

                    start_x = patch_x * self.stride_xy
                    start_y = patch_y * self.stride_xy
                    start_z = patch_z * self.stride_z

                    patch_image = image[:, start_x:start_x + self.patch_size[0],
                                  start_y:start_y + self.patch_size[1],
                                  start_z:start_z + self.patch_size[2]]

                    patch_label = label[start_x:start_x + self.patch_size[0],
                                  start_y:start_y + self.patch_size[1],
                                  start_z:start_z + self.patch_size[2]]

                    # 如果 patch 太小，请从图像的末尾提取
                    if patch_image.shape[1] < self.patch_size[0]:
                        start_x = w - self.patch_size[0]
                    if patch_image.shape[2] < self.patch_size[1]:
                        start_y = h - self.patch_size[1]
                    if patch_image.shape[3] < self.patch_size[2]:
                        start_z = d - self.patch_size[2]

                    patch_image = image[:, start_x:start_x + self.patch_size[0],
                                  start_y:start_y + self.patch_size[1],
                                  start_z:start_z + self.patch_size[2]]

                    patch_label = label[start_x:start_x + self.patch_size[0],
                                  start_y:start_y + self.patch_size[1],
                                  start_z:start_z + self.patch_size[2]]

                    sample = {'image': patch_image, 'label': patch_label}
                    return sample

                current_idx += num_patches
            raise IndexError("Index out of range")
        else:
            random_patch = random.choice(self.random_patches)
            return random_patch

def get_one_hot_label(gt, label_intensities=None, channel_first=False):
    if label_intensities is None:
        label_intensities = sorted(torch.unique(gt))
        # 获取类别值
    num_classes = len(label_intensities)  # 类别数
    label = torch.round(gt)
    if channel_first:
        label = torch.zeros((num_classes, *label.shape), dtype=torch.float32)

        for k in range(num_classes):
            label[k] = (gt == label_intensities[k])

        label[0] = ~torch.sum(label[1:], dim=0).bool()
    else:
        label = torch.zeros((*label.shape, num_classes), dtype=torch.float32)
        # 创建一个全0数组，形状为label+通道数
        for k in range(num_classes):
            label[..., k] = (gt == label_intensities[k])
            # 对应通道的布尔值，相等的位置为1
        label[..., 0] = ~torch.sum(label[..., 1:], dim=-1).bool()
        # 通过logical_not取反获取背景通道
    return label
