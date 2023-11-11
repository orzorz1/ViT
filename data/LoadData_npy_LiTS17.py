import sys
import numpy
import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset
import torch.nn as nn
import random
np.set_printoptions(threshold=np.inf)
np.set_printoptions(suppress=True)
import gc
from memory_profiler import profile
import time


def get_bounding_box(img):
    width, height, deep = img.shape
    box = [0, 0, 0, 0, 0, 0]  # [width_low, width_high, height_low, height_high, deep_low, deep_high]
    for i in range(width):
        img_x = img[i, :, :]
        a = numpy.ones(img_x.shape)
        if (img_x * a).sum() != 0:
            box[0] = i
            break
    for i in range(width):
        img_x = img[width-i-1, :, :]
        a = numpy.ones(img_x.shape)
        if (img_x * a).sum() != 0:
            box[1] = width-i-1
            break
    for i in range(height):
        img_y = img[:, i, :]
        a = numpy.ones(img_y.shape)
        if (img_y * a).sum() != 0:
            box[2] = i
            break
    for i in range(height):
        img_y = img[:, height-i-1, :]
        a = numpy.ones(img_y.shape)
        if (img_y * a).sum() != 0:
            box[3] = height-i-1
            break
    for i in range(deep):
        img_z = img[:, :, i]
        a = numpy.ones(img_z.shape)
        if (img_z * a).sum() != 0:
            box[4] = i
            break
    for i in range(deep):
        img_z = img[:, :, deep-i-1]
        a = numpy.ones(img_z.shape)
        if (img_z * a).sum() != 0:
            box[5] = deep-i-1
            break
    del img_x,img_y,img_z,a
    gc.collect()
    return box

def setHU(img_arr, min, max):
    img_arr = np.clip(img_arr, min, max)
    img_arr = img_arr.astype(np.int16)
    return img_arr

def set_label(x):

    return x

# (width, height, deep) -> (channel, deep, width, height)
def reshape(pic):
    pic = np.expand_dims(pic, axis=0)
    return pic

def read_dataset(path):
    img_arr = np.load(path)
    img_arr = setHU(img_arr, -200, 250)
    return img_arr

def read_dataset_test(path):
    img_arr = np.load(path)
    # img_arr = img_arr * 2400 / 256 - 1000
    img_arr = setHU(img_arr, -200, 250)
    return img_arr

def read_label(path):
    img_arr = np.load(path)
    img_arr = img_arr.astype(np.int16)
    return img_arr

# 从一个图像中随机取包含肿瘤的patch
def get_patchs_from_one_img(img_x, img_y, patch_size, number):
    start = time.time()
    box = get_bounding_box(img_y)
    width, height, deep = img_y.shape
    verte_range = [
        max(box[0] - patch_size[0], 0), min(box[1], width - patch_size[0]),
        max(box[2] - patch_size[1], 0), min(box[3], height - patch_size[1]),
        max(box[4] - patch_size[2], 0), min(box[5], deep - patch_size[2])
    ]
    verte_coords = [
        (
            random.randint(verte_range[0], verte_range[1]),
            random.randint(verte_range[2], verte_range[3]),
            random.randint(verte_range[4], verte_range[5])
        )
        for _ in range(number)
    ]

    patchs_x = [
        [
            np.expand_dims(
                img_x[v[0]:v[0] + patch_size[0], v[1]:v[1] + patch_size[1], v[2]:v[2] + patch_size[2]], axis=0
            ),
            np.array(v)
        ]
        for v in verte_coords
    ]

    patchs_y = [
        [img_y[v[0]:v[0] + patch_size[0], v[1]:v[1] + patch_size[1], v[2]:v[2] + patch_size[2]]]
        for v in verte_coords
    ]
    print("加载patch耗时：", time.time() - start)
    return np.array(patchs_x), np.array(patchs_y)


# 将从多张图片取的随机patch组合在一起
def get_patches(dirX, dirY, begin, end, patch_size, seed):
    random.seed(seed)
    patches_X = np.array([1])
    patches_Y = np.array([1])
    l = [x for x in range(begin, end+1)]
    random.shuffle(l)
    for i in l:
        patches_x = []
        patches_y = []
        gc.collect()
        print("正在加载第", i, "张图像")
        path_x = dirX[i]
        x = read_dataset(path_x)
        path_y = dirY[i]
        y = read_label(path_y)
        x1, y1 = get_patchs_from_one_img(x, y, patch_size, 140)
        for j in range(140):
            patches_x.append(x1[j])
            patches_y.append(y1[j])
        x2 = np.array(mean_patch(reshape(x), patch_size, 1))
        y2 = np.array(mean_patch(reshape(y), patch_size, 1))
        permutation = np.random.permutation(x2.shape[0])
        x2 = x2[permutation]
        y2 = y2[permutation]
        a = 70 if x2.shape[0] > 70 else x2.shape[0]
        for j in range(a):
            x2[j][0] = x2[j][0].astype(np.int16)
            patches_x.append(x2[j])
            patches_y.append(y2[j][0])
        del x, y, x1, y1, x2, y2
        gc.collect()
        patches_x = np.array(patches_x)
        patches_y = np.array(patches_y)
        if i == l[0]:
            patches_X = patches_x
            patches_Y = patches_y
        else:
            patches_X = np.concatenate((patches_X, patches_x), axis=0)
            patches_Y = np.concatenate((patches_Y, patches_y), axis=0)
    print(patches_X.shape)
    print(patches_Y.shape)
    return patches_X, patches_Y


class load_dataset_one(Dataset):
    def __init__(self, dirX, dirY, index, patch_size):
        path_x = dirX[index]
        x = read_dataset(path_x)
        path_y = dirY[index]
        y = read_label(path_y)
        patchs_x, patchs_y = get_patchs_from_one_img(x, y, patch_size, 30)
        print("数据加载完成，shape：", patchs_y.shape)
        imgs = []
        for i in range(patchs_x.shape[0]):
            imgs.append((patchs_x[i], patchs_y[i]))
        del x, y, patchs_x, patchs_y
        gc.collect()
        self.imgs = imgs

    def __getitem__(self, index):
        img, label = self.imgs[index]
        position = img[1]
        img = np.array(img[0]).astype(float)
        target = np.array(label)
        return torch.from_numpy(img).to(torch.float32), torch.from_numpy(target).long(), position

    def __len__(self):
        return len(self.imgs)

class load_dataset(Dataset):
    def __init__(self, dirX, dirY, begin, end, seed, patch_size):
        """

        :rtype: object
        """
        patchs_x, patchs_y = get_patches(dirX, dirY, begin, end, patch_size, seed)
        print("数据加载完成，shape：", patchs_y.shape)
        imgs = []
        for i in range(patchs_x.shape[0]):
            imgs.append((patchs_x[i], patchs_y[i]))
        del patchs_x, patchs_y
        self.imgs = imgs

    def __getitem__(self, index):
        img, label = self.imgs[index]
        position = img[1]
        img = np.array(img[0]).astype(float)
        target = np.array(label)
        return torch.from_numpy(img).to(torch.float32), torch.from_numpy(target).long(), position

    def __len__(self):
        return len(self.imgs)
# 随机取patch

# 均匀取patch
def mean_patch(img_arr, size, overlap_factor):
    patchs = []
    patch_size = size
    channel, width, height, deep = img_arr.shape
    for i in range(0, width-patch_size[0]+1, patch_size[0]//overlap_factor):
        for j in range(0, height-patch_size[1]+1, patch_size[1]//overlap_factor):
            for k in range(0, deep-patch_size[2]+1, patch_size[2]//overlap_factor):
                patch = []
                patch.append(img_arr[:,i:i+patch_size[0],j:j+patch_size[1],k:k+patch_size[2]])
                patch.append([i, j, k])
                patchs.append(patch) #patchs[index][0]为patch，patchs[index][1]为patch在原始图像中的位置
    for k in range(0, deep-patch_size[2]+1, patch_size[2]//overlap_factor):
        j = height - patch_size[1]
        for i in range(0, width - patch_size[0] + 1, patch_size[0] // overlap_factor):
            patch = []
            patch.append(img_arr[:, i:i + patch_size[0], j:j + patch_size[1], k:k + patch_size[2]])
            patch.append([i, j, k])
            patchs.append(patch)
    for j in range(0, height-patch_size[1]+1, patch_size[1]//overlap_factor):
        k = deep - patch_size[2]
        for i in range(0, width - patch_size[0] + 1, patch_size[0] // overlap_factor):
            patch = []
            patch.append(img_arr[:, i:i + patch_size[0], j:j + patch_size[1], k:k + patch_size[2]])
            patch.append([i, j, k])
            patchs.append(patch)
    for k in range(0, deep-patch_size[2]+1, patch_size[2]//overlap_factor):
        i = width - patch_size[0]
        for j in range(0, height - patch_size[1] + 1, patch_size[1] // overlap_factor):
            patch = []
            patch.append(img_arr[:, i:i + patch_size[0], j:j + patch_size[1], k:k + patch_size[2]])
            patch.append([i, j, k])
            patchs.append(patch)
    i = width - patch_size[0]
    j = height - patch_size[1]
    k = deep - patch_size[2]
    patch = []
    patch.append(img_arr[:, i:i + patch_size[0], j:j + patch_size[1], k:k + patch_size[2]])
    patch.append([i, j, k])
    patchs.append(patch)
    del img_arr
    gc.collect()
    return np.array(patchs)

class load_dataset_test(Dataset):
    def __init__(self, dirX, dirY, index, patch_size):
        path_x = dirX[index]
        x = read_dataset(path_x)
        x = reshape(x)
        # save_nii(x.astype(np.int16)[0], "_" + str(index), index)
        path_y = dirY[index]
        y = read_label(path_y)
        y = reshape(y)
        patchs_x = mean_patch(x, patch_size, 4)
        patchs_y = mean_patch(y, patch_size, 4)
        print("数据加载完成")
        imgs = []
        for i in range(len(patchs_x)):
            imgs.append((patchs_x[i], patchs_y[i]))
        del x, y, patchs_x, patchs_y
        gc.collect()
        self.imgs = imgs

    def __getitem__(self, index):
        img, label = self.imgs[index]
        position = img[1]
        img = np.array(img[0]).astype(float)
        target = np.array(label[0])
        return torch.from_numpy(img).to(torch.float32), torch.from_numpy(target).long(), position

    def __len__(self):
        return len(self.imgs)


from config.LiTS17.config_resnet18 import *
if __name__ == '__main__':
    for i in range(1, 10):
        print(i)
        patchs_x, patchs_y = get_patches(train_image_list, train_label_list, 0, 99, patch_size, i)
        print("数据加载完成，shape：", patchs_y.shape)
        imgs = []
        for i in range(patchs_x.shape[0]):
            imgs.append((patchs_x[i], patchs_y[i]))
        del patchs_x, patchs_y
        np.save("train_data_" + str(i), imgs)

# path = "../dataset/crossmoda2021_ldn_{index}_Label.nii.gz".format(index=1)
# print(get_bounding_box(img))

# path = "../dataset/crossmoda2021_ldn_{index}_Label.nii.gz".format(index=1)
# img = read_label(path)
# img = reshape(img)
# img1 = patch(img,[128,128,32])