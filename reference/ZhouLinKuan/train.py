import argparse
import logging
import random
import shutil
import sys
import time

import numpy as np
import torch.multiprocessing as mp
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from pathlib import Path
from torch import optim
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image
from evaluate import evaluate
from unet.unet_model import UNet
from newloader import Getfile, SlideWindowTrainDataset, get_one_hot_label
from utils.dice_score import DiceLoss
import os

parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
parser.add_argument('--data_path', type=str,
                    default='data', help='Name of Experiment')
parser.add_argument('--checkpoint_path', type=str,
                    default='checkpoints', help='Name of Experiment')
parser.add_argument('--epochs', '-e', metavar='E', type=int, default=20, help='Number of epochs')
parser.add_argument('--batch_size', '-b', dest='batch_size', metavar='B', type=int, default=5, help='Batch size')
parser.add_argument('--lr', '-l', metavar='LR', type=float, default=1e-4,
                    help='Learning rate', dest='lr')
parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
parser.add_argument('--scale', '-s', type=float, default=1, help='Downscaling factor of the images')
parser.add_argument('--validation', '-v', dest='val', type=float, default=10.0,
                    help='Percent of the data that is used as validation (0-100)')
parser.add_argument('--save_checkpoint', type=bool, default=True, help='save_checkpoint')
parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')

parser.add_argument('--classes', '-c', type=int, default=8, help='Number of classes')

parser.add_argument('--patch_size', type=list, default=[128, 128, 96],
                    help='patch size of network input')
parser.add_argument('--stride_xy', type=int, default=118,
                    help='stride_xy')
parser.add_argument('--stride_z', type=int, default=86,
                    help='stride_z')
parser.add_argument('--seed', type=int, default=1337, help='random seed')
args = parser.parse_args()


def worker_init_fn(worker_id):
    random.seed(args.seed + worker_id)


def train(args):
    learning_rate = args.lr
    train_data_path = args.data_path
    batch_size = args.batch_size
    epochs = args.epochs
    num_classes = args.classes
    save_checkpoint = args.save_checkpoint
    dir_checkpoint = args.checkpoint_path
    patch_size = args.patch_size
    stride_xy = args.stride_xy
    stride_z = args.stride_z
    start_time = time.time()
    print('start')
    train_dataset = Getfile(base_dir=train_data_path, split='train')

    try:
        slide_window_trainset = SlideWindowTrainDataset(
            base_dataset=train_dataset,
            patch_size=patch_size,
            stride_xy=stride_xy,
            stride_z=stride_z,
            num_classes=num_classes,
            num_random_patches=1
        )

        # dataset = Gettrainfile(train_data_path)
        train_loader = DataLoader(slide_window_trainset, batch_size=batch_size, shuffle=True,
                                  num_workers=0, pin_memory=True, worker_init_fn=worker_init_fn)

    except (AssertionError, RuntimeError, IndexError):
        raise RuntimeError("Failed to load the dataset.")

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Checkpoints:     {save_checkpoint}
        Device:          {device.type}
    ''')
    model.train()
    optimizer = optim.Adam(model.module.parameters(), lr=learning_rate, weight_decay=0.0001)
    ce_loss = CrossEntropyLoss()
    logging.info('{} iterations per epoch'.format(len(train_loader)))

    for epoch in range(1, epochs + 1):
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch}/{epochs}', unit='batch')

        for batch in progress_bar:
            end_time = time.time()
            execution_time = end_time - start_time
            print(f"时间: {execution_time} 秒")
            start_time = time.time()
            patch_num = 0
            patch_loss = torch.tensor(0.0, device=device, requires_grad=True)  # 创建一个张量来累积梯度
            volume_batch, label_batch = batch['image'].cuda(), batch['label'].cuda()
            optimizer.zero_grad()  # 在每个小批次前清零梯度

            for patch_idx in range(volume_batch.size(0)):
                patch = volume_batch[patch_idx, :, :, :, :]
                patch = patch.cuda()

                patch = patch.unsqueeze(0).expand(batch_size, -1, -1, -1, -1)
                image = patch.to(torch.float32)

                patch_label = label_batch[patch_idx, :, :, :]
                patch_label = patch_label.cuda()
                patch_label = patch_label.cuda()
                patch_label = patch_label.unsqueeze(0).expand(batch_size, -1, -1, -1)
                # patch_label = patch_label.to(torch.int64)
                patch_label = get_one_hot_label(patch_label, label_intensities=(
                    0.0, 205.0, 420.0, 500.0, 550.0, 600.0, 820.0, 850.0))
                patch_label = patch_label.permute(0, 4, 1, 2, 3)
                true_mask = patch_label.cuda()
                masks_pred = model(image).cuda()
                # masks_pred = masks_pred.float()
                # loss_dice = DiceLoss(num_classes)(masks_pred, true_mask)
                loss_ce = ce_loss(masks_pred, true_mask.to(torch.float32))
                # loss = 0.5 * (loss_dice + loss_ce)
                loss = loss_ce
                patch_loss = patch_loss + loss
                patch_num += 1

                # logging.info('info/total_loss: %f, epoch: %d', loss, epoch)
                # logging.info('info/loss_ce: %f, epoch: %d', loss_ce, epoch)
                # logging.info('info/loss_dice: %f, epoch: %d', loss_dice, epoch)

                # logging.info(
                #   'epoch %d : loss : %f, loss_ce: %f, loss_dice: %f',
                #  epoch, loss.item(), loss_ce.item(), loss_dice.item()
                # )
                # logging.info('patch_loss: %f, epoch: %d', loss, epoch)
                # logging.info('val_score: {}'.format(val_score))
            # patch_progress_bar.update(1)
            # patch_progress_bar.close()

            patch_loss.backward()
            optimizer.step()
            batch_loss = patch_loss / patch_num
            logging.info('batch_loss: %f, epoch: %d', batch_loss, epoch)
        lr_ = learning_rate * (1.0 - epoch / (epochs + 1)) ** 0.9
        logging.info('lr: %f, epoch: %d', lr_, epoch)
        val_score = evaluate(model, train_data_path, device, batch_size, num_classes=num_classes, patch_size=patch_size,
                             stride_xy=stride_xy, stride_z=stride_z)
        logging.info('val_score: %f, epoch: %d', val_score, epoch)
        if save_checkpoint:
            Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
            state = model.state_dict()
            check_path = os.path.join(dir_checkpoint, f'checkpoint_epoch{epoch}.pth')
            torch.save(state, check_path)
            logging.info(f'已保存检查点 {epoch}!')
    return "Training Finished!"


if __name__ == '__main__':
    mp.set_start_method('spawn')
    print("Current working directory:", os.getcwd())
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    log_file = 'training_log.txt'
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    # 将FileHandler添加到日志记录器
    logger = logging.getLogger()
    logger.addHandler(file_handler)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')
    torch.backends.cudnn.enable = True
    torch.backends.cudnn.benchmark = True
    # Change here to adapt to your data
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel
    model = UNet(in_channels=1, out_channels=args.classes)
    model = nn.DataParallel(model, device_ids=[0, 1])  # 指定要使用的GPU设备编号

    if args.load:
        state_dict = torch.load(args.load, map_location=device)
        del state_dict['mask_values']
        model.load_state_dict(state_dict)
        logging.info(f'Model loaded from {args.load}')
    model.to(device=device)
    train(args)
