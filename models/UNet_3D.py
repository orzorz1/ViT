import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import modules.UNet_parts as up
import torch.optim as optim
import numpy as np
import modules.UNet_parts
import torchsummary


class UNet_3D(nn.Module):
    def __init__(self, channel_in, channel_out):
        super().__init__()
        self.left_conv_1 = up.double_conv(channel_in, 64)
        self.pool_1 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.left_conv_2 = up.double_conv(64, 128)
        self.pool_2 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.left_conv_3 = up.double_conv(128, 256)
        self.pool_3 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.left_conv_4 = up.double_conv(256, 512)
        self.pool_4 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.left_conv_5 = up.double_conv(512, 1024)

        self.deconv_1 = nn.ConvTranspose3d(1024, 512, kernel_size=2, stride=2)
        self.right_conv_1 = up.double_conv(1024, 512)
        self.deconv_2 = nn.ConvTranspose3d(512, 256, kernel_size=2, stride=2)
        self.right_conv_2 = up.double_conv(512, 256)
        self.deconv_3 = nn.ConvTranspose3d(256, 128, kernel_size=2, stride=2)
        self.right_conv_3 = up.double_conv(256, 128)
        self.deconv_4 = nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2)
        self.right_conv_4 = up.double_conv(128, 64)
        self.right_conv_5 = nn.Conv3d(64, channel_out, (3,3,3), padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 1：进行编码过程
        feature_1 = self.left_conv_1(x)
        x = self.pool_1(feature_1)

        feature_2 = self.left_conv_2(x)
        x = self.pool_2(feature_2)

        feature_3 = self.left_conv_3(x)
        x = self.pool_3(feature_3)

        feature_4 = self.left_conv_4(x)
        x = self.pool_4(feature_4)

        x = self.left_conv_5(x)

        # 2：进行解码过程
        x = self.deconv_1(x)
        # 特征拼接、
        x = torch.cat((feature_4, x), dim=1)
        x = self.right_conv_1(x)

        x = self.deconv_2(x)
        x = torch.cat((feature_3, x), dim=1)
        x = self.right_conv_2(x)

        x = self.deconv_3(x)

        x = torch.cat((feature_2, x), dim=1)
        x = self.right_conv_3(x)

        x= self.deconv_4(x)
        x = torch.cat((feature_1, x), dim=1)
        x = self.right_conv_4(x)

        x = self.right_conv_5(x)
        # out = self.sigmoid(out)

        return x




if __name__ == "__main__":
    input = torch.rand(48, 1, 64, 64, 32)
    print("input_size:", input.size())
    model = UNet_3D(1,2)
    device = torch.device("cuda")
    input = input.to(device)
    model = model.to(device)
    print(torch.cuda.memory_summary())
    # torchsummary.summary(model, (1, 128, 128, 32), batch_size=4, device="cpu")
    ouput = model(input)
    print("output_size:", ouput.size())

