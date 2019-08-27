# !/usr/bin/python3
# coding: utf-8
# @author: Deng Junwei
# @date: 2019/3/16
# @institute:SJTU

'''This a pytorch implementation of Resnet-50~152 based on http://ethereon.github.io/netscope/#/gist/db945b393d40bfa26006'''

import torch
import torch.nn as nn
from torch import functional as F

# Device identification
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # Try to find out if the computer have a CUDA with Nivida GPU, else we will use CPU to work

# Convolution layer interface
def Conv2d3X3(in_channels, out_channels, stride = 1): # 3 x 3 kernel, with padding = 1 (for not reduce the output) and bias = False, we will only use these two layers in the process
	return nn.Conv2d(in_channels, out_channels, stride = stride, kernel_size = 3, padding = 1, bias = False)
def Conv2d1X1(in_channels, out_channels, stride = 1): # 1 x 1 kernel, with padding = 1 (for not reduce the output) and bias = False, we will only use these two layers in the process
	return nn.Conv2d(in_channels, out_channels, stride = stride, kernel_size = 1, padding = 0, bias = False)

# Residual block for Resnet-50, 101, 152
class Residual_block(nn.Module):
	'''
		Residual block for Resnet-50, 101, 152
		1X1 3X3 1X1 with batchnorm and relu and possibly consample block on residual route
	'''
	def __init__(self, in_channels, out_channels, stride = 1, downsample = None):
		super(Residual_block, self).__init__()
		self.conv1 = Conv2d1X1(in_channels, int(out_channels/4), stride = stride)
		self.bn1 = nn.BatchNorm2d(int(out_channels/4))
		self.relu1 = nn.ReLU(inplace = True)
		self.conv2 = Conv2d3X3(int(out_channels/4), int(out_channels/4), stride = 1)
		self.bn2 = nn.BatchNorm2d(int(out_channels/4))
		self.relu2 = nn.ReLU(inplace = True)
		self.conv3 = Conv2d1X1(int(out_channels/4), out_channels, stride = 1)
		self.bn3 = nn.BatchNorm2d(out_channels)
		self.relu3 = nn.ReLU(inplace = True)
		self.downsample = downsample

	def forward(self, x):
		residual = x
		out = self.conv1(x)
		out = self.bn1(out)
		out = self.relu1(out)
		out = self.conv2(out)
		out = self.bn2(out)
		out = self.relu2(out)
		out = self.conv3(out)
		out = self.bn3(out)
		if self.downsample:
			residual = self.downsample(x)
		out += residual
		out = self.relu3(out)
		return out

# convolutional block on upsampling route
class double_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=0),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=0),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        x = self.conv(x)
        return x

# up-sampling block
class up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True):
        super(up, self).__init__()
        if bilinear:
            self.up = nn.Sequential(nn.Upsample(scale_factor=2, mode='nearest'),
									nn.Conv2d(in_channels = in_ch, out_channels = out_ch, kernel_size = 3, padding = 1))
        else:
            self.up = nn.ConvTranspose2d(in_ch, in_ch//2, 2, stride=2)# I changed the code from https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_parts.py

    def forward(self, x1, x2):  
        '''
		TODO:
		change x2 to raw data
		'''
		# x1 is the up-sampling matrix, x2 is the croped input
        x1 = self.up(x1)
        # x1 = x1.to(device)
        x2 = x2.to(device)  
        # for padding issues, see 
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat((x2, x1), dim=1) # dimention = 3 because we have dimention = [batch_size, height, width, channel]
        return x

