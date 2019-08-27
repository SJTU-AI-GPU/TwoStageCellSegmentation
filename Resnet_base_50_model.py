# !/usr/bin/python3
# coding: utf-8
# @author: Deng Junwei
# @date: 2019/3/17
# @institute:SJTU

'''This a pytorch implementation of Resnet-50~152 based on http://ethereon.github.io/netscope/#/gist/db945b393d40bfa26006'''

import torch
import torch.nn as nn
from torch import functional as F
from Resnet_base_50_parts import *
from torchvision import transforms
import numpy as np
import time
loader = transforms.ToTensor() 
unloader = transforms.ToPILImage()

# model prototype
class Unet_Resnet(nn.Module):
    def __init__(self, in_channels, layer_num = [3,4,6,3]):
        super(Unet_Resnet,self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, stride = 1, kernel_size = 7, padding = 3, bias = False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace = True)
        self.in_channels = 64
        self.pool1 = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)
        self.conv2_x = self.make_Resnet_layer(128, layer_num[0], stride = 1)# 256,512,1024,2048
        self.conv3_x = self.make_Resnet_layer(256, layer_num[1], stride = 2)
        self.conv4_x = self.make_Resnet_layer(512, layer_num[2], stride = 2)
        self.conv5_x = self.make_Resnet_layer(1024, layer_num[3], stride = 2)
        self.dbconv_5 = double_conv(1024, 1024)
        self.dbconv_4 = double_conv(1024, 512)
        self.dbconv_3 = double_conv(512, 256)
        self.dbconv_2 = double_conv(256, 128)
        self.dbconv_1 = double_conv(128, 128)
        self.out4D = nn.Conv2d(in_channels = 128, out_channels = 4, kernel_size = 1)
        self.up_4 = up(1024, 512)
        self.up_3 = up(512, 256)
        self.up_2 = up(256, 128)
        self.up_1 = up(128, 64)
        self.sig = nn.Sigmoid()
        self.softmax = nn.Softmax2d()
    def make_Resnet_layer(self, out_channels, blocks, stride=1):
		#block is Residual block, out_channels is your output willing output channerls, blocks is the numbers you want in Resnet-50 is [3,4,6,3]
    	layer=[]
    	block1 = Residual_block(
    		in_channels = self.in_channels, out_channels = out_channels, stride = stride, 
    		downsample = nn.Sequential(nn.Conv2d(self.in_channels, out_channels, stride = stride, 
    			kernel_size = 1, padding = 0, bias = False), 
    		nn.BatchNorm2d(out_channels)))
    	layer.append(block1)
    	self.in_channels = out_channels
    	for i in range(1,blocks):
    		layer.append(Residual_block(
    			in_channels = out_channels, out_channels = out_channels))
    	return nn.Sequential(*layer)
    # transforms.CenterCrop(size)
    def crop(self,size,x):
        tmp = np.zeros((x.shape[0],x.shape[1],size,size))
        tmp = torch.from_numpy(tmp).type(torch.FloatTensor)
        crop_num = int((x.shape[2]-size)/2)
        for i in range(x.shape[0]):
            tmp[i,:,:,:] = x[i,:,crop_num:(x.shape[2]-crop_num),:crop_num:(x.shape[2]-crop_num)]
        return tmp
    def forward(self, x):
#        print(time.time())
        out_1 = self.conv1(x) # 624*624*64
        out_1 = self.bn1(out_1) # 624*624*64
        out_1 = self.relu1(out_1) # 624*624*64
        out_1_crop = self.crop(504,out_1) # 504*504*64
        out_2 = self.pool1(out_1) # 312*312*64
        out_2 = self.conv2_x(out_2) # 312*312*128
        out_2_crop = self.crop(256, out_2) # 256*256*128
        out_3 = self.conv3_x(out_2) # 156*156*256
        out_3_crop = self.crop(132, out_3) # 132*132*256
        out_4 = self.conv4_x(out_3) # 78*78*512
        out_4_crop = self.crop(70,out_4) # 70*70*512
        out_5 = self.conv5_x(out_4) # 39*39*1024
#        print(time.time())
        
        out_5_out = self.dbconv_5(out_5)
        out_4_in = self.up_4(out_5_out, out_4_crop)
        out_4_out = self.dbconv_4(out_4_in)
        out_3_in = self.up_3(out_4_out, out_3_crop)
        out_3_out = self.dbconv_3(out_3_in)
        out_2_in = self.up_2(out_3_out, out_2_crop)
        out_2_out = self.dbconv_2(out_2_in)
        out_1_in = self.up_1(out_2_out, out_1_crop)
        out_1_out = self.dbconv_1(out_1_in) # 500*500*128
        out = self.out4D(out_1_out)
#        print(time.time())
        # out = self.softmax(out)        
        return out
        
