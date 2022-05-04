#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 18 14:34:19 2021

@author: raj
"""


import torch
import torch.nn as nn
import torch.nn.functional as fn

import layer


def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                     stride=stride, padding=1, bias=False)

# Residual block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.InstanceNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.InstanceNorm2d(out_channels)
        self.downsample = downsample
        
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

class generative_net(nn.Module):
    
    CHANNELS = [1024, 512, 256, 128, 64, 32, 16]

    def __init__(self, resolution, in_nchannel=512, rnd_dim= 2, h_dim = 62, enc_p=0, dec_p=0, adain_layer=None, filled_cls=False):
        nn.Module.__init__(self)

        self.resolution = resolution

        # Input sparse tensor must have tensor stride 128.
        ch = self.CHANNELS
        self.grid_size = 32
        self.filled_cls = filled_cls
        
        self.layers = [2, 2, 2, 2, 2, 2, 2]
        self.in_channels = 16
        self.block = ResidualBlock
        self.conv = conv3x3(3, 16)
        
        self.bn = nn.InstanceNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self.make_layer(self.block, 16, self.layers[0])
        self.layer2 = self.make_layer(self.block, 32, self.layers[1], 2)
        self.layer3 = self.make_layer(self.block, 64, self.layers[2], 2)
        self.layer4 = self.make_layer(self.block, 128, self.layers[3], 2)
        self.layer5 = self.make_layer(self.block, 256, self.layers[4], 2)
        self.layer6 = self.make_layer(self.block, 512, self.layers[5], 2)
        self.layer7 = self.make_layer(self.block, 1024, self.layers[6], 2)
        self.adpool = nn.AdaptiveAvgPool2d(1)
        
        self.decoder = layer.AdaptiveDecoder(nn.ModuleList([
            nn.Dropout3d(dec_p),
            nn.InstanceNorm3d(512),
            nn.Conv3d(512, 512, 3, padding=1, bias=False),
            nn.Dropout3d(dec_p),
            nn.InstanceNorm3d(512),
            nn.ELU(),
            nn.Upsample(scale_factor=2, mode='trilinear'),  # 4
            nn.Conv3d(512, 512, 3, padding=1, bias=False),
            nn.Dropout3d(dec_p),
            nn.InstanceNorm3d(512),
            nn.ELU(),
            nn.Conv3d(512, 256, 3, padding=1, bias=False),
            nn.Dropout3d(dec_p),
            nn.InstanceNorm3d(256),
            nn.ELU(),
            nn.Upsample(scale_factor=2, mode='trilinear'),  # 8
            nn.Conv3d(256, 256, 3, padding=1, bias=False),
            nn.Dropout3d(dec_p),
            nn.InstanceNorm3d(256),
            nn.ELU(),
            nn.Conv3d(256, 128, 3, padding=1, bias=False),
            nn.Dropout3d(dec_p),
            nn.InstanceNorm3d(128),
            nn.ELU(),
            nn.Upsample(scale_factor=2, mode='trilinear'),  # 16
            nn.Conv3d(128, 128, 3, padding=1, bias=False),
            nn.Dropout3d(dec_p),
            nn.InstanceNorm3d(128),
            nn.ELU(),
            nn.Conv3d(128, 64, 3, padding=1, bias=False),
            nn.Dropout3d(dec_p),
            nn.InstanceNorm3d(64),
            nn.ELU(),
            nn.Upsample(scale_factor=2, mode='trilinear'),  # 32
            nn.Conv3d(64, 64, 3, padding=1, bias=False),
            nn.Dropout3d(dec_p),
            nn.InstanceNorm3d(64),
            nn.ELU(),
            nn.Conv3d(64, h_dim, 3, padding=1, bias=False),
            nn.Dropout3d(dec_p),
            nn.InstanceNorm3d(h_dim)
        ]), max_layer=adain_layer)
        
        self.generator = layer.PointCloudGenerator(
            nn.Sequential(nn.Conv1d(h_dim + rnd_dim, 64, 1),
                          nn.ELU(),
                          nn.Conv1d(64, 64, 1),
                          nn.ELU(),
                          nn.Conv1d(64, 32, 1),
                          nn.ELU(),
                          nn.Conv1d(32, 32, 1),
                          nn.ELU(),
                          nn.Conv1d(32, 16, 1),
                          nn.ELU(),
                          nn.Conv1d(16, 16, 1),
                          nn.ELU(),
                          nn.Conv1d(16, 8, 1),
                          nn.ELU(),
                          nn.Conv1d(8, 3, 1)),
            rnd_dim=rnd_dim, res=self.grid_size)
        
        self.density_estimator = nn.Sequential(
            nn.Conv3d(h_dim, 16, 1, bias=False),
            nn.BatchNorm3d(16),
            nn.ELU(),
            nn.Conv3d(16, 8, 1, bias=False),
            nn.BatchNorm3d(8),
            nn.ELU(),
            nn.Conv3d(8, 4, 1, bias=False),
            nn.BatchNorm3d(4),
            nn.ELU(),
            nn.Conv3d(4, 2, 1),
        )

        self.adaptive = nn.Sequential(
            nn.Linear(1024, sum(self.decoder.slices))
        )
    def generate_points(self, w, n_points=15000, regular_sampling=True):
        b = w.shape[0]
        x_rec = self.decoder(w)

        est = self.density_estimator(x_rec)
        dens = fn.relu(est[:, 0])
        dens_cls = est[:, 1].unsqueeze(1)
        dens = dens.view(b, -1)

        dens_s = dens.sum(-1).unsqueeze(1)
        mask = dens_s < 1e-12
        ones = torch.ones_like(dens_s)
        dens_s[mask] = ones[mask]
        dens = dens / dens_s
        dens = dens.view(b, 1, self.grid_size, self.grid_size, self.grid_size)

        if self.filled_cls:
            filled = torch.sigmoid(dens_cls).round()
            dens_ = filled * dens
            for i in range(b):
                if dens_[i].sum().item() < 1e-12:
                    dens_[i] = dens[i]
        else:
            dens_ = dens

        cloud, reg = self.generator(x_rec, dens_, n_points)

        return cloud, dens, dens_cls.squeeze(), reg

    def decode(self, z, n_points=15000, regular_sampling=True):
        b = z.shape[0]
        w = self.adaptive(z.view(b, -1))
        return self.generate_points(w, n_points, regular_sampling)
    
    def make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if (stride != 1) or (self.in_channels != out_channels):
            downsample = nn.Sequential(
                conv3x3(self.in_channels, out_channels, stride=stride),
                nn.InstanceNorm2d(out_channels))
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for i in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x, n_points=15000, regular_sampling=False):
        ### Image Encoder ###
        z = self.conv(x)
        z = self.bn(z)
        z = self.relu(z)
        z = self.layer1(z)
        z = self.layer2(z)
        z = self.layer3(z)
        z = self.layer4(z)
        z = self.layer5(z)
        z = self.layer6(z)
        z = self.layer7(z)
        z = self.adpool(z).reshape(-1, 1024)
        return self.decode(z, n_points, regular_sampling)