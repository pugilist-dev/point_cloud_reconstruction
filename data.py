#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 18 14:33:53 2021

@author: raj
"""

import torch
from PIL import Image, ImageOps, ImageFilter
import open3d as o3d
import numpy as np
import os
import random

def get_firetower_pairs(path, mode = "train"):
    img_list = os.listdir(path+'/images/'+mode)
    for img in img_list:
        if not(img.endswith(".png")):
            img_list.remove(img)
    pc_list = os.listdir(path+'/point_cloud/'+mode)
    for pc in pc_list:
        if not(pc.endswith(".ply")):
            pc_list.remove(pc)
    return img_list, pc_list

class firetower(torch.utils.data.Dataset):
    def __init__(self, images, pc, args,  mode="train", img_transform=None,
                 pc_transform=None):
        super(firetower, self).__init__()
        self.mode = mode
        self.images = images.values.tolist()
        self.pc = pc.values.tolist()
        self.img_transform = img_transform
        self.pc_transform = pc_transform
        self.base_size = args.base_size
        self.crop_size = args.crop_size
        self.root = "/home/raj/ft_im_to_pc/data/interim/"
        assert (len(self.images) == len(self.pc))
        if len(self.images) == 0:
            raise RuntimeError("Found 0 images in subfolders of: " + self.root + "\n")
        if len(self.pc) == 0:
            raise RuntimeError("Found 0 point clouds in subfolders of: " + self.root + "\n")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        img = Image.open(self.root+"images/"+self.mode+"/"+
                         self.images[index][0]).convert('RGB')
        pcd = o3d.io.read_point_cloud(self.root+"point_cloud/"+
                                            self.mode+"/"+self.pc[index][0])
        
        points = np.asarray(pcd.points)
        pmax = points.max(0, keepdims=True)
        pmin = points.min(0, keepdims=True)
        pcd.points = o3d.utility.Vector3dVector(
            (points - pmin) / (pmax - pmin).max()
        )
        xyz = self.get_points(pcd)
        img = self.sync_transform(img)
        if self.img_transform:
            img = self.img_transform(img)
        xyz = self.pc_transform(xyz)
        xyz = xyz.float() 
        return (img, torch.reshape(xyz, (3,-1)))
            
    def sync_transform(self, img):
        # Random Mirror
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        crop_size = self.crop_size
        short_size = random.randint(int(self.base_size * 0.5), int(self.base_size * 2.0))
        w, h = img.size
        if h > w:
            ow = short_size
            oh = int(1.0 * h * ow / w)
        else:
            oh = short_size
            ow = int(1.0 * w * oh / h)
        img = img.resize((ow, oh), Image.BILINEAR)
        # pad crop
        if short_size < crop_size:
            padh = crop_size - oh if oh < crop_size else 0
            padw = crop_size - ow if ow < crop_size else 0
            img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
        # random crop crop_size
        w, h = img.size
        x1 = random.randint(0, w - crop_size)
        y1 = random.randint(0, h - crop_size)
        img = img.crop((x1, y1, x1 + crop_size, y1 + crop_size))
        if random.random() < 0.5:
            img = img.filter(ImageFilter.GaussianBlur(
                radius=random.random()))
        return img
    
    
    def get_points(self, pcd):
        pts = np.asarray(pcd.points)
        number_rows = pts.shape[0]
        random_indices = np.random.choice(number_rows, size=15000, replace=True)
        points = pts[random_indices, :]
        return points
        
        