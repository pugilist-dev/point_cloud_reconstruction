#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 18 14:28:54 2021

@author: raj
"""
import argparse
import os
import pickle
import random

# Deep Learning packages
import numpy as np
import torch
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from tqdm import tqdm
import pandas as pd
from torchvision import transforms

# Util packages
import data
import dists
import models
import util
from data import get_firetower_pairs, firetower

def apply_net(network, dataloader, optimizer, device, opt, flag = "train"):
    chamfer_sum = 0
    p_norm_sum = 0
    density_sum = 0
    dense_cls_sum = 0
    reg_sum = 0
    sample_sum = 0
    loader = tqdm(dataloader, dynamic_ncols=True)
    for batch_data in loader:
        ### Start the data slicing here ###
        img = batch_data[0]
        img = img.to(device)
        target = batch_data[1]
        target = target.to(device)
        targetdens = util.densCalc(target, 32)
        ### Data slicing ends here ###
        
        
        ### Apply the network here ###
        if not opt.no_cls:
            mask = (targetdens > 0).squeeze()
            pos_exp = mask.view(target.shape[0], -1).float().sum(dim=-1)
            neg_exp = targetdens.shape[-1] ** 3 - pos_exp
            pos_weight = neg_exp / pos_exp
            weight = torch.ones(mask.squeeze().shape).float().to(device)
            for i in range(weight.shape[0]):
                weight[i, mask[i]] = pos_weight[i]
                
        pred, dens, dens_cls, reg = network(img, n_points=opt.n_out_points)
        if opt.no_cls:
            dense_cls_loss = torch.zeros(1).to(pred)
        else:
            dense_cls_loss = F.binary_cross_entropy_with_logits(dens_cls,
                                                                mask.float(),
                                                                weight=weight,
                                                                reduction='none').mean() * 100
        chamfer_loss = dists.chamfer(pred, target).sum() * 1e3
        if opt.p_norm >= 0:
            p_loss = dists.dist_norm(pred, target, p=opt.p_norm).sum() * 1e1
        else:
            p_loss = torch.zeros(1).to(pred)

        density_loss = F.mse_loss(dens, targetdens, reduction='mean') * 1e10

        if opt.no_dist_reg:
            dist_regularization = torch.zeros(1).to(pred)
        else:
            dist_regularization = reg.sum()

        loss = chamfer_loss + density_loss + dense_cls_loss + dist_regularization + p_loss

        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        reg_sum += dist_regularization.item()
        chamfer_sum += chamfer_loss.item()
        p_norm_sum += p_loss.item()
        density_sum += density_loss.item()
        dense_cls_sum += dense_cls_loss.item()
        sample_sum += img.shape[0]

        if optimizer is not None:
            prefix = "(Training) "
        else:
            prefix = "(Validation) "
        loader.set_description(
            prefix + "Chamfer Loss: {:.{prec}f} P Norm Loss: {:.{prec}f} Density Loss: {:.{prec}f}  Density BCE {:.{prec}f} Regularization: {:.{prec}f}".format(
                chamfer_sum / float(sample_sum), p_norm_sum / float(sample_sum), density_sum / float(sample_sum),
                dense_cls_sum / float(sample_sum), reg_sum / float(sample_sum), prec=5))

    chamfer_avg = float(chamfer_sum) / float(len(dataloader.dataset))
    p_norm_avg = float(p_norm_sum) / float(len(dataloader.dataset))
    density_avg = float(density_sum) / float(len(dataloader.dataset))
    dense_cls_avg = dense_cls_sum / float(len(dataloader.dataset))
    reg_avg = reg_sum / float(len(dataloader.dataset))

    return chamfer_avg, p_norm_avg, density_avg, dense_cls_avg, reg_avg, pred.detach().cpu()

def train_net(net, dataloader, optimizer, device, opt):
    net.train()
    return apply_net(net, dataloader, optimizer, device, opt)

def eval_net(net, dataloader, device, opt, flag="train"):
    net.eval()
    with torch.no_grad():
        return apply_net(net, dataloader, None, device, opt, flag=flag)

def train(opt):
    if not os.path.exists(os.path.join("models", opt.run_name)):
        os.makedirs(os.path.join("models", opt.run_name))
    with open(os.path.join("models", opt.run_name, 'opt.pkl'), 'wb') as f:
        pickle.dump(opt, f)
    
    img_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
        ])
    pc_transform = transforms.Compose([
            transforms.ToTensor()
        ])
    
    images, point_cloud = get_firetower_pairs(opt.data_path, mode="train")
    images, pc = pd.DataFrame(sorted(images)), pd.DataFrame(sorted(point_cloud))
    train_data = firetower(
        args = opt,
        images=images,
        pc = pc,
        mode="train",
        img_transform = img_transform,
        pc_transform = pc_transform)

    if opt.restrict_class > -1:
        train_data.restrict_class([opt.restrict_class])
    
    val_images, val_pc = get_firetower_pairs(opt.data_path, mode="val")
    val_images, val_pc = pd.DataFrame(sorted(val_images)), pd.DataFrame(sorted(val_pc))
    val_data = firetower(
        args = opt,
        images=val_images,
        pc = val_pc,
        mode="val",
        img_transform = img_transform,
        pc_transform = pc_transform
        )

    if opt.restrict_class > -1:
        val_data.restrict_class([opt.restrict_class])

    train_dataloader = torch.utils.data.DataLoader(train_data,
                                                   batch_size=opt.batch_size,
                                                   shuffle=True,
                                                   drop_last=True)

    val_dataloader = torch.utils.data.DataLoader(val_data,
                                                 batch_size=opt.batch_size,
                                                 shuffle=False,
                                                 drop_last=False)

    device = torch.device('cpu') if opt.no_cuda else torch.device('cuda')

    net = models.generative_net(args.resolution, in_nchannel=512,
                                      rnd_dim=2,
                                      enc_p=opt.enc_dropout,
                                      dec_p=opt.dec_dropout,
                                      adain_layer=opt.adain_layer).to(device)
    
    if torch.cuda.device_count() > 1:
            net = torch.nn.DataParallel(net, device_ids=[0, 1, 2, 3])
    net.to(args.device)

    writer = SummaryWriter('runs/{}'.format(opt.run_name))

    total_params = 0
    for param in net.parameters():
        total_params += np.prod(param.size())
    print("Network parameters: {}".format(total_params))

    optimizer = torch.optim.Adam(net.parameters(), lr=opt.lr, amsgrad=True)

    best_loss_train = float("inf")
    best_loss_val = float("inf")

    for epoch in range(opt.n_epochs):
        # Training
        chamfer_loss, p_norm_loss, density_loss, \
        density_bce_loss, reg_loss, preds = train_net(net, train_dataloader, optimizer, device, opt)
        val_chamfer_loss, val_p_norm_loss, val_density_loss, \
        val_density_bce_loss, val_reg_loss, val_preds = eval_net(net, val_dataloader, device, opt, flag="val")

        # Saving
        try:
            writer.add_scalars('chamfer', {'train': chamfer_loss,
                                           'validation': val_chamfer_loss}, epoch)
            writer.add_scalars('p_norm', {'train': p_norm_loss,
                                          'validation': val_p_norm_loss}, epoch)
            writer.add_scalars('density', {'train': density_loss,
                                           'validation': val_density_loss}, epoch)
            writer.add_scalars('density_bce', {'train': density_bce_loss,
                                               'validation': val_density_bce_loss}, epoch)
            writer.add_scalars('regularization', {'train': reg_loss,
                                                  'validation': val_reg_loss}, epoch)

            preds_path = os.path.join("models", opt.run_name, "preds_last.torch")
            torch.save(preds, preds_path)
        except PermissionError:
            pass

        if chamfer_loss < best_loss_train:
            best_loss_train = chamfer_loss
            try:
                model_path = os.path.join("models", opt.run_name, "model_train.state")
                optim_path = os.path.join("models", opt.run_name, "optim_train.state")
                torch.save(net.state_dict(), model_path)
                torch.save(optimizer.state_dict(), optim_path)
                with open(os.path.join("models", opt.run_name, 'opt.pkl'), 'wb') as f:
                    pickle.dump(opt, f)
                print('Epoch {}: {}'.format(epoch, model_path))
            except PermissionError:
                pass

        if val_chamfer_loss < best_loss_val:
            best_loss_val = val_chamfer_loss
            try:
                model_path = os.path.join("models", opt.run_name, "model_val.state")
                optim_path = os.path.join("models", opt.run_name, "optim_val.state")
                torch.save(net.state_dict(), model_path)
                torch.save(optimizer.state_dict(), optim_path)
                print('Epoch {}: {}'.format(epoch, model_path))
            except PermissionError:
                pass

    print('\ndone')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--resolution", type=int, default=128)
    parser.add_argument('--run-name', default='full')
    parser.add_argument('--data-path', default = "/home/raj/ft_im_to_pc/data/interim/")
    parser.add_argument('--restrict-class', type=int, default=-1)
    parser.add_argument('--n-epochs', type=int, default=10000)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--base_size', type=int, default=2048,
                        help='Base input image size')
    parser.add_argument('--crop_size', type=int, default=256,
                        help='cropping input image size')
    parser.add_argument('--lr', type=float, default=0.0046)
    parser.add_argument('--n-in-points', type=int, default=15000)
    parser.add_argument('--n-out-points', type=int, default=15000)
    parser.add_argument('--enc-dropout', type=float, default=0)
    parser.add_argument('--dec-dropout', type=float, default=0.2)
    parser.add_argument('--adain-layer', type=int, default=None)
    parser.add_argument('--no-cls', action='store_true')
    parser.add_argument('--no-dist-reg', action='store_true')
    parser.add_argument('--p-norm', type=int, default=5)
    parser.add_argument('--random-sampling', action='store_true')
    parser.add_argument('--random-seed', type=int, default=42)
    parser.add_argument('--no-cuda', action='store_true')
    args = parser.parse_args()
    if args is not None:
        print(args)
        torch.manual_seed(args.random_seed)
        np.random.seed(args.random_seed)
        random.seed(args.random_seed)
        train(args)