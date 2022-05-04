#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 18 14:34:07 2021

@author: raj
"""

import torch


# input is expected in form (b x) c x n
def dist_mat_squared(x, y):
    x = x.float()
    y = y.float()
    assert x.dim() == 3 or x.dim() == 2

    xx = torch.sum(x ** 2, dim=-2).unsqueeze(-1)
    yy = torch.sum(y ** 2, dim=-2)
    if x.dim() == 3:
        yy = yy.unsqueeze(-2)
        dists = torch.bmm(x.transpose(2, 1), y)
    else:
        dists = torch.matmul(x.t(), y)
    dists *= -2
    dists += yy
    dists += xx

    return dists


def dist_norm_p(x, y, p=2):
    d = dist_mat_squared(x, y)
    if x.dim() == 2:
        dists_1 = (x - y[:, d.min(-1)[1]]).norm(dim=0, p=p)
        dists_2 = (x[:, d.min(-2)[1]] - y).norm(dim=0, p=p)
    else:  # dim is 3
        b_d_ind_1 = d.min(-1)[1]
        b_d_ind_2 = d.min(-2)[1]
        b_dists_1 = []
        b_dists_2 = []
        for i in range(b_d_ind_1.shape[0]):
            b_dists_1.append((x[i] - y[i, :, b_d_ind_1[i]]).norm(dim=0, p=p))
            b_dists_2.append((x[i, :, b_d_ind_2[i]] - y[i]).norm(dim=0, p=p))
        dists_1 = torch.stack(b_dists_1)
        dists_2 = torch.stack(b_dists_2)
    return dists_1, dists_2


def dist_norm(x, y, p=2, points_p=2):
    dists_1, dists_2 = dist_norm_p(x, y, points_p)
    return dists_1.norm(p=p, dim=-1) + dists_2.norm(p=p, dim=-1)


def chamfer(x, y, weights_x=None, weights_y=None):
    d = dist_mat_squared(x, y)
    dist1 = d.min(-1)[0]
    dist2 = d.min(-2)[0]
    if weights_x is not None:
        dist1 = dist1 * weights_x * x.shape[-1]
    if weights_y is not None:
        dist2 = dist2 * weights_y * y.shape[-1]
    return dist1.mean(-1) + dist2.mean(-1)
