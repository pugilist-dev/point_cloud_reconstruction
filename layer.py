#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 22 15:49:40 2021

@author: raj
"""

import math

import torch
import torch.nn as nn

import util


# Generate points on a box grid given generator parameters for each cell and the number of points
class PointCloudGenerator(nn.Module):
    def __init__(self, generator, rnd_dim=2, res=16):
        super(self.__class__, self).__init__()

        self.base_dim = rnd_dim
        self.generator = generator

        grid = util.meshgrid(res)
        self.o = (((grid + 0.5) / res) - 0.5).view(3, -1)
        self.s = res

    def forward(self, x, dens, n_points):
        b, c, g, _, _ = x.shape
        self.o = self.o.to(x.device)

        # Sample Density
        n = util.densSample(dens, n_points)

        # We call self.generator with the corresponding box descriptor and 2 random features for each point in the cell
        # The output is then offset to the correct position in the grid
        # this function is only efficient if the maximum number of points per grid cell is small

        n = n.view(b, -1)
        x = x.view(b, c, -1)
        gen_inp = []
        gen_off = []
        for i in range(b):
            indices = []  # of cells, inserted as many times as number of wanted points
            for j in range(1, n[i].max() + 1):
                ind = (n[i] >= j).nonzero().squeeze(-1)
                indices.append(ind)
            indices = torch.cat(indices)

            x_ind = x[i, :, indices]
            o_ind = self.o[:, indices]
            b_rnd = torch.rand(self.base_dim, n_points).to(x_ind.device) * 2.0 - 1.0
            b_inp = torch.cat([x_ind, b_rnd], dim=0)
            gen_inp.append(b_inp)
            gen_off.append(o_ind)
        gen_inp = torch.stack(gen_inp)
        gen_off = torch.stack(gen_off)

        out = self.generator(gen_inp)
        norm = out.norm(dim=1)
        reg = (norm - (math.sqrt(3) / self.s)).clamp(0)  # twice the size needed to cover a grid-cell

        return out + gen_off, reg

    def forward_fixed_pattern(self, x, dens, n):
        b, c, g, _, _ = x.shape
        self.o = self.o.to(x.device)

        N = util.densSample(dens, n)

        # We call self.generator with the corresponding box descriptor and 2 random features for each point in the cell
        # The output is then offset to the correct position in the grid
        # this function is only efficient if the maximum number of points per grid cell is small

        N = N.view(b, -1)
        x = x.view(b, c, -1)
        gen_inp = []
        gen_off = []
        for i in range(b):
            batch_inp = []
            batch_off = []
            for j in range(1, N.max() + 1):
                ind = (N[i] == j).nonzero().squeeze(-1)
                if ind.shape[0] is not 0:
                    x_ind = x[i, :, ind].repeat([1, j])
                    o_ind = self.o[:, ind].repeat([1, j])
                    b_rnd = util.fixed_sample(j, ind.shape[0]).to(x_ind) * 2.0 - 1.0
                    b_inp = torch.cat([x_ind, b_rnd], dim=0)
                    batch_inp.append(b_inp)
                    batch_off.append(o_ind)
            gen_inp.append(torch.cat(batch_inp, dim=1))
            gen_off.append(torch.cat(batch_off, dim=1))
        gen_inp = torch.stack(gen_inp)
        gen_off = torch.stack(gen_off)

        out = self.generator(gen_inp)
        norm = out.norm(dim=1)
        reg = (norm - (math.sqrt(3) / (self.s))).clamp(0)  # twice the size needed to cover a gridcell

        return out + gen_off, reg


class AdaptiveDecoder(nn.Module):
    def __init__(self, decoder, n_classes=None, max_layer=None):
        super(self.__class__, self).__init__()
        assert (isinstance(decoder, nn.ModuleList))

        self.decoder = decoder

        self.slices = []
        self.norm_indices = []
        self.conditional = n_classes is not None

        first = True
        for i, l in enumerate(self.decoder):
            if isinstance(l, nn.InstanceNorm3d):
                if first:
                    if self.conditional:
                        self.inp = nn.Linear(n_classes, l.num_features * 2 * 2 * 2)
                    else:
                        self.inp = nn.Parameter(torch.randn([1, l.num_features, 2, 2, 2]))
                    first = False
                self.norm_indices.append(i)
                self.slices.append(l.num_features * 2)

        if max_layer is None:
            self.max_layer = len(self.norm_indices)
        else:
            self.max_layer = max_layer

    def forward(self, w, cls=None):
        size = 0
        j = 0
        b = w.shape[0]
        if self.conditional:
            x = self.inp(cls).view(b, -1, 2, 2, 2)  # in case of condition cls is expected to be a one-hot vector
        else:
            x = self.inp.repeat([b, 1, 1, 1, 1])
        for i, l in enumerate(self.decoder):
            x = l(x)
            if j < self.max_layer and i == self.norm_indices[j]:
                s = w[:, size:size + self.slices[j], None, None, None]
                size += self.slices[j]

                x = x * s[:, :self.slices[j] // 2]
                x = x + s[:, self.slices[j] // 2:]
                j += 1

        return x
