## Copyright (C) 2019, Huan Zhang <huan@huan-zhang.com>
##                     Hongge Chen <chenhg@mit.edu>
##                     Chaowei Xiao <xiaocw@umich.edu>
## 
## This program is licenced under the BSD 2-Clause License,
## contained in the LICENCE file in this directory.
##

# from convex_adversarial import Dense, DenseSequential

import torch
import torch.nn as nn

from model_defs import Flatten

def IBP_large(in_ch, in_dim, linear_size=512): 
    model = nn.Sequential(
        nn.Conv2d(in_ch, 64, 3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(64, 64, 3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(64, 128, 3, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(128, 128, 3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(128, 128, 3, stride=1, padding=1),
        nn.ReLU(),
        Flatten(),
        nn.Linear((in_dim//2) * (in_dim//2) * 128, linear_size),
        nn.ReLU(),
        nn.Linear(linear_size,10)
    )
    return model

def IBP_4_layer(in_ch, in_dim, linear_size=512): 
    model = nn.Sequential(
        nn.Conv2d(in_ch, 64, 3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(64, 64, 3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(64, 128, 3, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(128, 128, 3, stride=1, padding=1),
        nn.ReLU(),
        Flatten(),
        nn.Linear((in_dim//2) * (in_dim//2) * 128, linear_size),
        nn.ReLU(),
        nn.Linear(linear_size,10)
    )
    return model


def IBP_3_layer(in_ch, in_dim, linear_size=512): 
    model = nn.Sequential(
        nn.Conv2d(in_ch, 64, 3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(64, 64, 3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(64, 128, 3, stride=2, padding=1),
        nn.ReLU(),
        Flatten(),
        nn.Linear((in_dim//2) * (in_dim//2) * 128, linear_size),
        nn.ReLU(),
        nn.Linear(linear_size,10)
    )
    return model

def IBP_2_layer(in_ch, in_dim, linear_size=512): 
    model = nn.Sequential(
        nn.Conv2d(in_ch, 64, 3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(64, 64, 3, stride=2, padding=1),
        nn.ReLU(),
        Flatten(),
        nn.Linear((in_dim//2) * (in_dim//2) * 64, linear_size),
        nn.ReLU(),
        nn.Linear(linear_size,10)
    )
    return model


def IBP_debug(in_ch, in_dim, linear_size=512): 
    model = nn.Sequential( 
        nn.Conv2d(1, 1, 3, stride=2, padding=1),
        nn.ReLU(), 
        nn.Conv2d(1, 1, 3, stride=2, padding=1),
        nn.ReLU(), 
        Flatten(),
        nn.Linear((in_dim//4) * (in_dim//4) * 1, 10), 
    )
    return model

if __name__ == '__main__':
    in_channel = 3
    in_dim = 32
    model_2 = IBP_2_layer(in_channel, in_dim, linear_size=512)
    model_3 = IBP_3_layer(in_channel, in_dim, linear_size=512)
    model_4 = IBP_4_layer(in_channel, in_dim, linear_size=512)
    model_5 = IBP_large(in_channel, in_dim, linear_size=512)

    x = torch.rand(5, in_channel, in_dim, in_dim)
    out2 = model_2(x)
    out3 = model_3(x)
    out4 = model_4(x)
    out5 = model_5(x)
    print(out2.shape)
    print(out3.shape)
    print(out4.shape)
    print(out5.shape)