# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 11:30:20 2023

@author: cakir
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicConv2d(nn.Module):
  def __init__(self, in_planes, out_planes, kernel_size = 3, stride = 1, padding = 0):
    super(BasicConv2d, self).__init__()
    self.conv2 = nn.Conv2d(in_planes, out_planes, kernel_size = kernel_size, stride = stride, padding = padding, bias = False)
    self.bn = nn.BatchNorm2d(out_planes)
    self.relu = nn.ReLU()

  def forward(self, x):
    x = self.relu(self.bn(self.conv2(x)))
    return x

class denseLayer(nn.Module):
  def __init__(self, in_planes, out_planes, repeat, stride = 1):
    super(denseLayer, self).__init__()
    self.inter_planes = out_planes * 4
    self.bn = nn.BatchNorm2d(in_planes)
    self.relu = nn.ReLU()
    self.basicConv = BasicConv2d(in_planes, self.inter_planes, kernel_size = 1, stride = stride)
    self.lastConv = nn.Conv2d(self.inter_planes, out_planes, kernel_size = 3, stride = stride, padding = 1)

  def forward(self, x):
    out = self.bn(x)
    out = self.relu(out)
    out = self.basicConv(out)
    out = self.lastConv(out)

    return torch.cat((out, x), 1)

class denseBlock(nn.Module):
  def __init__(self, in_planes, out_planes, repeat):
    super(denseBlock, self).__init__()

    self.denseblock1 = self._make_denseLayer(in_planes, out_planes, repeat)

  def _make_denseLayer(self, in_planes, out_planes, repeat, stride = 1):
    block = []
    for i in range(repeat):
      block.append(
          denseLayer(in_planes, out_planes, repeat, stride = 1)
      )
      in_planes += out_planes

    return nn.Sequential(*block)

  def forward(self, x):
    return self.denseblock1(x)

class transitionLayer(nn.Module):
  def __init__(self, in_planes, out_planes):
    super(transitionLayer, self).__init__()
    self.transition = nn.Sequential(
        nn.BatchNorm2d(in_planes),
        nn.ReLU(),
        nn.Conv2d(in_planes, out_planes, kernel_size = 1, stride = 1, bias = False),
        nn.AvgPool2d(2, 2)
        )
    
  def forward(self, x):
      return self.transition(x)
  
class denseNet(nn.Module):
  def __init__(self, num_classes = 1000):
    super(denseNet, self).__init__()

    self.blockPlanes = [64, 128, 256, 512]
    self.blockSizes = [6, 12, 24, 16]
    
    layer = list()

    layer.append(BasicConv2d(3, 64, kernel_size = 7, stride = 2, padding = 3))
    layer.append(nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1))

    for idx, plane in enumerate(self.blockPlanes):
        layer.append(denseBlock(plane, 32, self.blockSizes[idx]))
        if idx != 3:
          layer.append(transitionLayer(plane*4, plane*2))

    layer.append(nn.BatchNorm2d(1024))

    self.layers = nn.Sequential(*layer)

    if num_classes == 2:
        self.fc = nn.Linear(1024, 1)
        self.activation = nn.Sigmoid()
    else:
        self.fc = nn.Linear(1024, num_classes)
        self.activation = nn.Softmax(dim=1)

  def forward(self, x):
    out = self.layers(x)
    out = F.relu(out, inplace=True)
    out = F.adaptive_avg_pool2d(out, (1, 1))
    out = torch.flatten(out, 1)
    out = self.activation(self.fc(out))

    return out
