# -*- coding: utf-8 -*-

import torch
from torch import nn
from torch.nn import init
from torch.nn.parameter import Parameter


class BatchNorm2d(nn.Module):
    # TODO START
    # Reference: https://github.com/ptrblck/pytorch_misc/blob/master/batch_norm_manual.py
    def __init__(self, num_features, eps=1e-10, momentum=0.1):
        super(BatchNorm2d, self).__init__()
        self.num_features = num_features
        self.momentum = momentum
        self.eps = eps
        # Parameters
        self.weight = nn.Parameter(torch.ones(num_features,
                                              requires_grad=True))
        self.bias = nn.Parameter(torch.zeros(num_features, requires_grad=True))

        # Store the average mean and variance
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))

        # Initialize your parameter

    def forward(self, input):
        # input: [batch_size, num_feature_map, height, width]
        batch_size, num_feature_map, height, width = input.shape
        mean, var = None, None
        if not self.training and batch_size == 1:
            mean = self.running_mean
            var = self.running_var
        else:
            mean = input.mean([0, 2, 3])
            var = input.var([0, 2, 3], unbiased=False)
            n = batch_size * height * width
            # Important! Without no_grad will cause memory leaks.
            with torch.no_grad():
                self.running_mean = self.momentum * mean + (
                    1 - self.momentum) * self.running_mean
                self.running_var = self.momentum * var * n / (
                    n - 1) + (1 - self.momentum) * self.running_var
        input = (input - mean[None, :, None, None]) / \
            (var[None, :, None, None] + self.eps).sqrt()
        input = input * self.weight[None, :, None,
                                    None] + self.bias[None, :, None, None]
        return input
    # TODO END


class Dropout(nn.Module):
    # TODO START
    def __init__(self, p=0.5):
        super(Dropout, self).__init__()
        self.p = p

    def forward(self, input):
        # input: [batch_size, num_feature_map, height, width]
        batch_size, num_feature_map, height, width = input.shape
        if self.training:
            return torch.bernoulli(torch.ones((batch_size, 1, height, width)) *
                                   (1 - self.p)).to(
                input.get_device()) * input / (1 - self.p)
        else:
            return input
    # TODO END


class Model(nn.Module):
    def __init__(self, batch_norm=True, drop_rate=0.5, h=32, w=32):
        super(Model, self).__init__()
        # TODO START
        # Define your layers here
        kernel = [5, 3]
        channel = [100, 60]
        if batch_norm:
            self.conv_layers = nn.Sequential(
                nn.Conv2d(
                    in_channels=3, out_channels=channel[0], kernel_size=kernel[0]),
                # nn.BatchNorm2d(channel[0]),
                BatchNorm2d(channel[0]),
                nn.ReLU(),
                # nn.Dropout2d(drop_rate),
                Dropout(drop_rate),
                nn.MaxPool2d(2),
                nn.Conv2d(in_channels=channel[0], out_channels=channel[1],
                          kernel_size=kernel[1]),
                # nn.BatchNorm2d(channel[1]),
                BatchNorm2d(channel[1]),
                nn.ReLU(),
                # nn.Dropout2d(drop_rate),
                Dropout(drop_rate),
                nn.MaxPool2d(2)
            )
        else:
            self.conv_layers = nn.Sequential(
                nn.Conv2d(
                    in_channels=3, out_channels=channel[0], kernel_size=kernel[0]),
                nn.ReLU(),
                # nn.Dropout2d(drop_rate),
                Dropout(drop_rate),
                nn.MaxPool2d(2),
                nn.Conv2d(in_channels=channel[0], out_channels=channel[1],
                          kernel_size=kernel[1]),
                nn.ReLU(),
                # nn.Dropout2d(drop_rate),
                Dropout(drop_rate),
                nn.MaxPool2d(2)
            )
        self.fc = nn.Linear(
            channel[1]*(((h-kernel[0]+1)//2-kernel[1]+1)//2) *
            (((w-kernel[0]+1)//2-kernel[1]+1)//2), 10)
        # TODO END
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x, y=None):
        # TODO START
        # the 10-class prediction output is named as "logits"
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        logits = self.fc(x)
        # TODO END
        pred = torch.argmax(logits, 1)  # Calculate the prediction result
        if y is None:
            return pred
        loss = self.loss(logits, y)
        correct_pred = (pred.int() == y.int())
        # Calculate the accuracy in this mini-batch
        acc = torch.mean(correct_pred.float())

        return loss, acc
