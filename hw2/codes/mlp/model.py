# -*- coding: utf-8 -*-

import torch
from torch import nn
from torch.nn import init
from torch.nn.parameter import Parameter


class BatchNorm1d(nn.Module):
    # TODO START
    # Reference: https://discuss.pytorch.org/t/implementing-batchnorm-in-pytorch-problem-with-updating-self-running-mean-and-self-running-var/49314
    # Reference: https://github.com/ptrblck/pytorch_misc/blob/master/batch_norm_manual.py
    # Reference: https://wiseodd.github.io/techblog/2016/07/04/batchnorm/
    def __init__(self, num_features, eps=1e-10, momentum=0.1):
        super(BatchNorm1d, self).__init__()
        self.eps = eps
        # use exponential moving average
        self.momentum = momentum
        self.num_features = num_features
        # Parameters
        self.weight = nn.Parameter(torch.ones(num_features,
                                              requires_grad=True))
        self.bias = nn.Parameter(torch.zeros(num_features, requires_grad=True))
        # Store the average mean and variance
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
        # Initialize your parameter

    def forward(self, input):
        # input: [batch_size, num_feature_map * height * width]
        mean, var = None, None
        batch_size = input.size(0)
        if not self.training and batch_size == 1:
            mean = self.running_mean
            var = self.running_var
        else:
            mean = torch.mean(input, dim=0, keepdim=False)
            var = torch.var(input, dim=0, keepdim=False)
            # Important! Without no_grad will cause memory leaks.
            with torch.no_grad():
                self.running_mean = self.momentum * mean + (
                    1 - self.momentum) * self.running_mean
                self.running_var = self.momentum * var * batch_size / (
                    batch_size - 1) + (1 - self.momentum) * self.running_var
        input = (input - mean) / (var + self.eps).sqrt()
        input = self.weight * input + self.bias
        return input

    # TODO END


class Dropout(nn.Module):
    # TODO START
    def __init__(self, p=0.5):
        super(Dropout, self).__init__()
        self.p = p

    def forward(self, input):
        # input: [batch_size, num_feature_map * height * width]
        if self.training:
            return torch.bernoulli(torch.ones_like(input) * (1 - self.p)).to(
                input.get_device()) * input / (1 - self.p)
        else:
            return input

    # TODO END


class Model(nn.Module):
    def __init__(self, batch_norm=True, drop_rate=0.5, num_features=32 * 32 * 3, hidden=256):
        super(Model, self).__init__()
        # TODO START
        # Define your layers here
        self.batch_norm = batch_norm
        self.fc1 = nn.Linear(num_features, hidden)
        if self.batch_norm:
            self.bn1 = BatchNorm1d(num_features=hidden)
        # self.bn1 = nn.BatchNorm1d(num_features=hidden)
        self.act = nn.ReLU()
        self.dropout = Dropout(drop_rate)
        # self.dropout = nn.Dropout(drop_rate)
        self.fc2 = nn.Linear(hidden, 10)
        # TODO END
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x, y=None):
        # TODO START
        # the 10-class prediction output is named as "logits"
        x = self.fc1(x)
        if self.batch_norm:
            x = self.bn1(x)
        x = self.act(x)
        x = self.dropout(x)
        logits = self.fc2(x)
        # TODO END

        pred = torch.argmax(logits, 1)  # Calculate the prediction result
        if y is None:
            return pred
        loss = self.loss(logits, y)
        correct_pred = (pred.int() == y.int())
        acc = torch.mean(
            correct_pred.float())  # Calculate the accuracy in this mini-batch

        return loss, acc
