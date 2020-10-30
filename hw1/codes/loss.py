from __future__ import division
import numpy as np


class EuclideanLoss(object):
    def __init__(self, name):
        self.name = name

    def forward(self, input, target):
        # TODO START
        self.delta = input - target
        self.loss = np.mean(np.linalg.norm(self.delta, axis=1)**2) / 2
        return self.loss
        # TODO END

    def backward(self, input, target):
        # TODO START
        return self.delta
        # TODO END


class SoftmaxCrossEntropyLoss(object):
    def __init__(self, name):
        self.name = name
        self.loss = np.zeros(1, dtype="f")

    def forward(self, input, target):
        # TODO START
        input = self.softmax(input)
        self.loss = -np.mean(np.multiply(np.log(input), target))
        self.delta = input - target
        return self.loss
        # TODO END

    def backward(self, input, target):
        # TODO START
        return self.delta
        # TODO END

    def softmax(self, u):
        u -= np.max(u)
        return np.exp(u) / (np.sum(np.exp(u), 1).reshape(-1, 1))


class HingeLoss(object):
    def __init__(self, name, threshold=0.05):
        self.name = name
        self.threshold = threshold

    def forward(self, input, target):
        # TODO START
        self.loss = np.mean(
            np.sum(np.maximum(
                0, self.threshold + input - input[target == 1].reshape(-1, 1))
                   * (target == 0),
                   axis=1))
        return self.loss
        # TODO END

    def backward(self, input, target):
        # TODO START
        delta = np.zeros_like(input)
        delta[(target == 0)
              & (self.threshold - input[target == 1].reshape(-1, 1) +
                 input > 0)] = 1
        delta[(target == 1)] = -np.sum(delta, axis=1)
        return delta
        # TODO END
