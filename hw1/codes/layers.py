import numpy as np
import math


class Layer(object):
    def __init__(self, name, trainable=False):
        self.name = name
        self.trainable = trainable
        self._saved_tensor = None

    def forward(self, input):
        pass

    def backward(self, grad_output):
        pass

    def update(self, config):
        pass

    def _saved_for_backward(self, tensor):
        '''The intermediate results computed during forward stage
        can be saved and reused for backward, for saving computation'''

        self._saved_tensor = tensor

    def __str__(self):
        return self.name


class Relu(Layer):
    def __init__(self, name):
        super(Relu, self).__init__(name)

    def forward(self, input):
        # TODO START
        self.input = input
        return self.relu(input)
        # TODO END

    def backward(self, grad_output):
        # TODO START
        return grad_output * self.relu_grad(self.input)
        # TODO END

    def relu(self, u):
        return np.maximum(u, 0)

    def relu_grad(self, u):
        return 1 * (u > 0)


class Sigmoid(Layer):
    def __init__(self, name):
        super(Sigmoid, self).__init__(name)

    def forward(self, input):
        # TODO START
        self.input = input
        return self.sigmoid(input)
        # TODO END

    def backward(self, grad_output):
        # TODO START
        return grad_output * self.sigmoid_grad(self.input)
        # TODO END

    def sigmoid(self, u):
        return 1 / (1 + np.exp(-u))

    def sigmoid_grad(self, u):
        return self.sigmoid(u) * (1 - self.sigmoid(u))


class Gelu(Layer):
    def __init__(self, name):
        super(Gelu, self).__init__(name)
        self.delta = 1e-5

    def forward(self, input):
        # TODO START
        self.input = input
        return self.gelu(input)
        # TODO END

    def backward(self, grad_output):
        # TODO START
        return grad_output * self.gelu_grad(self.input)
        # TODO END

    def gelu(self, u):
        return 0.5 * u * (
            1 + np.tanh(np.sqrt(2 / np.pi) * (u + 0.044715 * np.power(u, 3))))

    def gelu_grad(self, u):
        return (self.gelu(u + self.delta) -
                self.gelu(u - self.delta)) / (2 * self.delta)


class Linear(Layer):
    def __init__(self, name, in_num, out_num, init_std):
        super(Linear, self).__init__(name, trainable=True)
        self.in_num = in_num
        self.out_num = out_num
        self.W = np.random.randn(in_num, out_num) * init_std
        self.b = np.zeros(out_num)

        self.grad_W = np.zeros((in_num, out_num))
        self.grad_b = np.zeros(out_num)

        self.diff_W = np.zeros((in_num, out_num))
        self.diff_b = np.zeros(out_num)

    def forward(self, input):
        # TODO START
        # print(self.W)
        self.input = input
        return self.input @ self.W + self.b
        # TODO END

    def backward(self, grad_output):
        # TODO START
        batch_size = grad_output.shape[0]
        self.grad_b = grad_output
        self.grad_W = (self.input.T @ grad_output) / batch_size
        return grad_output @ self.W.T
        # TODO END

    def update(self, config):
        mm = config['momentum']
        lr = config['learning_rate']
        wd = config['weight_decay']
        self.diff_W = mm * self.diff_W + (self.grad_W + wd * self.W)
        self.W = self.W - lr * self.diff_W
        self.diff_b = mm * self.diff_b + (self.grad_b + wd * self.b)
        self.b = self.b - lr * self.diff_b

    def __str__(self):
        return "fc(%d-%d)" % (self.in_num, self.out_num)
