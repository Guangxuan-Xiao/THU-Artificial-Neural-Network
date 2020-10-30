import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from network import Network
from utils import LOG_INFO
from layers import Relu, Sigmoid, Linear, Gelu
from loss import EuclideanLoss, SoftmaxCrossEntropyLoss, HingeLoss
from solve_net import train_net, test_net
from load_data import load_mnist_2d
import argparse
train_data, test_data, train_label, test_label = load_mnist_2d('../data')
import numpy as np
import time


def plot(epochs, train, test, label, file="plot.png"):
    plt.figure()
    plt.plot(epochs, train, label="Training")
    plt.plot(epochs, test, label="Testing")
    plt.xlabel("Epochs")
    plt.ylabel(label)
    plt.legend()
    plt.savefig(file)


# Your model defintion here
# You should explore different model architecture
from mlp2 import model

# loss = SoftmaxCrossEntropyLoss(name="CE")
# loss = EuclideanLoss(name='MSE')
loss = HingeLoss(name="Hinge5", threshold=5)


model_name = str(model) + "_" + loss.name
print(model_name)
# Training configuration
# You should adjust these hyperparameters
# NOTE: one iteration means model forward-backwards one batch of samples.
#       one epoch means model has gone through all the training samples.
#       'disp_freq' denotes number of iterations in one epoch to display information.

config = {
    'learning_rate': 0.05,
    'weight_decay': 0.0,
    'momentum': 0.0,
    'batch_size': 100,
    'max_epoch': 100,
    'disp_freq': 200,
    'test_epoch': 1
}

train_loss_list = []
train_acc_list = []
test_loss_list = []
test_acc_list = []
epochs = []
time_start = time.time()
for epoch in range(config['max_epoch']):
    LOG_INFO('Training @ %d epoch...' % (epoch))
    train_loss, train_acc = train_net(model, loss, config, train_data,
                                      train_label, config['batch_size'],
                                      config['disp_freq'])

    LOG_INFO('Testing @ %d epoch...' % (epoch))
    test_loss, test_acc = test_net(model, loss, test_data, test_label,
                                   config['batch_size'])
    epochs.append(epoch + 1)
    train_loss_list.append(train_loss)
    train_acc_list.append(train_acc)
    test_loss_list.append(test_loss)
    test_acc_list.append(test_acc)
time_end = time.time()
plot(epochs,
     train_loss_list,
     test_loss_list,
     "Loss",
     file="../plots/%s_loss.png" % model_name)
plot(epochs,
     train_acc_list,
     test_acc_list,
     "Accuracy",
     file="../plots/%s_acc.png" % model_name)

with open("../results/%s_result.txt" % model_name, "w+") as f:
    train_loss, train_acc = train_net(model, loss, config, train_data,
                                      train_label, config['batch_size'],
                                      config['disp_freq'])
    test_loss, test_acc = test_net(model, loss, test_data, test_label,
                                   config['batch_size'])
    print(model_name, file=f)
    print(config, file=f)
    print("Time cost: ", time_end - time_start, 's', file=f)
    print("\nFinal Train\nLoss: %f\nAcc: %f" % (train_loss, train_acc), file=f)
    print("\nFinal Test\nLoss: %f\nAcc: %f" % (test_loss, test_acc), file=f)
