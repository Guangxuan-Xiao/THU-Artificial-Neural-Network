from utils import LOG_INFO, onehot_encoding, calculate_acc
import numpy as np


def data_iterator(x, y, batch_size, shuffle=True):
    indx = list(range(len(x)))
    if shuffle:
        np.random.shuffle(indx)

    for start_idx in range(0, len(x), batch_size):
        end_idx = min(start_idx + batch_size, len(x))
        yield x[indx[start_idx:end_idx]], y[indx[start_idx:end_idx]]


def train_net(model, loss, config, inputs, labels, batch_size, disp_freq):

    iter_counter = 0
    loss_list = []
    acc_list = []
    loss_temp_list = []
    acc_temp_list = []
    for input, label in data_iterator(inputs, labels, batch_size):
        target = onehot_encoding(label, 10)
        iter_counter += 1

        # forward net
        output = model.forward(input)
        # calculate loss
        loss_value = loss.forward(output, target)
        # generate gradient w.r.t loss
        grad = loss.backward(output, target)
        # backward gradient

        model.backward(grad)
        # update layers' weights
        model.update(config)

        acc_value = calculate_acc(output, label)
        loss_temp_list.append(loss_value)
        acc_temp_list.append(acc_value)

        if iter_counter % disp_freq == 0:
            loss_mean = np.mean(loss_temp_list)
            acc_mean = np.mean(acc_temp_list)
            msg = '  Training iter %d, batch loss %.4f, batch acc %.4f' % (
                iter_counter, loss_mean, acc_mean)
            loss_temp_list = []
            acc_temp_list = []
            LOG_INFO(msg)
            loss_list.append(loss_mean)
            acc_list.append(acc_mean)
    return np.mean(loss_list), np.mean(acc_list)


def test_net(model, loss, inputs, labels, batch_size):
    loss_list = []
    acc_list = []
    for input, label in data_iterator(inputs,
                                      labels,
                                      batch_size,
                                      shuffle=False):
        target = onehot_encoding(label, 10)
        output = model.forward(input)
        loss_value = loss.forward(output, target)
        acc_value = calculate_acc(output, label)
        loss_list.append(loss_value)
        acc_list.append(acc_value)
    loss_mean = np.mean(loss_list)
    acc_mean = np.mean(acc_list)
    msg = '    Testing, total mean loss %.5f, total acc %.5f' % (loss_mean,
                                                                 acc_mean)
    LOG_INFO(msg)
    return loss_mean, acc_mean
