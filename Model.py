import math

import numpy as np
import pandas as pd
from enum import Enum

dataset = pd.read_csv('penguins.csv')


def preprocess(data):
    data[data.columns[0]] = pd.Categorical(data[data.columns[0]],
                                           categories=['Adelie', 'Gentoo', 'Chinstrap']).codes

    data[data.columns[1]].fillna(inplace=True, value=data[data.columns[1]].mean())

    data[data.columns[2]].fillna(inplace=True, value=data[data.columns[2]].mean())
    data[data.columns[3]].fillna(inplace=True, value=data[data.columns[3]].mean())

    data[data.columns[4]].fillna(inplace=True, value='unknown')
    data[data.columns[4]] = pd.Categorical(data[data.columns[4]],
                                           categories=['unknown', 'male', 'female']).codes

    data[data.columns[5]].fillna(inplace=True, value=data[data.columns[5]].mean())

    data = (data - data.min()) / (data.max() - data.min())

    return data


def train_test_split(data):
    y1 = data[0:50]  # the start index of y1
    y2 = data[50:100]  # the start index of y2
    y3 = data[100:150]  # the start index of y3

    train_data = pd.concat([y1[0:30], y2[0:30], y3[0:30]]).sample(frac=1, random_state=1).reset_index(drop=True)
    test_data = pd.concat([y1[30:50], y2[30:50], y3[30:50]]).sample(frac=1, random_state=1).reset_index(drop=True)

    return train_data, test_data


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def tanh(x):
    return (1 - math.exp(-x)) / (1 + math.exp(-x))


class Activation(Enum):
    tanh = 0,
    sigmoid = 1,


class DeepModel:
    def __init__(self, data: pd.DataFrame, num_layers: int, num_of_neurons: list,
                 eta: float, epoch: int, activation_function: Activation, bias: bool):
        self.data = data
        self.num_layers = num_layers
        self.num_of_neurons = num_of_neurons

        self.activation_function = activation_function
        self.eta = eta
        self.epoch = epoch

        self.train_data, self.test_data = train_test_split(data)

        self.x_train = self.train_data[self.train_data.columns[1:]]
        self.y_train = self.train_data[self.train_data.columns[0]]

        self.x_test = self.test_data[self.test_data.columns[1:]]
        self.y_test = self.test_data[self.test_data.columns[0]]

        if bias:
            b = pd.DataFrame(np.ones(len(self.train_data)), columns=['bias'])
            self.x_train = pd.concat([b, self.x_train], axis=1)
            b = pd.DataFrame(np.ones(len(self.test_data)), columns=['bias'])
            self.x_test = pd.concat([b, self.x_test], axis=1)

        # input weights
        w = np.random.rand(num_of_neurons[0], len(self.x_train.columns))
        self.weights_arr = [w]

        # hidden layers weights
        for i in range(num_layers):
            if i + 1 >= num_layers:
                break
            d = num_of_neurons[i]
            if bias:
                d += 1
            w = np.random.rand(num_of_neurons[i + 1], d)
            self.weights_arr.append(w)

        # output weights
        if bias:
            self.weights_arr.append(np.random.rand(3, num_of_neurons[num_layers - 1] + 1))
        else:
            self.weights_arr.append(np.random.rand(3, num_of_neurons[num_layers - 1]))
        print(self.weights_arr)  # all the weight matrix
        # print(self.weights_arr[0])  # matrix of the first layer
        # print(self.weights_arr[0][2])  # weight vector of the third neuron

    def train(self):
        x = self.x_train
        y = self.y_train
        for i in range(self.epoch):
            for j in range(len(x)):
                # forward feed
                f_nets = self.Forward_Feed(j, x)
                # print(f_arr)
                # backward feed
                out = []
                if y.values[j] == 0:
                    out = [1, 0, 0]
                elif y.values[j] == 0.5:
                    out = [0, 1, 0]
                elif y.values[j] == 1:
                    out = [0, 0, 1]

                self.Backward_Feed(f_nets, out)

    def Backward_Feed(self, f_nets, out):
        sigma_arr = []
        for k in reversed(range(self.num_layers+1)):
            if k == self.num_layers:
                sigma = []
                for l in range(3):
                    temp = f_nets[len(f_nets) - 1][l]
                    sigma.append((out[l] - temp) * temp * (1 - temp))
                sigma_arr.append(sigma)  # arr of sigmas adds the output
            else:
                sigma = []
                for m in range(len(f_nets[k])):
                    print(self.weights_arr[k][m])
                    s = np.transpose(sigma_arr[0]).dot(self.weights_arr[k][m])
                    # s = 0
                    # for l in range(len(sigma_arr[k - 1])):
                    #     s += sigma_arr[0][l] * self.weights_arr[k][l]
                    sigma.append(s * f_nets[k][m])
                sigma_arr.insert(0, sigma)
        return sigma_arr

    def Forward_Feed(self, j, x):
        f_arr = []  # f_arr[0] the result of the activation fun for all the first layer
        # f_arr[0][0] the result of the activation fun for the first neuron of the first layer
        iteration_data = [x.values[j]]
        for i in range(self.num_layers + 1):
            f_layer = []
            if i != self.num_layers:  # not output layer
                for j in range(self.num_of_neurons[i]):
                    # print(self.weights_arr[k][l])
                    if self.activation_function == Activation.sigmoid:
                        f_layer.append(sigmoid(np.transpose(self.weights_arr[i][j]).dot(iteration_data[i])))
                    elif self.activation_function == Activation.tanh:
                        f_layer.append(tanh(np.transpose(self.weights_arr[i][j]).dot(iteration_data[i])))
            else:
                for j in range(3):
                    if self.activation_function == Activation.sigmoid:
                        f_layer.append(sigmoid(np.transpose(self.weights_arr[i][j]).dot(iteration_data[i])))
                    elif self.activation_function == Activation.tanh:
                        f_layer.append(tanh(np.transpose(self.weights_arr[i][j]).dot(iteration_data[i])))

            # print(f_layer)
            f_arr.append(f_layer)
            iteration_data.append(f_layer)
        return f_arr


data = preprocess(dataset)
m = DeepModel(data=data, num_layers=2, num_of_neurons=[4, 2], eta=0.001,
              epoch=10, activation_function=Activation.sigmoid, bias=False)
m.train()
# import cv2
# train_data = pd.read_csv('mnist_train.csv')
# img = train_data.iloc[0][1:]
# img = np.asarray(img.values)
# img = img.reshape(28, 28)
# plt.imshow(img)
# plt.show()
