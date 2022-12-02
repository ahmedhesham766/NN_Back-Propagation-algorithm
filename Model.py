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

        if activation_function == Activation.sigmoid:
            self.activation_function = sigmoid
        elif activation_function == Activation.tanh:
            self.activation_function = tanh
        self.eta = eta
        self.epoch = epoch

        self.train_data, self.test_data = train_test_split(data)
        self.train_data = self.train_data.sample(frac=1).reset_index(drop=True)
        self.test_data = self.test_data.sample(frac=1).reset_index(drop=True)
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
            # for i in range(num_layers):
            #     num_of_neurons[i] += 1
        else:
            self.weights_arr.append(np.random.rand(3, num_of_neurons[num_layers - 1]))
        self.bias = bias

    def Train(self):
        acc = 0
        for i in range(self.epoch):
            for j in range(len(self.x_train)):
                f_nets = self.Forward_Feet(self.x_train.values[j])
                y = f_nets[-1]
                y_index = -1
                if y[0] > y[1] and y[0] > y[2]:
                    y_index = 0
                elif y[1] > y[0] and y[1] > y[2]:
                    y_index = 1
                elif y[2] > y[1] and y[2] > y[0]:
                    y_index = 2

                if self.y_train.values[j] == 0 and y_index == 0:
                    acc += 1
                elif self.y_train.values[j] == 0.5 and y_index == 1:
                    acc += 1
                elif self.y_train.values[j] == 1 and y_index == 2:
                    acc += 1

                sigmas = self.Backward_Feet(f_nets, self.y_train.values[j])
                self.Update_Weights(sigmas, f_nets, self.x_train.values[j])
            acc /= len(self.x_train)
            print(f'train accuracy in epoch {i} = {acc}')

    def Forward_Feet(self, row_data):
        f_values = []
        input = [row_data]
        for i in range(self.num_layers):
            layer_values = []
            for j in range(self.num_of_neurons[i]):
                net = np.transpose(self.weights_arr[i][j]).dot(input[i])
                value = self.activation_function(net)
                layer_values.append(value)
            f_values.append(layer_values)
            if self.bias:
                layer_values.insert(0, 1)
            input.append(layer_values)
        layer_values = []

        input = f_values[-1]

        for i in range(3):
            net = np.transpose(self.weights_arr[-1][i]).dot(input)
            value = self.activation_function(net)
            layer_values.append(value)
        f_values.append(layer_values)
        return f_values

    def Backward_Feet(self, f_nets, y):
        output = [0, 0, 0]
        if y == 0:
            output = [1, 0, 0]
        elif y == 0.5:
            output = [0, 1, 0]
        elif y == 1:
            output = [0, 0, 1]

        sigma_arr = []
        sigma = []
        for i in range(3):
            sigma.append((output[i] - f_nets[-1][i]) * self.Gradient(f_nets[-1][i]))
        sigma_arr.insert(0, sigma)

        for i in reversed(range(self.num_layers)):
            sigma = []
            for j in range(self.num_of_neurons[i]):
                s = 0
                for k in range(len(sigma_arr[0])):
                    s += self.weights_arr[i + 1][k][j] * sigma_arr[0][k]
                sigma.append(s * self.Gradient(f_nets[i][j]))
            sigma_arr.insert(0, sigma)
        return sigma_arr

    def Update_Weights(self, sigmas, f_nets, row_data):
        f_nets.insert(0, row_data)
        for i in range(len(self.weights_arr)):
            for j in range(len(self.weights_arr[i])):
                for k in range(len(self.weights_arr[i][j])):
                    self.weights_arr[i][j][k] += self.eta * sigmas[i][j] * f_nets[i][k]

    def Gradient(self, x):
        if self.activation_function == sigmoid:
            return x * (1 - x)
        else:
            return 1 - (x * x)

    def Test(self):
        accuracy = 0
        confusion_matrix = [
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
        ]
        y_index = -1
        t_index = -1
        for i in range(len(self.x_test)):
            f_nets = self.Forward_Feet(self.x_train.values[i])
            y = f_nets[-1]
            t = [0, 0, 0]

            if y[0] > y[1] and y[0] > y[2]:
                y = [1, 0, 0]
                y_index = 0
            elif y[1] > y[0] and y[1] > y[2]:
                y = [0, 1, 0]
                y_index = 1
            elif y[2] > y[1] and y[2] > y[0]:
                y = [0, 0, 1]
                y_index = 2

            if self.y_test.values[i] == 0:
                t = [1, 0, 0]
                t_index = 0
            elif self.y_test.values[i] == 0.5:
                t = [0, 1, 0]
                t_index = 1
            elif self.y_test.values[i] == 1:
                t = [0, 0, 1]
                t_index = 2

            if t == y:
                accuracy += 1

            confusion_matrix[y_index][t_index] += 1

        # calculate testing accuracy
        accuracy = (accuracy / len(self.x_test)) * 100

        print(f'Testing Accuracy: {accuracy:.2f}%')

        # for i in range(3):
        #     if tp[i] + fp[i] != 0:
        #         precision = tp[i] / (tp[i] + fp[i])
        #         recall = tp[i] / (tp[i] + fn[i])
        #     else:
        #         precision = 0
        #         recall = 0
        #     print(f'Testing Precision: {precision * 100:.2f}%')
        #     print(f'Testing Recall: {recall * 100:.2f}%')
        print(f'Confusion Matrix:')
        print(f'\t\tone\t|\ttwo\t|\tthree')
        print(f'one:\t{confusion_matrix[0][0]}\t|\t{confusion_matrix[0][1]}\t|\t{confusion_matrix[0][2]}\t')
        print(f'two:\t{confusion_matrix[1][0]}\t|\t{confusion_matrix[1][1]}\t|\t{confusion_matrix[1][2]}\t')
        print(f'three:\t{confusion_matrix[2][0]}\t|\t{confusion_matrix[2][1]}\t|\t{confusion_matrix[2][2]}\t')
        print('--------------------------------')
    # def train(self):
    #     x = self.x_train
    #     y = self.y_train
    #     for i in range(self.epoch):
    #         for j in range(len(x)):
    #             # forward feed
    #             f_nets = self.Forward_Feed(j, x)
    #             # print(f_arr)
    #             # backward feed
    #             out = []
    #             if y.values[j] == 0:
    #                 out = [1, 0, 0]
    #             elif y.values[j] == 0.5:
    #                 out = [0, 1, 0]
    #             elif y.values[j] == 1:
    #                 out = [0, 0, 1]
    #
    #             self.Backward_Feed(f_nets, out)

    # def Backward_Feed(self, f_nets, out):
    #     sigma_arr = []
    #     for k in reversed(range(self.num_layers+1)):
    #         if k == self.num_layers:
    #             sigma = []
    #             for l in range(3):
    #                 temp = f_nets[len(f_nets) - 1][l]
    #                 sigma.append((out[l] - temp) * temp * (1 - temp))
    #             sigma_arr.append(sigma)  # arr of sigmas adds the output
    #         else:
    #             sigma = []
    #             for m in range(len(f_nets[k])):
    #                 print(self.weights_arr[k][m])
    #                 s = np.transpose(sigma_arr[0]).dot(self.weights_arr[k][m])
    #                 # s = 0
    #                 # for l in range(len(sigma_arr[k - 1])):
    #                 #     s += sigma_arr[0][l] * self.weights_arr[k][l]
    #                 sigma.append(s * f_nets[k][m])
    #             sigma_arr.insert(0, sigma)
    #     return sigma_arr

    # def Forward_Feed(self, j, x):
    #     f_arr = []  # f_arr[0] the result of the activation fun for all the first layer
    #     # f_arr[0][0] the result of the activation fun for the first neuron of the first layer
    #     iteration_data = [x.values[j]]
    #     for i in range(self.num_layers + 1):
    #         f_layer = []
    #         if i != self.num_layers:  # not output layer
    #             for j in range(self.num_of_neurons[i]):
    #                 # print(self.weights_arr[k][l])
    #                 if self.activation_function == Activation.sigmoid:
    #                     f_layer.append(sigmoid(np.transpose(self.weights_arr[i][j]).dot(iteration_data[i])))
    #                 elif self.activation_function == Activation.tanh:
    #                     f_layer.append(tanh(np.transpose(self.weights_arr[i][j]).dot(iteration_data[i])))
    #         else:
    #             for j in range(3):
    #                 if self.activation_function == Activation.sigmoid:
    #                     f_layer.append(sigmoid(np.transpose(self.weights_arr[i][j]).dot(iteration_data[i])))
    #                 elif self.activation_function == Activation.tanh:
    #                     f_layer.append(tanh(np.transpose(self.weights_arr[i][j]).dot(iteration_data[i])))
    #
    #         # print(f_layer)
    #         f_arr.append(f_layer)
    #         iteration_data.append(f_layer)
    #     return f_arr


data = preprocess(dataset)
m = DeepModel(data=data, num_layers=2, num_of_neurons=[3, 4], eta=0.1,
              epoch=1000, activation_function=Activation.sigmoid, bias=True)
m.Train()
m.Test()
# m.train()
# import cv2
# train_data = pd.read_csv('mnist_train.csv')
# img = train_data.iloc[0][1:]
# img = np.asarray(img.values)
# img = img.reshape(28, 28)
# plt.imshow(img)
# plt.show()
