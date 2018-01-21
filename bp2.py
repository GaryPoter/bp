# -*- coding:utf-8 -*-
# author: 何伟
# -*- coding:utf-8 -*-
# author: 何伟
import numpy as np
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_decision_regions

# input X and label y
X = np.loadtxt('X.txt')
y = np.loadtxt('y.txt', dtype='int')

# # Train the logistic rgeression classifier
#  －－－－－－－－－－－－－－－－－－－－－－－－－－－
#  BP
#  定义梯度下降一些有用的变量和参数
num_examples = len(X)  # training set size
nn_input_dim = len(X[0])  # input layer dimensionality
nn_output_dim = 2  # output layer dimensionality
epsilon = 0.01  # learning rate for gradient descent
reg_lambda = 0.01  # regularization strength


class TempModel:
    #  初始化各层权值矩阵W和偏置量b
    def __init__(self, *args):
        np.random.seed(0)
        a = [nn_input_dim]
        a.extend([dim for dim in args])
        a.append(nn_output_dim)
        self.n = len(a)
        self.W = []
        self.b = []
        self.dW = []
        self.db = []
        self.delta = []
        self.output = []
        self.input = []
        self.y_out = np.zeros((200, 2))
        for i in range(self.n - 1):
            t_W, t_b = self.get_mat(a[i], a[i + 1])
            self.W.append(t_W)
            self.b.append(t_b)
            self.db.append(0)
            self.dW.append(0)
            self.delta.append(0)
            self.output.append(0)
            self.input.append(0)

    #  构造一个row * column 的矩阵w 和 col维度的偏置量b
    def get_mat(self, row, col):
        return np.random.randn(row, col) / np.sqrt(row), np.zeros((1, col))

    #  x_input是输入, w是权值矩阵, b是该层的输出, f是阶跃函数,默认为tanh
    def forward_one_layer(self, x_input, w, b, f=np.tanh):
        v_input = np.dot(x_input, w) + b
        output = f(v_input)
        return v_input, output

    #  阶跃函数的反函数
    def inverse_step_function(self, x):
        return 1 - np.power(x, 2)

    #  阶跃函数
    def step_function(self, x):
        return 1 / (1 + np.exp(-x))

    #  得到隐层的delta
    # def get_delta(self, delta, W, y_last, f=inverse_step_function):
    #     return delta.dot(W.T) * f(self, y_last)

    #  得到隐层的delta
    def get_delta1(self, i, f=inverse_step_function):
        self.delta[i - 1] = self.delta[i].dot(self.W[i].T) * f(self, self.output[i - 1])

    # 隐层变化
    def back_in_layer(self, delta, W, y_last, y_ll):
        cur_delta = self.get_delta(delta, W, y_last)
        dW = np.dot(y_ll.T, cur_delta)
        db = np.sum(cur_delta, axis=0, keepdims=True)
        return cur_delta, dW, db

    def back_in_layer1(self, i):
        self.get_delta1(i)
        y_ll = X.transpose()
        if i >= 2:
            y_ll = self.output[i - 2].transpose()
        self.dW[i - 1] = np.dot(y_ll, self.delta[i - 1])
        self.db[i - 1] = np.sum(self.delta[i - 1], axis=0, keepdims=True)

    #  输出层变化
    def back_in_output_layer(self, delta, y_last):
        cur_delta = delta
        cur_delta[range(num_examples), y] -= 1
        #  正则化
        dW = np.dot(y_last.T, cur_delta) + reg_lambda * self.W[-1]
        db = np.sum(cur_delta, axis=0)
        return cur_delta, dW, db

    #  输出层变化
    def back_in_output_layer1(self):
        self.delta[-1] = self.y_out
        self.delta[-1][range(num_examples), y] -= 1
        #  正则化
        self.dW[-1] = np.dot(self.output[-2].T, self.delta[-1]) + reg_lambda * self.W[-1]
        self.db[-1] = np.sum(self.delta[-1], axis=0)

    #  正则化
    def regular_arg(self, *args):
        return

    #  计算误差
    def calculate_loss(self):
        self.forward_propagation(X)
        tmp = self.y_out[range(num_examples), y]
        corect_logprobs = -np.log(tmp)
        data_loss = np.sum(corect_logprobs)
        data_loss += reg_lambda / 2 * sum([np.sum(np.square(key))for key in self.W])
        return 1./num_examples * data_loss

    #  预测Ｘ的类别
    def predict(self, X):
        self.forward_propagation(X)
        return np.argmax(self.y_out, axis=1)


    #  反向传播链
    def back_propagation(self, delta):
        self.delta[-1], self.dW[-1], self.db[-1] = self.back_in_output_layer(delta, self.output[-2])
        n = self.n - 3
        for i in range(n):
            self.delta[n - i], self.dW[n - i], self.db[n - i] = self.back_in_layer(self.delta[n - i + 1],
                                                                                   self.W[n - i + 1],
                                                                                   y_last=self.output[n - i],
                                                                                   y_ll=self.output[n - i - 1])
        self.delta[0], self.dW[0], self.db[0] = self.back_in_layer(self.delta[1],
                                                                   self.W[1],
                                                                   y_last=self.output[0],
                                                                   y_ll=X)

        for i in range(self.n - 1):
            #  正则化
            self.dW[i] += reg_lambda * self.W[i]
            self.W[i] += -epsilon * self.dW[i]
            self.b[i] += -epsilon * self.db[i]

    def back_propagation1(self):
        #  输出层
        self.back_in_output_layer1()
        n = self.n - 3
        #  后续隐层
        for i in range(n + 1):
            self.back_in_layer1(n - i + 1)

        for i in range(self.n - 1):
            #  正则化
            self.dW[i] += reg_lambda * self.W[i]
            self.W[i] += -epsilon * self.dW[i]
            self.b[i] += -epsilon * self.db[i]

    #  x_input是输入, w是权值矩阵, b是该层的输出, f是阶跃函数,默认为tanh

    def forward_one_layer1(self, x_input, i, f=np.tanh):
        self.input[i] = np.dot(x_input, self.W[i]) + self.b[i]
        self.output[i] = f(self.input[i])

    def forward_propagation(self, X):
        tmp = X
        for i in range(self.n - 1):
            self.input[i], self.output[i] = self.forward_one_layer(tmp, self.W[i], self.b[i])
            tmp = self.output[i]
        #  使用softmax函数进行多类划分
        self.input[-1], self.output[-1] = self.forward_one_layer(self.output[-2], self.W[-1], self.b[-1], f=np.exp)
        # v_input4, output4 = self.forward_one_layer(output3, self.W[3], self.b[3], f=np.exp)
        self.y_out = self.output[-1] / np.sum(self.output[-1], axis=1, keepdims=True)

    def forward_propagation1(self, X):
        tmp = X
        for i in range(self.n - 1):
            self.forward_one_layer1(tmp, i)
            tmp = self.output[i]
        #  使用softmax函数进行多类划分
        self.forward_one_layer1(self.output[-2], -1, f=np.exp)
        # v_input4, output4 = self.forward_one_layer(output3, self.W[3], self.b[3], f=np.exp)
        self.y_out = self.output[-1] / np.sum(self.output[-1], axis=1, keepdims=True)


    #  建立模型
    def build_model(self, num_passes=20000, print_loss=False):
        for i in range(0, num_passes):
            # Forward propagation
            self.forward_propagation1(X)

            # Backpropagation
            self.back_propagation1()
            if print_loss and i % 1000 == 0:
                print("Loss after iteration %i: %f" % (i, self.calculate_loss()))
        for i in range(self.n - 1):
            np.savetxt('b' + str(i + 1) + '.txt', self.b[i])
            np.savetxt('W' + str(i + 1) + '.txt', self.W[i])

            
if __name__ == '__main__':
    model = TempModel(20, 10, 5)

    model.build_model(print_loss=True)

    # plot_decision_regions(X, y, model, legend=0)
    # plt.show()

