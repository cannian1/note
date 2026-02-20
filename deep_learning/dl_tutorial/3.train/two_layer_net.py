import numpy as np
from common.functions import softmax, sigmoid, cross_entropy
from common.gradient import numerical_gradient

class TwoLayerNet:
    # 初始化
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        """

        :param input_size:
        :param hidden_size:
        :param output_size:
        :param weight_init_std: 标准差权重
        """
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)
    # 前向传播（预测）
    def forward(self, X):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']
        a1 = X @ W1 + b1
        z1 = sigmoid(a1)
        a2 = z1 @ W2 + b2
        y = softmax(a2)
        return y
    # 计算损失
    def loss(self, x, t):
        y = self.forward(x)
        loss_value = cross_entropy(y, t)
        return loss_value
    # 计算准确度
    def accuracy(self, x, t):
        y_proba = self.forward(x) # 预测概率
        # 根据最大概率得到预测的分类号
        y = np.argmax(y_proba, axis=1)
        # 与正确解标签对比，得到准确率
        accuracy = np.sum(y == t) / x.shape[0]
        return accuracy
    # 计算梯度
    def numerical_gradient(self, x, t):
        # 定义目标函数
        loss_f = lambda w: self.loss(x, t)
        # 对每个参数，使用数值微分方法计算梯度
        grads = {}
        grads['W1'] = numerical_gradient(loss_f, self.params['W1'])
        grads['b1'] = numerical_gradient(loss_f, self.params['b1'])
        grads['W2'] = numerical_gradient(loss_f, self.params['W2'])
        grads['b2'] = numerical_gradient(loss_f, self.params['b2'])
        return grads