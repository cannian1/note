import numpy as np

from common.functions import softmax, cross_entropy
from common.gradient import numerical_gradient


# 定义一个简单神经网络类
class SimpleNet():
    def __init__(self):
        # 初始化
        self.W = np.random.randn(2, 3)  # 2 * 3 的矩阵，符合正态分布

    # 前向传播
    def forward(self, X):
        a = X @ self.W
        y = softmax(a)
        return y

    # 计算损失值
    def loss(self, x, t):
        y = self.forward(x)
        loss = cross_entropy(y, t)
        return loss


if __name__ == '__main__':
    # 定义数据
    x = np.array([0.6, 0.9])
    t = np.array([0, 0, 1])  # 三分类的独热编码

    # 定义神经网络的模型
    net = SimpleNet()
    # 计算梯度
    f = lambda _: net.loss(x, t)
    gradw = numerical_gradient(f, net.W)

    # 2 * 3 的矩阵，表示损失函数相对于每个权重的梯度
    print(gradw)
