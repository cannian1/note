import numpy as np


def step_function(x):
    """阶跃函数"""
    return 1 if x >= 0 else 0


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def tanh(x):
    return np.tanh(x)


def relu(x):
    return np.maximum(0, x)


def softmax(x):
    if x.ndim == 2:
        x = x - np.max(x, axis=1, keepdims=True)
        y = np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)
        return y
    x = x - np.max(x)  # 防止溢出
    return np.exp(x) / np.sum(np.exp(x))


# 恒等函数
def identity(x):
    return x


# 损失函数
def mean_squared_error(y_true, t_pred):
    """
    MSE 一般用于回归问题
    :param y_true: 预测值
    :param t_pred: 目标真实值
    :return:
    """
    return 0.5 * np.sum((y_true - t_pred) ** 2)


# 矩阵乘法，y矩阵形状 n * k
# n 是输入数据条数，k是分类个数，每一行是一个数据的预测结果
def cross_entropy(y, t):
    """
    交叉熵误差 一般用于分类问题
    :param y: 预测值（神经网络输出的概率）
    :param t: 目标真实值（标签）
    :return:
    """

    # 将 y 转为二维
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
    # 处理 one-hot 编码：将 t 转换为顺序编码（类别标签）
    if y.size == t.size:
        t = t.argmax(axis=1)

    n = y.shape[0]  # y 的行数
    return -np.sum(np.log(y[np.arange(n), t] + 1e-10)) / n
