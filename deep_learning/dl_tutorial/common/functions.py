import numpy as np


def step_function(x):
    """阶跃函数"""
    return 1 if x >= 0 else 0


# sigmoid 函数可以用做激活函数和输出层函数，双曲正切函数只能用作激活函数
# tanh 关于原点对称


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def tanh(x):
    return np.tanh(x)


# 深层神经网络的隐藏层激活函数一般是 relu，不会出现梯度减少的现象
# relu 也可以用在二分类问题的输出层
def relu(x):
    return np.maximum(0, x)


# 经常用于多分类或二分类问题的输出层
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
