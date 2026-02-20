import numpy as np


# 数值微分求导，传入x是一个标量
def numerical_diff(f, x):
    h = 1e-4
    return (f(x + h) - f(x - h)) / (2 * h)


# 数值微分求梯度，传入 x 是一个向量
def _numerical_gradient(f, x):
    h = 1e-4
    grad = np.zeros_like(x)

    # 遍历 x 中的特征 xi
    for i in range(x.size):
        tmp = x[i]
        x[i] = tmp + h
        fxh1 = f(x)
        x[i] = tmp - h
        fxh2 = f(x)
        # 利用中心差分公式计算偏导数
        grad[i] = (fxh1 - fxh2) / (2 * h)
        # 恢复x[i]的值
        x[i] = tmp

    return grad


# 传入 X 是一个矩阵
def numerical_gradient(f, X):
    # 判断维度
    if X.ndim == 1:
        return _numerical_gradient(f, X)
    else:
        grad = np.zeros_like(X)
        # 遍历 X 中的每一行数据，分别求梯度
        for i, v in enumerate(X):
            grad[i] = _numerical_gradient(f, v)
        return grad
