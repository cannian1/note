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
    x = x - np.max(x) # 防止溢出
    return np.exp(x) / np.sum(np.exp(x))

def identity(x):
    return x