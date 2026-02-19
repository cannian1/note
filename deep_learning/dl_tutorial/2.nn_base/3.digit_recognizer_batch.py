# 手写数字识别
import numpy as np
import pandas as pd
import joblib  # 保存/加载模型
from sklearn.model_selection import train_test_split  # 划分训练集测试集
from sklearn.preprocessing import MinMaxScaler  # 归一化
from common.functions import sigmoid, softmax


# 读取数据
def get_data():
    # 1. 从文件加载数据集
    data = pd.read_csv('../data/train.csv')
    # 2. 划分数据集
    x = data.drop('label', axis=1)
    y = data['label']
    # 将特征数据 x 和标签数据 y 按相同索引随机打乱后, 划分为训练集和测试集
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
    # 3. 特征工程：归一化
    scaler = MinMaxScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    return x_test, y_test


# 前向传播
def forward_propagation(network, x):
    w1, w2, w3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    # 逐层进行计算传递
    a1 = x @ w1 + b1
    z1 = sigmoid(a1)
    a2 = z1 @ w2 + b2
    z2 = sigmoid(a2)
    a3 = z2 @ w3 + b3

    # 输出层
    y = softmax(a3)
    return y


# 初始化神经网络
def init_network():
    # 直接从文件中加载字典对象
    network = joblib.load('../data/nn_sample')
    return network


if __name__ == '__main__':
    # 1. 获取测试数据
    x, y = get_data()

    # 2. 创建模型（加载参数）
    network = init_network()

    batch_size = 100
    accuracy_cnt = 0
    n = x.shape[0]

    # 3. 循环迭代：分批次做测试，前向传播，并累积预测准确个数
    for i in range(0, n, batch_size):
        # 3.1 取出当前批次的数据
        x_batch = x[i:i + batch_size]
        # 3.2 前向传播
        y_batch = forward_propagation(network, x_batch)
        # 3.3 将输出分类概率转换为分类标签
        y_pred = np.argmax(y_batch, axis=1)
        # 3.4 累加准确个数
        accuracy_cnt += np.sum(y_pred == y[i:i + batch_size])

    # 4. 计算分类准确率
    print("Accuracy: ", accuracy_cnt / n)
