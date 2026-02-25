import torch
import numpy as np
import matplotlib.pyplot as plt

from torch.optim import SGD, RMSprop

# 定义函数
def f(X):
    return 0.05 * X[0]**2 + X[1]**2

# 定义函数，实现梯度下降法
def gradient_descent(X, optimizer, num_iters):
    # 拷贝当前X的值，放入列表中
    X_arr = X.detach().numpy().copy()
    for i in range(num_iters):
        # 1. 前向传播，得到"损失值"
        y = f(X)
        # 2. 反向传播
        y.backward()
        # 3. 更新参数
        optimizer.step()
        # 4. 梯度清零
        optimizer.zero_grad()

        # 将更新之后的X，保存到列表中
        X_arr = np.vstack([X_arr, X.detach().numpy()])
    return X_arr

# 主流程

# 1. 参数X初始化
X = torch.tensor([-7.0, 2.0], requires_grad=True)

# 2. 定义超参数
lr = 0.1
num_iters = 80

# 3. 优化器对比
# 3.1 SGD
X_clone = X.clone().detach().requires_grad_(True)
optimizer = SGD( [X_clone], lr=lr)

# 梯度下降
X_arr1 = gradient_descent(X_clone, optimizer, num_iters)
# 画出点轨迹
plt.plot(X_arr1[:, 0], X_arr1[:, 1], 'r')


# 3.2 RMSProp
X_clone = X.clone().detach().requires_grad_(True)
optimizer = RMSprop( [X_clone], lr=lr, alpha=0.9)

# 梯度下降
X_arr2 = gradient_descent(X_clone, optimizer, num_iters)
# 画出点轨迹
plt.plot(X_arr2[:, 0], X_arr2[:, 1], 'b')

# 画出等高线
x1_grid, x2_grid = np.meshgrid(np.linspace(-7,7,100), np.linspace(-2,2, 100))
y_grid = 0.05 * x1_grid**2 + x2_grid**2
plt.contour(x1_grid, x2_grid, y_grid, levels=30, colors='gray')
plt.legend(["SGD", "RMSProp"])
plt.show()