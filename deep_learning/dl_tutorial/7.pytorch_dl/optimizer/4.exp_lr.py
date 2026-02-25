import torch
import numpy as np
import matplotlib.pyplot as plt

from torch.optim import SGD
from torch.optim.lr_scheduler import ExponentialLR

def f(X):
    return 0.05 * X[0]**2 + X[1]**2

# 主流程

# 1. 参数X初始化
X = torch.tensor([-7.0, 2.0], requires_grad=True)

# 2. 定义超参数
lr = 0.9
num_iters = 500

# 3. 定义优化器 SGD
optimizer = SGD( [X], lr=lr )

# 4. 定义学习率衰减策略（指数衰减）
lr_scheduler = ExponentialLR(optimizer, gamma=0.99)

# 拷贝当前X的值，放入列表中
X_arr = X.detach().numpy().copy()
lr_list = []
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
    lr_list.append(optimizer.param_groups[0]['lr'])

    # 5. 更新学习率
    lr_scheduler.step()

plt.rcParams['font.sans-serif'] = ['KaiTi']
plt.rcParams['axes.unicode_minus'] = False

# 画图，划分子图
fig, ax = plt.subplots(1, 2, figsize=(12, 4))

x1_grid, x2_grid = np.meshgrid(np.linspace(-7,7,100), np.linspace(-2,2, 100))
y_grid = 0.05 * x1_grid**2 + x2_grid**2

# 画等高线和X点轨迹
ax[0].contour(x1_grid, x2_grid, y_grid, levels=30, colors='gray')
ax[0].plot(X_arr[:, 0], X_arr[:, 1], 'r')
ax[0].set_title('梯度下降过程')

# 画出学习率衰减曲线
ax[1].plot(lr_list, 'k')
ax[1].set_title("学习率衰减")

plt.show()