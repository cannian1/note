import torch
from torch import nn, optim


# 定义神经网络模型
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        # 只定义一个全连接层
        self.linear = nn.Linear(5,5)
        # 权重初始化
        self.linear.weight.data = torch.tensor(
            [
                [0.1, 0.2, 0.3],
                [0.4, 0.5, 0.6],
                [0.7, 0.8, 0.9],
                [1.0, 1.1, 1.2],
                [1.3, 1.4, 1.5]
            ]
        ).T
        self.linear.bias.data = torch.tensor([1.0, 2.0, 3.0])

    # 前向传播
    def forward(self, x):
        x = self.linear(x)
        return x

# 1. 定义数据
# 输入数据 2×5
X = torch.tensor([[1,2,3,4,5], [6,7,8,9,10]], dtype=torch.float)
# 目标值 2×3
target = torch.tensor([[0,0,0],[0,0,0]], dtype=torch.float)

# 2. 创建模型
model = Model()

# 3. 前向传播，预测输出
output = model(X)

# 4. 定义损失函数，计算损失
loss = nn.MSELoss()
loss_value = loss(output, target)

# 5. 反向传播，计算梯度
loss_value.backward()

# 6. 定义优化器 随机梯度下降
optimizer = optim.SGD(model.parameters(), lr=0.1)

# 7. 更新参数
optimizer.step()
optimizer.zero_grad()

# 打印模型参数
for param in model.state_dict():
    print(param)
    print(model.state_dict()[param])