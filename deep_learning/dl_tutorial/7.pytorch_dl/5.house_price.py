import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split  # 划分数据集
from sklearn.compose import ColumnTransformer  # 列转换器
from sklearn.pipeline import Pipeline  # 管道操作
from sklearn.impute import SimpleImputer  # 缺省值处理
from sklearn.preprocessing import StandardScaler, OneHotEncoder  # 标准化和独热编码
from torch.utils.data import TensorDataset, DataLoader  # 数据集和数据加载器


# 房价预测

# 创建训练集
def create_dataset():
    # 1. 从文件读取数据
    data = pd.read_csv('../data/house_prices.csv')
    # 2. 去除无关列 (去除 标题为 Id 的列，原地操作)
    data.drop(["Id"], axis=1, inplace=True)
    # 3. 划分特征和目标
    X = data.drop(["SalePrice"], axis=1)
    y = data["SalePrice"]
    # 4. 划分训练集和测试集
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # 5. 特征工程（特征转换）
    # 5.1 按照特征数据类型划分数值型和类别型
    numerical_features = X.select_dtypes(exclude=['object']).columns  # 数值型特征是排除 object 类型
    categorical_features = X.select_dtypes(include=['str', 'object']).columns  # 类别型特征
    # 5.2 定义列转换器
    # 5.2.1 数值型特征，用平均值填充缺失项，再进行标准化
    numerical_transformer = Pipeline(
        steps=[
            ('fillna', SimpleImputer(strategy='mean')),
            ('std', StandardScaler())
        ]
    )
    # 5.2.2 类别型特征：用默认值填充缺失项，再做独热编码
    categorical_transformer = Pipeline(
        steps=[
            ('fillna', SimpleImputer(strategy='constant', fill_value='NaN')),
            # 如果某个特征的类在训练集没有出现，在测试集出现，就会是个 unknown 的类，默认行为会报错，这里配置为忽略
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ]
    )
    # 5.2.3 组合列转换器
    transformer = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )
    # 5.3 进行特征转换，构建新的列，组成最终的数据集
    x_train = transformer.fit_transform(x_train)
    x_test = transformer.transform(x_test)

    x_train = pd.DataFrame(x_train.toarray(), columns=transformer.get_feature_names_out())
    x_test = pd.DataFrame(x_test.toarray(), columns=transformer.get_feature_names_out())

    # 6. 构建 Tensor 数据集
    train_dataset = TensorDataset(torch.tensor(x_train.values).float(), torch.tensor(y_train.values).float())
    test_dataset = TensorDataset(torch.tensor(x_test.values).float(), torch.tensor(y_test.values).float())
    # 返回训练集和测试集，以及特征的数量
    return train_dataset, test_dataset, x_train.shape[1]


if __name__ == '__main__':
    # 1. 加载数据
    train_dataset, test_dataset, feature_num = create_dataset()
    # print(feature_num)

    # 2. 创建模型
    model = nn.Sequential(
        nn.Linear(feature_num, 128),
        nn.BatchNorm1d(128),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(128, 1),
    )


    # 3. 自定义损失函数
    def log_rmse(y_pred, target):
        """

        :param y_pred: 预测值
        :param target: 目标值
        :return:
        """
        y_pred = torch.clamp(y_pred, 1, float("inf"))  # 防止 log 结果为负数
        mse = nn.MSELoss()
        return torch.sqrt(mse(torch.log(y_pred), torch.log(target))) # todo 为什么要开根号？


    # 4. 模型训练和测试
    def train_test(model, train_dataset, test_dataset, lr, epoch_num, batch_size, device):
        """

        :param model: 模型
        :param train_dataset: 训练数据集
        :param test_dataset: 测试数据集
        :param lr: 学习率
        :param epoch_num: 轮次
        :param batch_size:
        :param device: 运行设备 cpu 或 cuda
        :return:
        """

        # 1. 初始化相关操作
        def init_params(layer):
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)

        # 1.1 参数初始化
        model.apply(init_params)
        # 1.2 将模型加载到设备
        model.to(device)
        # 1.3 定义优化器
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        # 定义训练误差和测试误差变化列表
        train_loss_list = []
        test_loss_list = []

        # 2. 模型训练
        for epoch in range(epoch_num):
            model.train() # 开启训练模式，正则化相关的 BatchNorm1d 和随机失活开启（测试的时候可以关闭）
            # 2.1 创建 DataLoader
            train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            train_loss_total = 0
            # 2.2 按批次迭代训练模型
            for batch_idx, (X, y) in enumerate(train_dataloader):
                X, y = X.to(device), y.to(device)
                # 2.3.1 前向传播
                y_pred = model(X)
                # 2.3.2 计算损失
                loss_value = log_rmse(y_pred.squeeze(), y)
                # 2.3.3 反向传播
                loss_value.backward()
                # 2.3.4 更新参数
                optimizer.step()
                optimizer.zero_grad() # 梯度清零

                # 累加损失
                train_loss_total += loss_value.item() * X.shape[0]

            this_train_loss = train_loss_total / len(train_dataset)
            train_loss_list.append(this_train_loss)

            # 3. 测试
            model.eval()
            # 3.1 定义DataLoader
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
            # 3.2 计算测试误差
            test_loss_total = 0
            with torch.no_grad():  # 测试时关闭梯度计算
                for X, y in test_loader:
                    X, y = X.to(device), y.to(device)
                    y_pred = model(X)
                    loss_value = log_rmse(y_pred.squeeze(), y)
                    test_loss_total += loss_value.item() * X.shape[0]
            this_test_loss = test_loss_total / len(test_dataset)
            test_loss_list.append(this_test_loss)

            print(f"epoch: {epoch + 1}, train loss: {this_train_loss}, test loss: {this_test_loss}")

        return train_loss_list, test_loss_list

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 超参数
    lr = 0.1
    epoch_num = 200
    batch_size = 64
    train_loss_list, test_loss_list = train_test(model, train_dataset, test_dataset, lr, epoch_num, batch_size,
                                                 device)

    # 画图
    plt.plot(train_loss_list, 'r-', label='train loss', linewidth=3)
    plt.plot(test_loss_list, 'k--', label='test loss', linewidth=2)

    plt.legend()
    plt.show()