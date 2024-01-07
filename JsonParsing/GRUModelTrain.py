#训练并保存模型
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import time

# 设备配置 - 将使用GPU（如果可用）
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# GRU模型定义
class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(GRUModel, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        out, _ = self.gru(x, h0)
        out = self.fc(out[:, -1, :])  # 取最后一个时间步的输出
        return out

# 数据加载函数
def load_npy(file_path):
    data = np.load(file_path, allow_pickle=True)  # 添加 allow_pickle=True 参数
    return torch.tensor(data, dtype=torch.float).to(device)


def main():
    # 超参数设置
    num_classes = 2  # 根据实际情况调整
    num_epochs = 300  # 示范迭代次数
    batch_size = 16  # 示范批次大小
    hidden_size = 32  # 示范隐藏层大小
    num_layers = 1  # 示范层数

    # 加载数据
    X_train = load_npy('Resource/dataSet/testData/X_train.npy')
    y_train = load_npy('Resource/dataSet/testData/y_train.npy').long()
    input_size = X_train.shape[2]  # 第三维是特征数量

    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

    # 初始化模型
    model = GRUModel(input_size, hidden_size, num_layers, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), weight_decay=1e-5)  # 添加L2正则化
    writer = SummaryWriter(log_dir='runs')  # 用于TensorBoard日志

    # 训练模型
    start_time = time.time()
    for epoch in range(num_epochs):
        model.train()  # 设置模型为训练模式
        train_loss = 0.0

        for sequences, labels in train_loader:
            sequences = sequences.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(sequences)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        print(f'Epoch [{epoch + 1}/{num_epochs}], 训练损失: {avg_train_loss:.4f}')
        writer.add_scalar('train_loss', avg_train_loss, epoch + 1)

    elapsed_time = time.time() - start_time
    print(f'训练完成，耗时 {elapsed_time:.2f}秒')

    # 保存训练好的模型
    model_dir = 'model'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    torch.save(model.state_dict(), os.path.join(model_dir, 'gru_model.ckpt'))
    writer.close()

if __name__ == "__main__":
    main()