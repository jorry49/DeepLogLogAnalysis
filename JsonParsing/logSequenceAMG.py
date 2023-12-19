import argparse
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.utils.data import TensorDataset, DataLoader

# 设备配置，检查是否有GPU可用，如果有则使用，否则使用CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 生成函数，从文件中读取日志数据并生成会话序列
def generate(file_name):
    num_sessions = 0
    inputs = []
    outputs = []
    file_path = os.path.join('Resource', 'data', file_name)

    with open(file_path, 'r') as f:
        for line in f.readlines():
            num_sessions += 1
            line = list(map(int, line.strip().split()))

            # 假设 window_size 已经定义
            for i in range(len(line) - window_size):
                inputs.append(line[i:i + window_size])
                outputs.append(line[i + window_size])

    print(f'会话数量({file_name}): {num_sessions}')
    print(f'序列数量({file_name}): {len(inputs)}')

    dataset = TensorDataset(torch.tensor(inputs, dtype=torch.float), torch.tensor(outputs))
    return dataset


# 深度学习模型定义
class Model(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_keys):
        super(Model, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_keys)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out


if __name__ == '__main__':
    # 超参数配置
    num_classes = 289  # 根据您的数据调整这个值
    num_epochs = 300
    batch_size = 2048
    input_size = 1  # 通常保持为1，除非有特殊情况
    model_dir = 'model'

    # 日志记录的字符串
    log = f'Adam_batch_size={batch_size}_epoch={num_epochs}'

    parser = argparse.ArgumentParser()
    parser.add_argument('-num_layers', default=2, type=int)
    parser.add_argument('-hidden_size', default=64, type=int)
    parser.add_argument('-window_size', default=10, type=int)  # 根据您的数据调整这个值
    args = parser.parse_args()
    num_layers = args.num_layers
    hidden_size = args.hidden_size
    window_size = args.window_size

    model = Model(input_size, hidden_size, num_layers, num_classes).to(device)

    # 请确保传入正确的文件名
    seq_dataset = generate('event_sequences.txt')
    dataloader = DataLoader(seq_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)

    # 生成训练数据集
    seq_dataset = generate('event_sequences.txt')
    dataloader = DataLoader(seq_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)

    # 使用TensorBoard进行可视化
    writer = SummaryWriter(log_dir=os.path.join('log', log))

    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    # 训练模型
    start_time = time.time()
    total_step = len(dataloader)
    for epoch in range(num_epochs):  # 循环多次遍历数据集
        train_loss = 0
        for step, (seq, label) in enumerate(dataloader):
            # 前向传播
            seq = seq.clone().detach().view(-1, window_size, input_size).to(device)
            output = model(seq)
            loss = criterion(output, label.to(device))

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
            writer.add_graph(model, seq)
        print(f'第 [{epoch + 1}/{num_epochs}] 轮, 训练损失: {train_loss / total_step:.4f}')
        writer.add_scalar('train_loss', train_loss / total_step, epoch + 1)
    elapsed_time = time.time() - start_time
    print(f'运行时间: {elapsed_time:.3f}s')

    # 保存模型参数
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)
    model_path = os.path.join(model_dir, f'{log}.pt')
    torch.save(model.state_dict(), model_path)
    print(f'模型已保存到: {model_path}')
    writer.close()
    print('训练完成')
