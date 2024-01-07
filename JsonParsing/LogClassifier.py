import numpy as np
import torch
import torch.nn as nn

# 定义GRU模型
class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(GRUModel, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)  # 注意添加.to(x.device)以支持GPU
        out, _ = self.gru(x, h0)
        out = self.fc(out[:, -1, :])
        return out

def preprocess_and_predict(model_path, test_data_path):
    # 加载测试数据
    X_test_normal = np.load(test_data_path)

    # 数据转换为torch tensor
    data = torch.tensor(X_test_normal, dtype=torch.float32)

    # 加载模型并进行预测
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GRUModel(input_size=data.shape[2], hidden_size=32, num_layers=1, num_classes=2).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    data = data.to(device)
    with torch.no_grad():
        outputs = model(data)
        _, predicted = torch.max(outputs.data, 1)

    # 统计预测为正常的日志数量
    normal_count = (predicted == 0).sum().item()

    # 打印结果
    print(f"总日志数: {len(predicted)}")
    print(f"预测为正常的日志数: {normal_count}")

if __name__ == "__main__":
    model_path = 'model/gru_model.ckpt'  # 更新为你的模型路径
    test_data_path = 'Resource/dataSet/testData/X_test_abnormal.npy'  # 更新为你的测试数据路径
    preprocess_and_predict(model_path, test_data_path)
