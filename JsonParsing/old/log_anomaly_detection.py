# 模型预测测试
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
# 假设以下导入的是你的自定义模块，确保它们已经定义并可用
from logClusterAnalysis import read_json_lines, extract_log_templates, tfidf_features, word2vec_features

# 设备配置 - 自动使用GPU（如果可用）
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 路径和模型参数配置
log_file_path = "Resource/data/elasticsearch_data/elasticsearch_data.json"  # 日志数据存储路径
model_path = 'model/gru_model.ckpt'  # 确保这是GRU模型的正确路径

# GRU模型定义
class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout_rate=0.5):
        super(GRUModel, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.gru = nn.GRU(input_size, hidden_size, num_layers,
                          batch_first=True, dropout=dropout_rate)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        out, _ = self.gru(x, h0)
        out = self.dropout(out[:, -1, :])  # Take the last output for each sequence
        out = self.fc(out)
        return out

# 动态获取input_size
def get_dynamic_input_size(log_file_path):
    new_logs = read_json_lines(log_file_path)
    log_data = pd.DataFrame(new_logs)
    log_data['template'] = extract_log_templates(log_data, 'message')
    tfidf_matrix = tfidf_features(log_data['template'])
    word2vec_matrix = word2vec_features(log_data['template'])
    combined_features = np.concatenate((tfidf_matrix, word2vec_matrix), axis=1)
    return combined_features.shape[1]  # 返回特征的维度

# 加载模型
def load_model(model_path, input_size, hidden_size, num_layers, num_classes, dropout_rate):
    model = GRUModel(input_size, hidden_size, num_layers, num_classes, dropout_rate)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()  # 设置为评估模式
    return model

# 使用模型进行预测
def predict(model, features):
    with torch.no_grad():  # 在预测时不计算梯度
        features_tensor = torch.tensor(features, dtype=torch.float).unsqueeze(0).to(device)  # 添加批次大小的维度
        outputs = model(features_tensor)
        _, predicted = torch.max(outputs.data, 1)
        return predicted.item()

def main():
    # Step 1: 读取最新的日志数据
    new_logs = read_json_lines(log_file_path)

    # 转换日志数据为DataFrame，便于处理
    log_data = pd.DataFrame(new_logs)

    # Step 2: 将日志数据编码化为模型需要的格式
    # 提取日志模板
    log_data['template'] = extract_log_templates(log_data, 'message')

    # 特征提取: TF-IDF 和 Word2Vec
    tfidf_matrix = tfidf_features(log_data['template'])
    word2vec_matrix = word2vec_features(log_data['template'])
    combined_features = np.concatenate((tfidf_matrix, word2vec_matrix), axis=1)

    # 确保特征维度与模型输入一致
    input_size = get_dynamic_input_size(log_file_path)  # 动态获取input_size
    hidden_size = 64  # 根据模型调整
    num_layers = 2  # 根据模型调整
    num_classes = 2  # 根据模型调整
    dropout_rate = 0.5  # 根据模型调整

    if combined_features.shape[1] != input_size:
        raise ValueError(f"Feature size must be {input_size}, but got {combined_features.shape[1]}")

    # Step 3: 加载模型并进行预测
    model = load_model(model_path, input_size, hidden_size, num_layers, num_classes, dropout_rate)
    predictions = predict(model, combined_features)

    # 输出预测结果
    print("预测结果：", predictions)

if __name__ == "__main__":
    main()
