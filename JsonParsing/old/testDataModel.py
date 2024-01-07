import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
import matplotlib.pyplot as plt
import json

# 设备配置 - 自动使用GPU（如果可用）
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# GRU模型定义
class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(GRUModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        out, _ = self.gru(x, h0)
        out = self.fc(out[:, -1, :])
        return out

# 加载数据集函数
def load_dataset(data_path, label_path):
    data = np.load(data_path)
    labels = np.load(label_path)
    tensor_data = torch.tensor(data, dtype=torch.float).to(device)
    tensor_labels = torch.tensor(labels).to(device)
    return DataLoader(TensorDataset(tensor_data, tensor_labels), batch_size=2048, shuffle=False)

# 评估模型
def evaluate_model(model, test_loader):
    model.eval()
    TP = TN = FP = FN = 0
    predicted_probs = []
    true_labels = []
    with torch.no_grad():
        for sequences, labels in test_loader:
            sequences = sequences.to(device)
            labels = labels.to(device)
            outputs = model(sequences)
            _, predicted = torch.max(outputs.data, 1)
            predicted_probs.append(outputs.cpu().numpy())
            true_labels.append(labels.cpu().numpy())
            for i in range(len(labels)):
                if labels[i] == predicted[i] == 1:
                    TP += 1
                elif labels[i] == predicted[i] == 0:
                    TN += 1
                elif labels[i] == 0 and predicted[i] == 1:
                    FP += 1
                elif labels[i] == 1 and predicted[i] == 0:
                    FN += 1

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    F1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    TPR = recall
    FPR = FP / (FP + TN) if (FP + TN) > 0 else 0

    metrics = {
        'Confusion Matrix': {'TP': TP, 'FP': FP, 'TN': TN, 'FN': FN},
        'Precision': precision,
        'Recall': recall,
        'F1 Score': F1,
        'TPR': TPR,
        'FPR': FPR
    }

    with open('metrics.json', 'w', encoding='utf-8') as json_file:
        json.dump(metrics, json_file, indent=4, ensure_ascii=False)

    predicted_probs = np.vstack(predicted_probs)
    true_labels = np.hstack(true_labels)

    return predicted_probs, true_labels

# 绘制ROC曲线
def plot_roc_curve(fpr, tpr, auc_score):
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {auc_score:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()

# 绘制精确率-召回率曲线
def plot_precision_recall_curve(precision, recall, average_precision):
    plt.figure(figsize=(6, 6))
    plt.step(recall, precision, color='b', alpha=0.2, where='post')
    plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title(f'Precision-Recall curve: AP={average_precision:.2f}')
    plt.show()

def main():
    # 数据集路径
    train_data_path = 'Resource/dataSet/X_train.npy'
    train_label_path = 'Resource/dataSet/y_train.npy'
    test_data_path = 'Resource/dataSet/X_test.npy'
    test_label_path = 'Resource/dataSet/y_test.npy'

    # 检查数据以自动匹配超参数
    train_data = np.load(train_data_path)
    num_classes = 2  # 根据实际情况调整
    input_size = train_data.shape[2]  # 第三维是特征数量

    # 超参数
    num_epochs = 300
    hidden_size = 64
    num_layers = 2

    # 初始化模型
    model = GRUModel(input_size, hidden_size, num_layers, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    # 加载数据集
    train_loader = load_dataset(train_data_path, train_label_path)
    test_loader = load_dataset(test_data_path, test_label_path)

    # 训练模型
    for epoch in range(num_epochs):
        model.train()
        for sequences, labels in train_loader:
            sequences = sequences.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(sequences)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    print('Training completed')

    # 评估模型
    predicted_probs, true_labels = evaluate_model(model, test_loader)

    # 计算并绘制ROC曲线和精确率-召回率曲线
    fpr, tpr, _ = roc_curve(true_labels, predicted_probs[:, 1])
    roc_auc = auc(fpr, tpr)
    plot_roc_curve(fpr, tpr, roc_auc)

    precision, recall, _ = precision_recall_curve(true_labels, predicted_probs[:, 1])
    average_precision = np.mean(precision)
    plot_precision_recall_curve(precision, recall, average_precision)

if __name__ == "__main__":
    main()
