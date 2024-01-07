import os
import json
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from datetime import datetime
from drain3 import TemplateMiner
from drain3.template_miner_config import TemplateMinerConfig
import smtplib
from email.mime.text import MIMEText
from email.header import Header

# 设备配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
        out = self.fc(out[:, -1, :])
        return out

def read_json_lines(file_path):
    with open(file_path, 'r') as file:
        return [json.loads(line.strip()) for line in file if line.strip()]

def parse_timestamp(timestamp):
    return datetime.strptime(timestamp, '%Y-%m-%dT%H:%M:%S.%fZ')

def extract_log_templates(data, message_column):
    config = TemplateMinerConfig()
    template_miner = TemplateMiner(config=config)
    templates = []
    for message in data[message_column]:
        result = template_miner.add_log_message(message)
        templates.append(result['template_mined'])
    return templates

def tfidf_features(templates):
    vectorizer = TfidfVectorizer(max_features=1000)
    tfidf_matrix = vectorizer.fit_transform(templates)
    return tfidf_matrix.toarray()

def cluster_log_templates(features, n_clusters=20):
    kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init=10)
    clusters = kmeans.fit_predict(features)
    return clusters

def predict_with_model(model, data):
    if len(data.size()) == 2:  # 如果输入是2D的，添加一个batch_size维度
        data = data.unsqueeze(0)
    model.eval()
    with torch.no_grad():
        outputs = model(data)
        _, predicted = torch.max(outputs.data, 1)
    return predicted

def send_email(to_email, subject, content):
    smtp_server = 'smtp.qq.com'
    sender = '3534976455@qq.com'  # 替换为你的发件人邮箱
    password = 'uoubggykuhogdbgh'  # 替换为你的邮箱密码或应用密码

    msg = MIMEText(content, 'plain', 'utf-8')
    msg['Subject'] = Header(subject, 'utf-8')
    msg['From'] = sender
    msg['To'] = to_email

    try:
        server = smtplib.SMTP(smtp_server, 587)
        server.starttls()
        server.login(sender, password)
        server.sendmail(sender, [to_email], msg.as_string())
        server.quit()
        print(f"已向{to_email}发送异常邮件")
    except Exception as e:
        print(f"发送异常邮件时出错：{e}")

# 新增函数以读取和更新时间戳
timestamp_file = 'Resource/data/elasticsearch_data/last_timestamp.txt'  # 存储时间戳的文件
start_date = datetime(2023, 12, 3)  # 默认开始日期

def read_last_timestamp():
    if os.path.exists(timestamp_file):
        with open(timestamp_file, "r") as file:
            last_timestamp = file.read().strip()
            return last_timestamp if last_timestamp else start_date.strftime('%Y-%m-%dT%H:%M:%S.%fZ')
    else:
        with open(timestamp_file, "w") as file:  # 创建文件
            file.write(start_date.strftime('%Y-%m-%dT%H:%M:%S.%fZ'))
        return start_date.strftime('%Y-%m-%dT%H:%M:%S.%fZ')

def update_timestamp_file(new_timestamp):
    with open(timestamp_file, "w") as file:
        file.write(new_timestamp)

def detect_and_notify():
    log_file_path = 'Resource/data/elasticsearch_data/cleaned_elasticsearch_data.json'  # 修改为你的日志文件路径
    model_path = 'model/gru_model.ckpt'  # 修改为你的模型文件路径

    last_timestamp = read_last_timestamp()  # 读取上一次的时间戳
    log_entries = read_json_lines(log_file_path)
    log_data = pd.DataFrame(log_entries)
    log_data['parsed_timestamp'] = log_data['timestamp'].apply(parse_timestamp)

    # 筛选新日志，仅处理时间戳大于last_timestamp的日志
    new_log_data = log_data[log_data['parsed_timestamp'] > datetime.strptime(last_timestamp, '%Y-%m-%dT%H:%M:%S.%fZ')]

    if new_log_data.empty:
        print("未发现新的日志条目。")
        return

    new_log_data['template'] = extract_log_templates(new_log_data, 'message')
    tfidf_matrix = tfidf_features(new_log_data['template'])
    new_log_data['cluster'] = cluster_log_templates(tfidf_matrix, n_clusters=20)

    exceptional_clusters = {
        0: 1, 1: 0, 2: 1, 3: 0, 4: 1, 5: 1, 6: 0, 7: 1, 8: 0, 9: 1,
        10: 1, 11: 0, 12: 1, 13: 1, 14: 1, 15: 1, 16: 0, 17: 0, 18: 1, 19: 0
    }
    exceptional_entries = new_log_data[new_log_data['cluster'].isin(exceptional_clusters)]

    if not exceptional_entries.empty:
        print("检测到异常日志:")
        for index, row in exceptional_entries.iterrows():
            print(f"异常日志索引: {index}, 类型: Cluster {row['cluster']}")
            print(f"日志内容: {row['template']}")
            # 发送邮件通知
            to_email = '1481215919@qq.com'  # 替换为实际接收者邮箱
            subject = '异常日志通知'
            content = f"检测到异常日志：\n{row['template']}"
            send_email(to_email, subject, content)

        # 更新时间戳为这批新日志中的最后一条
        update_timestamp_file(new_log_data['parsed_timestamp'].max().strftime('%Y-%m-%dT%H:%M:%S.%fZ'))
    else:
        print("未检测到异常日志。")

    # 调试信息
    print(f"Total predictions made: {len(new_log_data)}")
    print(f"Found {len(exceptional_entries)} exceptional log entries.")

    # 加载模型
    model_input_size = tfidf_matrix.shape[1]  # 更新模型输入维度
    model = GRUModel(model_input_size, hidden_size=32, num_layers=1, num_classes=2).to(device)

    try :
        model.load_state_dict(torch.load(model_path, map_location=device))
        print("模型加载成功.")
    except RuntimeError as e:
        print(f"模型加载失败: {e}")
        print("您可能需要重新训练模型或确保模型与数据兼容。")

    # 进行预测
    if not exceptional_entries.empty:
        print("\n预测结果:")
        for index, row in exceptional_entries.iterrows():
            input_data = torch.tensor(tfidf_matrix[index]).float().unsqueeze(0).to(device)
            pred = predict_with_model(model, input_data)
            print(f"异常日志索引: {index}, 类型: Cluster {row['cluster']}")
            print(f"日志内容: {row['template']}")
            print(f"预测结果: {'异常' if pred == 1 else '正常'}")

# 执行一次检测和通知
detect_and_notify()
