import os
import json
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
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

def detect_and_notify():
    log_file_path = 'Resource/data/elasticsearch_data/cleaned_elasticsearch_data.json'  # 修改为你的日志文件路径
    model_path = 'model/gru_model.ckpt'  # 修改为你的模型文件路径

    log_entries = read_json_lines(log_file_path)
    log_data = pd.DataFrame(log_entries)

    log_data['template'] = extract_log_templates(log_data, 'message')
    tfidf_matrix = tfidf_features(log_data['template'])
    log_data['cluster'] = cluster_log_templates(tfidf_matrix, n_clusters=20)

    # 定义异常日志集群
    exceptional_clusters = {
        0: 1, 1: 0, 2: 1, 3: 0, 4: 1, 5: 1, 6: 0, 7: 1, 8: 0, 9: 1,
        10: 1, 11: 0, 12: 1, 13: 1, 14: 1, 15: 1, 16: 0, 17: 0, 18: 1, 19: 0
    }

    # 检测异常日志
    exceptional_entries = log_data[log_data['cluster'].map(lambda x: exceptional_clusters[x] == 1)]

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

    print(f"Total predictions made: {len(log_data)}")
    print(f"Found {len(exceptional_entries)} exceptional log entries.")

# 执行一次检测和通知
detect_and_notify()
