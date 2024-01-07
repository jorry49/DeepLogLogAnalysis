import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from drain3 import TemplateMiner
from drain3.template_miner_config import TemplateMinerConfig
from sklearn.feature_extraction.text import TfidfVectorizer
import json
import os

# 定义窗口大小
window_size = 5

# 从文件读取日志数据
def read_json_lines(file_path):
    with open(file_path, 'r') as file:
        return [json.loads(line.strip()) for line in file.readlines()]

# 解析时间戳
def parse_timestamp(timestamp):
    return datetime.strptime(timestamp, '%Y-%m-%dT%H:%M:%S.%fZ')

# 使用 drain3 提取日志模板
def extract_log_templates(data, message_column):
    config = TemplateMinerConfig()
    template_miner = TemplateMiner(config=config)
    templates = []
    for message in data[message_column]:
        result = template_miner.add_log_message(message)
        templates.append(result['template_mined'])
    return templates

# 聚类日志模板
def cluster_log_templates(templates, n_clusters=20):
    vectorizer = TfidfVectorizer(max_features=1000)
    tfidf_matrix = vectorizer.fit_transform(templates)
    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=0)
    clusters = kmeans.fit_predict(tfidf_matrix)
    return clusters

# 根据聚类结果分配标签
def assign_labels(clusters):
    label_map = {
        0: 1,  # 异常 - 认证失败
        1: 0,  # 正常 - cron任务会话开启
        2: 1,  # 异常 - 断开连接（可疑）
        3: 0,  # 正常 - 命令执行
        4: 1,  # 异常 - 接收断开（可疑）
        5: 1,  # 异常 - 无效用户密码失败
        6: 0,  # 正常 - 日常服务启停
        7: 1,  # 异常 - 用户检查失败（可疑）
        8: 0,  # 正常 - cron任务会话关闭
        9: 1,  # 异常 - 连接关闭（可疑）
        10: 1,  # 异常 - 无效用户尝试
        11: 0,  # 正常 - 服务停用
        12: 1,  # 异常 - 无法协商（可疑）
        13: 1,  # 异常 - 密码失败
        14: 1,  # 异常 - 无法协商加密方法
        15: 1,  # 异常 - 密码失败和断开连接
        16: 0,  # 正常 - 命令执行（维护任务）
        17: 0,  # 正常 - 日志文件旋转结束
        18: 1,  # 异常 - 无效用户尝试
        19: 0  # 正常 - 日常任务执行
    }
    labels = [label_map[cluster] for cluster in clusters]
    return labels

def main():
    log_file_path = 'Resource/data/elasticsearch_data/cleaned_elasticsearch_data.json'
    log_entries = read_json_lines(log_file_path)
    log_data = pd.DataFrame(log_entries)

    log_data['parsed_timestamp'] = log_data['timestamp'].apply(parse_timestamp)
    log_data['template'] = extract_log_templates(log_data, 'message')
    clusters = cluster_log_templates(log_data['template'], 20)
    log_data['label'] = assign_labels(clusters)

    tfidf_matrix = TfidfVectorizer(max_features=1000).fit_transform(log_data['template']).toarray()
    sequences = np.array([tfidf_matrix[i:i + window_size] for i in range(len(tfidf_matrix) - window_size + 1)])
    labels = np.array(log_data['label'].values[window_size-1:])

    # 分割原始数据集
    X_train, X_test, y_train, y_test = train_test_split(sequences, labels, test_size=0.2, random_state=42)

    # 筛选出正常和异常数据
    normal_indices = np.where(labels == 0)[0]
    abnormal_indices = np.where(labels == 1)[0]

    X_test_normal = sequences[normal_indices]
    y_test_normal = labels[normal_indices]

    X_test_abnormal = sequences[abnormal_indices]
    y_test_abnormal = labels[abnormal_indices]

    dataset_path = 'Resource/dataSet/testData'
    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)

    # 保存原始测试集
    np.save(os.path.join(dataset_path, 'X_train.npy'), X_train)
    np.save(os.path.join(dataset_path, 'X_test.npy'), X_test)
    np.save(os.path.join(dataset_path, 'y_train.npy'), y_train)
    np.save(os.path.join(dataset_path, 'y_test.npy'), y_test)

    # 保存正常测试集
    np.save(os.path.join(dataset_path, 'X_test_normal.npy'), X_test_normal)
    np.save(os.path.join(dataset_path, 'y_test_normal.npy'), y_test_normal)

    # 保存异常测试集
    np.save(os.path.join(dataset_path, 'X_test_abnormal.npy'), X_test_abnormal)
    np.save(os.path.join(dataset_path, 'y_test_abnormal.npy'), y_test_abnormal)

if __name__ == "__main__":
    main()
