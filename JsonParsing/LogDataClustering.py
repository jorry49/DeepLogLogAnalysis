#解析日志
import json
from datetime import datetime
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from drain3 import TemplateMiner
from drain3.template_miner_config import TemplateMinerConfig
import numpy as np
import pandas as pd

# 从文件读取日志数据
def read_json_lines(file_path):
    with open(file_path, 'r') as file:
        return [json.loads(line.strip()) for line in file if line.strip()]

# 解析时间戳
def parse_timestamp(timestamp):
    # 根据您的日志格式调整时间格式
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

# TF-IDF 特征提取
def tfidf_features(templates):
    vectorizer = TfidfVectorizer(max_features=1000)
    tfidf_matrix = vectorizer.fit_transform(templates)
    return tfidf_matrix.toarray()

# 聚类日志模板
def cluster_log_templates(features, n_clusters=20):  # 修改聚类数为20
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    clusters = kmeans.fit_predict(features)
    return clusters

# 主函数
def main():
    log_file_path = 'Resource/data/elasticsearch_data/cleaned_elasticsearch_data.json'
    log_entries = read_json_lines(log_file_path)
    log_data = pd.DataFrame(log_entries)

    # 适配日志数据字段
    log_data['parsed_timestamp'] = log_data['timestamp'].apply(parse_timestamp)
    log_data['template'] = extract_log_templates(log_data, 'message')

    # 向量化日志模板，只使用TF-IDF
    tfidf_matrix = tfidf_features(log_data['template'])

    # 聚类
    log_data['cluster'] = cluster_log_templates(tfidf_matrix, n_clusters=20)  # 修改聚类数为20

    # 打印聚类结果
    print("聚类结果：")
    for cluster_id in range(20):  # 修改循环范围为20
        print(f"Cluster {cluster_id}:")
        cluster_templates = log_data[log_data['cluster'] == cluster_id]['template'].unique()
        print("模板示例:")
        for template in cluster_templates[:5]:
            print(template)
        print("\n")

if __name__ == "__main__":
    main()
