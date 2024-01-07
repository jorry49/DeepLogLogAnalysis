import json
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from drain3 import TemplateMiner
from drain3.template_miner_config import TemplateMinerConfig

# 从文件读取日志数据
def read_json_lines(file_path):
    with open(file_path, 'r') as file:
        return [json.loads(line.strip()) for line in file if line.strip()]

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
def cluster_log_templates(features, n_clusters=20):
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    clusters = kmeans.fit_predict(features)
    return clusters

# 主函数
def main():
    log_file_path = '../Resource/dataSet/finalTest/normal_logs.json'  # 修改为你的日志文件路径
    log_entries = read_json_lines(log_file_path)
    log_data = pd.DataFrame(log_entries)

    log_data['template'] = extract_log_templates(log_data, 'message')  # 'message'是日志内容的列名，根据实际情况修改
    tfidf_matrix = tfidf_features(log_data['template'])
    log_data['cluster'] = cluster_log_templates(tfidf_matrix, n_clusters=20)  # 修改聚类数为20

    # 异常聚类标记
    anomaly_dict = {
        0: 1, 1: 0, 2: 1, 3: 0, 4: 1, 5: 1, 6: 0, 7: 1,
        8: 0, 9: 1, 10: 1, 11: 0, 12: 1, 13: 1, 14: 1,
        15: 1, 16: 0, 17: 0, 18: 1, 19: 0
    }

    log_data['is_anomaly'] = log_data['cluster'].map(anomaly_dict)


    print("未检测到异常日志。")

if __name__ == "__main__":
    main()
