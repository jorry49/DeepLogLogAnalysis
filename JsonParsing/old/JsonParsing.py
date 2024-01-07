#解析日志文件
import json
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec
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


# Word2Vec 特征提取
def word2vec_features(templates):
    tokenized_templates = [template.split() for template in templates]
    model = Word2Vec(tokenized_templates, vector_size=100, window=5, min_count=1, workers=4)
    word2vec_matrix = np.array(
        [np.mean([model.wv[word] for word in template if word in model.wv], axis=0) for template in
         tokenized_templates])
    return word2vec_matrix


# 聚类日志模板
def cluster_log_templates(features, n_clusters=10):
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    clusters = kmeans.fit_predict(features)
    return clusters


# 主函数
def main():
    log_file_path = 'Resource/data/elasticsearch_data/elasticsearch_data.json'
    log_entries = read_json_lines(log_file_path)
    log_data = pd.DataFrame(log_entries)

    log_data['parsed_timestamp'] = log_data['@timestamp'].apply(parse_timestamp)
    log_data['template'] = extract_log_templates(log_data, 'message')

    # 向量化日志模板
    tfidf_matrix = tfidf_features(log_data['template'])
    word2vec_matrix = word2vec_features(log_data['template'])

    # 联合 TF-IDF 和 Word2Vec 特征
    combined_features = np.concatenate((tfidf_matrix, word2vec_matrix), axis=1)

    # 聚类
    log_data['cluster'] = cluster_log_templates(combined_features, n_clusters=10)

    # 打印聚类结果
    print("聚类结果：")
    for cluster_id in range(10):
        print(f"Cluster {cluster_id}:")
        cluster_templates = log_data[log_data['cluster'] == cluster_id]['template'].unique()
        print("模板示例:")
        for template in cluster_templates[:5]:
            print(template)
        print("\n")


if __name__ == "__main__":
    main()
