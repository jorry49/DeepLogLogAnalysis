#划分数据集
import json
import os
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from drain3 import TemplateMiner
from drain3.template_miner_config import TemplateMinerConfig
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec
import numpy as np

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
def cluster_log_templates(templates, n_clusters=10):
    encoder = LabelEncoder()
    encoded_templates = encoder.fit_transform(templates)
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    clusters = kmeans.fit_predict(encoded_templates.reshape(-1, 1))
    return clusters, encoder.classes_

# 特征提取: TF-IDF
def tfidf_features(templates):
    vectorizer = TfidfVectorizer(max_features=1000)
    tfidf_matrix = vectorizer.fit_transform(templates)
    return tfidf_matrix.toarray()

# 特征提取: Word2Vec
def word2vec_features(templates):
    tokenized_templates = [template.split() for template in templates]
    model = Word2Vec(tokenized_templates, vector_size=100, window=5, min_count=1, workers=4)
    word2vec_matrix = np.array(
        [np.mean([model.wv[word] for word in template if word in model.wv], axis=0)
         for template in tokenized_templates])
    return word2vec_matrix

# 结合TF-IDF和Word2Vec特征
def combine_features(tfidf_matrix, word2vec_matrix):
    combined_features = np.concatenate((tfidf_matrix, word2vec_matrix), axis=1)
    return combined_features

# 创建序列
def create_sequences(tfidf_matrix, window_size=5):
    sequences = []
    for i in range(len(tfidf_matrix) - window_size + 1):
        sequences.append(tfidf_matrix[i:i + window_size])
    return np.array(sequences)

# 分割数据集
def split_datasets(sequences, labels):
    X_train, X_test, y_train, y_test = train_test_split(sequences, labels, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

def assign_labels(clusters):
    labels = [1 if cluster in [3, 5] else 0 for cluster in clusters]
    return labels

def main():
    log_file_path = 'Resource/data/elasticsearch_data/elasticsearch_data.json'
    log_entries = read_json_lines(log_file_path)
    log_data = pd.DataFrame(log_entries)

    log_data['parsed_timestamp'] = log_data['@timestamp'].apply(parse_timestamp)
    log_data['template'] = extract_log_templates(log_data, 'message')
    clusters, _ = cluster_log_templates(log_data['template'], 10)
    log_data['label'] = assign_labels(clusters)

    tfidf_matrix = tfidf_features(log_data['template'])
    word2vec_matrix = word2vec_features(log_data['template'])
    combined_features = combine_features(tfidf_matrix, word2vec_matrix)

    # 标准化特征
    scaler = StandardScaler()
    standardized_features = scaler.fit_transform(combined_features)

    sequences = create_sequences(standardized_features, window_size)
    labels = log_data['label'].values[window_size-1:]

    X_train, X_test, y_train, y_test = split_datasets(sequences, labels)

    dataset_path = '../Resource/dataSet'
    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)

    np.save(os.path.join(dataset_path, 'X_train.npy'), X_train)
    np.save(os.path.join(dataset_path, 'X_test.npy'), X_test)
    np.save(os.path.join(dataset_path, 'y_train.npy'), y_train)
    np.save(os.path.join(dataset_path, 'y_test.npy'), y_test)

if __name__ == "__main__":
    main()
