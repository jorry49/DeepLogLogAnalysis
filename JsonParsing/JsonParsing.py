import json
import numpy as np  # 确保导入了 numpy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from drain3 import TemplateMiner
from drain3.template_miner_config import TemplateMinerConfig
from sklearn.feature_extraction.text import CountVectorizer

# 直接加载 JSON 数据
def load_json(filepath):
    with open(filepath, 'r', encoding='utf-8') as file:
        return json.load(file)


# 使用 drain3 提取模板
def extract_templates(data):
    config = TemplateMinerConfig()
    template_miner = TemplateMiner(config=config)
    templates = []
    for entry in data:
        message = entry['message']
        result = template_miner.add_log_message(message)
        templates.append(result.get("template_mined"))
    return templates


# 加载 JSON 数据
data = load_json('Resource/syslog.json')
# 提取模板
templates = extract_templates(data)

# 过滤掉 None 值
filtered_templates = [template for template in templates if template]

# 转换为 TF-IDF 特征
vectorizer = TfidfVectorizer(max_features=1000)
X = vectorizer.fit_transform(filtered_templates)

# 应用 K-means 聚类
kmeans = KMeans(n_clusters=5, random_state=0)
kmeans.fit(X)



# 分配的聚类标签
labels = kmeans.labels_

# 输出聚类结果及分析
for i in range(kmeans.n_clusters):
    print(f"Cluster {i}:")
    cluster_indices = [j for j, label in enumerate(labels) if label == i]

    # 聚类的日志模板示例
    cluster_examples = [filtered_templates[index] for index in cluster_indices]

    # 分析每个聚类中的常见术语
    vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
    X_cluster = vectorizer.fit_transform(cluster_examples)
    X_cluster_array = np.asarray(X_cluster.sum(axis=0))
    common_terms = vectorizer.inverse_transform(X_cluster_array)[0]

    # 打印动态生成的聚类标题
    print("这个聚类包括以下关键词: " + ", ".join(common_terms[:5]))

    # 打印每个聚类的前几个模板作为示例
    for example in cluster_examples[:5]:  # 只显示每个聚类的前5个模板
        print(example)

    print("\n")

# 这里可以保存聚类结果，以便进一步分析
# 例如，将聚类标签和模板写入文件
cluster_results = [{'template': template, 'cluster_label': str(label)} for template, label in
                   zip(filtered_templates, labels)]
with open('Resource/cluster_results.json', 'w', encoding='utf-8') as f:
    json.dump(cluster_results, f, ensure_ascii=False, indent=4)

# 使用 CountVectorizer 对模板进行标记化和构建词汇表
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(filtered_templates)

# 查看词汇表
vocabulary = vectorizer.get_feature_names_out()
print("词汇表:")
print(vocabulary)

# 查看数值编码的结果
print("数值编码的模板:")
print(X.toarray())

# 如果需要对特定聚类进一步分析
for cluster_num in range(kmeans.n_clusters):
    print(f"Cluster {cluster_num}:")
    cluster_indices = [j for j, label in enumerate(labels) if label == cluster_num]

    # 提取特定聚类的模板
    cluster_templates = [filtered_templates[index] for index in cluster_indices]

    # 使用同一个向量化器来转换这些模板
    X_cluster = vectorizer.transform(cluster_templates)

    # 查看特定聚类中的模板的数值编码
    print(f"聚类 {cluster_num} 的数值编码:")
    print(X_cluster.toarray())
    print("\n")

# 计算每个单词的平均 TF-IDF 得分（应该在循环外部）
average_tfidf_scores = X.mean(axis=0)
words = vectorizer.get_feature_names_out()
word_scores = zip(words, average_tfidf_scores.tolist()[0])

# 排序并打印得分最高的单词
sorted_word_scores = sorted(word_scores, key=lambda x: x[1], reverse=True)
print("平均 TF-IDF 得分最高的单词:")
for word, score in sorted_word_scores[:10]:  # 打印前10个
    print(f"{word}: {score}")