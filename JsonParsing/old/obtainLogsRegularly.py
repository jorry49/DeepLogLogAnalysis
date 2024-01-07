#日志收集
import schedule
import time
import requests
import json
import os
import urllib3
from datetime import datetime, timedelta
from requests.auth import HTTPBasicAuth
import logging

# 禁用不安全请求警告
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# 配置
data_folder = "Resource/data/elasticsearch_data"
timestamp_file = os.path.join(data_folder, "last_timestamp.txt")
elasticsearch_url = "https://124.221.91.139:9200"
elasticsearch_auth = HTTPBasicAuth("elastic", "Zs020609")

logging.basicConfig(filename='log_collector.log', level=logging.INFO)

def read_last_timestamp():
    try:
        if os.path.exists(timestamp_file):
            with open(timestamp_file, "r") as file:
                last_timestamp = file.read().strip()
                return last_timestamp if last_timestamp else None
    except Exception as e:
        logging.error(f"读取时间戳文件失败: {e}")
    return None

def update_timestamp_file(new_timestamp):
    try:
        with open(timestamp_file, "w") as file:
            file.write(new_timestamp)
        logging.info("时间戳已更新")
    except Exception as e:
        logging.error(f"更新时间戳文件失败: {e}")

def fetch_elasticsearch_data(url, query, index_date):
    headers = {
        "Content-Type": "application/json"
    }

    try:
        response = requests.get(url, headers=headers, data=json.dumps(query),
                                auth=elasticsearch_auth, verify=False)
        if response.status_code == 200:
            data = response.json()
            return save_data(data)
        else:
            logging.error(f"错误：{index_date} - {response.status_code} - {response.text}")
    except requests.RequestException as e:
        logging.error(f"请求失败：{index_date} - {e}")

def save_data(data):
    if not os.path.exists(data_folder):
        os.makedirs(data_folder)

    filename = os.path.join(data_folder, "elasticsearch_data.json")
    changes_detected = False
    with open(filename, "a") as file:
        for hit in data.get('hits', {}).get('hits', []):
            data_json = json.dumps(hit.get('_source', {}))
            file.write(data_json + '\n')
            if "data_json" not in globals() or data_json != globals()["data_json"]:
                changes_detected = True
                globals()["data_json"] = data_json
    return changes_detected

def print_status(changes_detected):
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    execution_time = time.time() - start_time

    # 使用 ANSI 控制码将光标移动到行首
    print(f"\r当前时间：{current_time}", end='')
    print(f"运行时间：{execution_time:.2f} 秒", end='')

    if changes_detected:
        print("\n在Elasticsearch数据中检测到变化。")
    else:
        print("\n在Elasticsearch数据中未检测到变化.")

# 定义时间框架函数
def last_hour_time_frame(current_date):
    return current_date - timedelta(hours=1)

def form_query(index_date, last_timestamp):
    idx_name = f"logstash-syslog-{index_date.strftime('%Y.%m.%d')}"
    url = f"{elasticsearch_url}/{idx_name}/_search"
    if last_timestamp:
        query = {
            "query": {
                "range": {
                    "@timestamp": {
                        "gte": last_timestamp,
                        "lt": index_date.strftime("%Y-%m-%dT%H:%M:%S")
                    }
                }
            }
        }
    else:
        query = {"query": {"match_all": {}}}
    return url, query

def fetch_data_for_time_frame(time_frame_func, last_timestamp):
    global start_time
    start_time = time.time()

    current_date = datetime.now()
    index_date = time_frame_func(current_date)

    url, query = form_query(index_date, last_timestamp)

    changes_detected = fetch_elasticsearch_data(url, query, index_date.strftime("%Y.%m.%d"))
    print_status(changes_detected)

    if changes_detected:
        update_timestamp_file(current_date.strftime("%Y-%m-%dT%H:%M:%S.%fZ"))

# 新增 fetch_data_last_hour 函数
def fetch_data_last_hour():
    logging.info("执行每1分钟更新...")
    last_timestamp = read_last_timestamp()
    fetch_data_for_time_frame(last_hour_time_frame, last_timestamp)

def fetch_initial_data():
    logging.info("正在获取所有日志...")
    last_timestamp = read_last_timestamp()
    start_date = datetime.strptime(last_timestamp, "%Y-%m-%dT%H:%M:%S.%fZ") if last_timestamp else datetime(2023, 12, 3)
    end_date = datetime.now()
    delta = end_date - start_date

    for i in range(delta.days + 1):
        day = start_date + timedelta(days=i)
        url, query = form_query(day, last_timestamp)
        changes_detected = fetch_elasticsearch_data(url, query, day.strftime("%Y.%m.%d"))

    # 检查时间戳是否更新
    update_timestamp_file(end_date.strftime("%Y-%m-%dT%H:%M:%S.%fZ"))

    logging.info("已完成收集旧日志。")
    logging.info("正在获取新日志...")

# 获取初始化数据
def run_log_collector():
    if not os.path.exists(timestamp_file):
        # 如果last_timestamp.txt不存在，创建一个并设置初始时间
        current_time = datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%fZ")
        update_timestamp_file(current_time)
    fetch_initial_data()

# 定时任务
schedule.every(1).minutes.do(fetch_data_last_hour)

while True:
    schedule.run_pending()
    time.sleep(1)
