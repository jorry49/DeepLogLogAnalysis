import os
import json
import requests
import time
from datetime import datetime, timedelta
from requests.auth import HTTPBasicAuth

# 配置
data_folder = "******************"
timestamp_file = os.path.join(data_folder, "last_timestamp.txt")
elasticsearch_url = "https://******************"
elasticsearch_auth = HTTPBasicAuth("*********", "**************")
start_date = datetime(2023, 12, 3)  # 日志开始日期
interval_seconds = 3600  # 设定时间间隔为1小时

# 禁用不安全HTTPS请求警告
requests.packages.urllib3.disable_warnings(requests.packages.urllib3.exceptions.InsecureRequestWarning)

def read_last_timestamp():
    if not os.path.exists(data_folder):
        os.makedirs(data_folder)
    if os.path.exists(timestamp_file):
        with open(timestamp_file, "r") as file:
            last_timestamp = file.read().strip()
            return last_timestamp if last_timestamp else start_date.isoformat()
    else:
        with open(timestamp_file, "w") as file:  # 创建文件
            file.write(start_date.isoformat())
        return start_date.isoformat()

def update_timestamp_file(new_timestamp):
    with open(timestamp_file, "w") as file:
        file.write(new_timestamp)

def fetch_and_clean_logs(start_date, end_date, last_timestamp):
    filename = os.path.join(data_folder, "cleaned_elasticsearch_data.json")
    cleaned_data_list = []  # 用于存储清洗后的日志数据

    # 遍历日期范围
    for single_date in (start_date + timedelta(n) for n in range(int((end_date - start_date).days) + 1)):
        date_str = single_date.strftime("%Y.%m.%d")
        url = f"{elasticsearch_url}/logstash-syslog-{date_str}/_search"
        query = {
            "query": {
                "range": {
                    "@timestamp": {
                        "gte": last_timestamp,  # 使用最后的时间戳
                        "lt": end_date.isoformat()  # 查询小于当前结束日期的日志
                    }
                }
            }
        }
        headers = {"Content-Type": "application/json"}

        try:
            response = requests.get(url, headers=headers, data=json.dumps(query),
                                    auth=elasticsearch_auth, verify=False)
            if response.status_code == 200:
                data = response.json()
                cleaned_data = clean_data(data)
                if cleaned_data:  # 有数据才写入
                    cleaned_data_list.extend(cleaned_data)  # 将清洗后的数据添加到列表中
                    # 更新时间戳为最后一条日志的时间
                    update_timestamp_file(cleaned_data[-1]['timestamp'])
            else:
                print(f"{date_str} 获取日志错误: {response.status_code} - {response.text}")
        except requests.RequestException as e:
            print(f"{date_str} 请求失败: {e}")

    # 将清洗后的数据写入JSON文件
    if cleaned_data_list:
        with open(filename, "a") as file:
            for entry in cleaned_data_list:
                file.write(json.dumps(entry) + '\n')

def clean_data(data):
    cleaned_data = []
    for log_entry in data['hits']['hits']:
        source = log_entry['_source']
        cleaned_entry = {
            "timestamp": source.get("@timestamp"),
            "event": source.get("event", {}).get("original"),
            "message": source.get("message"),
            "host": source.get("host", {}).get("hostname"),
            "ip": source.get("host", {}).get("ip"),
            "service_type": source.get("service", {}).get("type"),
            "process_name": source.get("process", {}).get("name"),
            "process_pid": source.get("process", {}).get("pid"),
            "log_details": source.get("log", {})
        }
        cleaned_data.append(cleaned_entry)
    return cleaned_data

def main():
    last_timestamp = read_last_timestamp()  # 读取上一次的时间戳
    while True:
        start_time = time.time()
        current_end_date = datetime.now()  # 更新结束日期为当前时间
        fetch_and_clean_logs(start_date, current_end_date, last_timestamp)
        last_timestamp = current_end_date.isoformat()  # 更新时间戳为当前查询的时间
        elapsed_time = time.time() - start_time
        print(f"\r当前时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}，已运行时间：{elapsed_time:.2f}秒", end='')
        time.sleep(interval_seconds)

if __name__ == "__main__":
    main()
