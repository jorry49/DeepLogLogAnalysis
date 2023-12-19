import pandas as pd
import datetime
from sklearn.preprocessing import LabelEncoder
from drain3 import TemplateMiner
from drain3.template_miner_config import TemplateMinerConfig

# 读取CSV文件
def load_csv(filepath):
    return pd.read_csv(filepath)

# 解析时间戳
def parse_timestamp(timestamp):
    return datetime.datetime.strptime(timestamp, '%b %d, %Y @ %H:%M:%S.%f')

# 使用 drain3 提取日志模板，并预处理 message 字段
def extract_log_templates(data, message_column):
    config = TemplateMinerConfig()
    template_miner = TemplateMiner(config=config)
    templates = []
    for message in data[message_column]:
        # 确保 message 是字符串类型
        if not isinstance(message, str):
            message = str(message)
        result = template_miner.add_log_message(message)
        templates.append(result.get("template_mined"))
    return templates

# 生成序列化数据集
def generate_sequences(data, time_window_minutes, timestamp_col, event_col):
    sequences = []
    start_time = data[timestamp_col].min()
    end_time = start_time + datetime.timedelta(minutes=time_window_minutes)

    while start_time < data[timestamp_col].max():
        window_data = data[(data[timestamp_col] >= start_time) &
                           (data[timestamp_col] < end_time)]
        if not window_data.empty:
            sequences.append(list(window_data[event_col]))
        start_time = end_time
        end_time += datetime.timedelta(minutes=time_window_minutes)

    return sequences

# 主程序
def main():
    csv_file_path = 'Resource/routineInspection.csv'
    log_data = load_csv(csv_file_path)

    # 解析时间戳
    log_data['parsed_timestamp'] = log_data['@timestamp'].apply(parse_timestamp)

    # 使用drain3提取日志模板
    log_data['template'] = extract_log_templates(log_data, 'message')

    # 对提取的模板进行编码
    encoder = LabelEncoder()
    log_data['event_code'] = encoder.fit_transform(log_data['template'])

    # 生成序列化数据集
    time_window_minutes = 5  # 可以根据需要调整时间窗口大小
    sequences = generate_sequences(log_data, time_window_minutes, 'parsed_timestamp', 'event_code')

    # 保存序列化数据集
    output_file = 'Resource/data/event_sequences.txt'
    with open(output_file, 'w') as file:
        for sequence in sequences:
            sequence_str = ' '.join(map(str, sequence))
            file.write(f"{sequence_str}\n")

    print(f"Total sequences created: {len(sequences)}")
    print(f"Sequence data saved to {output_file}")

if __name__ == "__main__":
    main()
