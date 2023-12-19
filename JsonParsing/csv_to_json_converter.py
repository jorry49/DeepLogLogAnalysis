import csv
import json

def convert_csv_to_json(csv_filepath, json_filepath):
    # 读取 CSV 文件并将每行转换为字典
    csv_data = []
    with open(csv_filepath, mode='r', encoding='utf-8') as csvfile:
        csvreader = csv.DictReader(csvfile)
        for row in csvreader:
            csv_data.append(row)

    # 将字典列表写入 JSON 文件
    with open(json_filepath, mode='w', encoding='utf-8') as jsonfile:
        json.dump(csv_data, jsonfile, ensure_ascii=False, indent=4)

# 指定 CSV 文件和将要保存 JSON 数据的文件路径
csv_filepath = 'Resource/routineInspection.csv'
json_filepath = 'Resource/syslog.json'

convert_csv_to_json(csv_filepath, json_filepath)
