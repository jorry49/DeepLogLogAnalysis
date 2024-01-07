import subprocess
import time
import datetime
import threading
import sys

scripts_half_hour = [
    'LogCollector.py',
    'modelPrediction.py',
]

scripts_one_hour = [
    'GRUModelTrain.py',
    'modelPrediction.py'
]


def timer(stop_event, start_time, script_name):
    while not stop_event.is_set():
        elapsed_time = datetime.datetime.now() - start_time
        # 转换为总秒数
        total_seconds = int(elapsed_time.total_seconds())
        # 计算时、分、秒
        hours, remainder = divmod(total_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)

        # 格式化字符串
        elapsed_str = f"{hours:02}:{minutes:02}:{seconds:02}"
        print(f"\r{script_name} 运行时间: {elapsed_str}", end="")
        time.sleep(1)


def run_scripts(scripts):
    for script in scripts:
        stop_event = threading.Event()
        try:
            print(f"\n开始运行 {script}...")
            start_time = datetime.datetime.now()

            # 启动计时器线程
            timer_thread = threading.Thread(target=timer, args=(stop_event, start_time, script))
            timer_thread.daemon = True  # 设置为守护线程
            timer_thread.start()

            # 运行脚本并捕获输出
            completed_process = subprocess.run(['python', script], check=True, stdout=subprocess.PIPE,
                                               stderr=subprocess.PIPE, text=True)

            # 打印标准输出和标准错误
            print(completed_process.stdout)
            print(completed_process.stderr, file=sys.stderr)

            # 通知计时器线程停止
            stop_event.set()
            timer_thread.join()
            print(f"\n{script} 运行完成")
        except subprocess.CalledProcessError as e:
            # 如果脚本运行出错，也打印错误信息
            print(f"\n运行 {script} 时发生错误: {e}", file=sys.stderr)
            print(e.stdout)
            print(e.stderr, file=sys.stderr)

            # 通知计时器线程停止
            stop_event.set()
            timer_thread.join()


while True:
    current_minute = time.localtime().tm_min

    if current_minute == 0 or current_minute == 30:
        print("\n正在运行每半小时的脚本...")
        run_scripts(scripts_half_hour)

    if current_minute == 0:
        print("\n正在运行每小时的脚本...")
        run_scripts(scripts_one_hour)

    # 每分钟检查一次
    time.sleep(60)
