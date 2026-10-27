
from datetime import datetime
import os
import pytz


def ensure_dir(dir_path):
    os.makedirs(dir_path, exist_ok=True)

def set_color(log, color, highlight=True):
    color_set = ["black", "red", "green", "yellow", "blue", "pink", "cyan", "white"]
    try:
        index = color_set.index(color)
    except:
        index = len(color_set) - 1
    prev_log = "\033["
    if highlight:
        prev_log += "1;3"
    else:
        prev_log += "0;3"
    prev_log += str(index) + "m"
    return prev_log + log + "\033[0m"

def get_local_time():
    # 获取北京时间（中国标准时间）
    tz = pytz.timezone('Asia/Shanghai')
    local_time = datetime.now(tz)
    return local_time.strftime('%Y-%m-%d_%H-%M-%S')  # 格式化时间为字符串



