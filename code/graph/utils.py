from sklearn.metrics.pairwise import cosine_similarity
from torch_geometric.data import Data
import numpy as np
import json
from collections import defaultdict
import os
from datetime import datetime, timedelta, timezone
from torch_geometric.data import Data

def invert_charges(id2idx_dict_or_path, id_filter):
    id_filter = set(id_filter)
    # 加载数据
    if isinstance(id2idx_dict_or_path, str):
        with open(id2idx_dict_or_path, 'r', encoding='utf-8') as f:
            id2idx = json.load(f)
    else:
        id2idx = id2idx_dict_or_path


    idx2ids = defaultdict(list)
    for id_, idx_val in id2idx.items():
        id_ = int(id_)
        if id_filter and id_ not in id_filter:
            continue
        
        if isinstance(idx_val, list):
            for idx in idx_val:
                idx2ids[idx].append(id_)
        else:
            idx2ids[idx_val].append(id_)
    return dict(idx2ids)


def inverse_dict(option_num, ids, rqid):
    n, k = rqid.shape
    value_options = []
    for i in range(len(option_num)):
        value_options.append(list(range(option_num[i])))


    inverse_dict_arr = {
        idx: {v: [] for v in value_options[idx]}
        for idx in range(k)
    }

    for i in range(n):
        id_value = ids[i]
        for idx, v in enumerate(rqid[i]):
            inverse_dict_arr[idx][v].append(int(id_value))

    list_lengths = []

    for key, value in inverse_dict_arr.items():
        row_lengths = []
        for subkey, subvalue in value.items():
            if isinstance(subvalue, list):
                row_lengths.append(len(subvalue))
        list_lengths.append(row_lengths)
    
    return inverse_dict_arr, list_lengths

def load_dict(path):
    with open(path, 'r') as f:
        _dict = json.load(f)
        return  {int(k): v for k, v in _dict.items()}
    
def build_save_dir():
    beijing_tz = timezone(timedelta(hours=8))
    beijing_time = datetime.now(beijing_tz)
    timestamp = beijing_time.strftime("%Y-%m-%d_%H-%M-%S")

    save_dir = os.path.join("./checkpoints", timestamp)
    os.makedirs(save_dir, exist_ok=True)
    return save_dir
