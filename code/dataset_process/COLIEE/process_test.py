import json
import numpy as np

def filter_dict_by_id_list(data_dict, id_list):
    """
    根据 id_list 筛选字典，保留键和值都在 id_list 中的项
    """
    return {
        key: [val for val in values if val in id_list]
        for key, values in data_dict.items()
        if int(key) in id_list  # 把 key 转换为整数进行比较
    }

def filter_dict_key_by_idlist(input_json_path, idlist, output_json_path):
    # 读取原始 JSON 文件
    with open(input_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 根据 idlist 筛选出相应的键值对，保持顺序不变
    return {
        key: data[str(key)] for key in idlist if str(key) in data
    }

def delete_values_from_dict(filtered_dict, relative_json):
    """
    根据 new_json 中的键值对，删除 filtered_dict 中对应的值
    """
    for key, val in relative_json.items():
        if key in filtered_dict:
            filtered_dict[key] = [item for item in filtered_dict[key] if item not in val]
    return filtered_dict

def limit_values_to_k(filtered_dict, k):
    """
    限制每个键的值列表最多包含前 k 个值
    """
    for key in filtered_dict:
        filtered_dict[key] = filtered_dict[key][:k]
    return filtered_dict

def select_values_and_save_to_npy(old_emb_path, id_list_path, output_npy_path):
    """
    根据 id_list 从 old_emb 中选择对应的值，并保存为 .npy 文件
    """
    old_emb = np.load(old_emb_path)

    # 读取 id_list 文件
    with open(id_list_path, 'r', encoding='utf-8') as f:
        id_list = json.load(f)

    selected_values = []

    for id_value in id_list:
        selected_values.append(old_emb[id_value])

    # 保存为 .npy 文件
    np.save(output_npy_path, selected_values)

    print(f"已将选中的值保存至 {output_npy_path}")

if __name__ == "__main__":
    # 输入输出路径设置
    similarity_path = 'feature/COLIEE_similar_topk_test.json'
    id_list_path = 'test/test_candidate_ids.json'
    relative_json_path = 'COLIEE_relative.json'  # 原相关字典文件路径
    # old_emb_path = 'COLIEE_emb.npy'
    old_emb_path = '/data/students/zzj/AAAI2026/final/COLIEE_white_emb.npy'

    test_id_path = 'test/test_ids.json'
    with open(test_id_path, 'r', encoding='utf-8') as f:
        test_id = set(json.load(f))  # 使用 set 提高查找效率

    output_top_k_json_path = 'test/test_hard_neg.json'  # 保存更新后的文件路径
    output_relative_path = 'test/test_relative.json'
    output_npy_path = 'test/test_candidate_emb.npy'  # 保存选中值的 .npy 文件路径

    k = 100  # 限制值的最大数量

    # 读取文件
    with open(similarity_path, 'r', encoding='utf-8') as f:
        old_json = json.load(f)

    with open(id_list_path, 'r', encoding='utf-8') as f:
        id_list = set(json.load(f))  # 使用 set 提高查找效率

    with open(relative_json_path, 'r', encoding='utf-8') as f:
        relative_json = json.load(f)

    # 过滤字典、删除指定值、限制每个键的最大值数量
    updated_dict = delete_values_from_dict(old_json, relative_json)
    updated_dict = limit_values_to_k(updated_dict, k)


    # 保存更新后的字典到文件
    with open(output_top_k_json_path, 'w', encoding='utf-8') as f:
        json.dump(updated_dict, f, ensure_ascii=False, indent=2)

    # 处理 relative_json 数据
    relative_data = filter_dict_by_id_list(relative_json, id_list)
    with open(output_relative_path, 'w', encoding='utf-8') as f:
        json.dump(relative_data, f, ensure_ascii=False, indent=2)

    print(f"更新后的字典已保存至 {output_top_k_json_path}, {output_relative_path}")

    # 执行选择值并保存为 .npy 文件
    select_values_and_save_to_npy(old_emb_path, test_id_path, output_npy_path)
    


