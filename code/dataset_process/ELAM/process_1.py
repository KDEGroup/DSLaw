import json

if __name__ == "__main__":
    # 输入文件路径和输出文件路径
    input_file_path = 'ELAM.json'
    sentence_output_file_path = 'ELAM_sentence.json'
    content_output_file_path = 'ELAM_content.json'
    relative_output_file_path = 'ELAM_relative.json'
    all_ids_output_file_path = 'ELAM_all_ids.json'

    # 创建一个新的字典来存储转换后的数据
    new_data = {}
    relation_data = {}
    negtive_case = {}
    all_ids = []

    # 逐行读取JSON数据
    with open(input_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                entry = json.loads(line.strip())
                case_a = entry.get("Case_A", [])
                case_b = entry.get("Case_B", [])
                label = entry.get("label", 0)

                # 分别将Case_A和Case_B的ID映射到新的数据结构中
                id_a, id_b = entry["id"].split('|')  # 假设id格式为 "aaa|bbb"

                # 如果id_a不重复, 将它赋值给Case_A
                if id_a not in new_data:
                    new_data[id_a] = case_a
                    relation_data[id_a] = []
                    negtive_case[id_a] = []
                    all_ids.append(id_a)

                # 如果id_b不重复, 将它赋值给Case_B
                if id_b not in new_data:
                    new_data[id_b] = case_b
                    relation_data[id_b] = []
                    negtive_case[id_b] = []
                    all_ids.append(id_b)

                # 标签大于0时放入正关系，否则放入负关系
                if label > 0:
                    relation_data[id_a].append(id_b)
                    relation_data[id_b].append(id_a)
                else:
                    negtive_case[id_a].append(id_b)
                    negtive_case[id_b].append(id_a)

            except json.JSONDecodeError as e:
                print(f"无法解析这一行: {line}")
                continue

    # 按字母顺序排序所有ID，并生成ID到数字的映射
    all_ids = sorted(all_ids)
    ids_dict = {id_: idx for idx, id_ in enumerate(all_ids)}

    # 保存all_ids映射
    with open(all_ids_output_file_path, 'w', encoding='utf-8') as f:
        json.dump(ids_dict, f, ensure_ascii=False, indent=4)

    # 函数：将键从原始ID转换为数字ID
    def remap_keys(data_dict):
        return {ids_dict[key]: data_dict[key] for key in sorted(data_dict.keys())}

    # 保存句子（原结构），并将ID转换为数字
    sentence_data_remapped = remap_keys(new_data)
    with open(sentence_output_file_path, 'w', encoding='utf-8') as f:
        json.dump(sentence_data_remapped, f, ensure_ascii=False, indent=4)

    # 合并句子并保存内容，同时转换ID
    content_data = {k: ''.join(v) for k, v in new_data.items()}
    content_data_remapped = {ids_dict[k]: content_data[k] for k in sorted(content_data.keys())}
    with open(content_output_file_path, 'w', encoding='utf-8') as f:
        json.dump(content_data_remapped, f, ensure_ascii=False, indent=4)

    # 保存关系数据，并将ID转换为数字
    relation_data_remapped = {ids_dict[k]: [ids_dict[x] for x in v] for k, v in sorted(relation_data.items())}
    with open(relative_output_file_path, 'w', encoding='utf-8') as f:
        json.dump(relation_data_remapped, f, ensure_ascii=False, indent=4)

    print("处理完成，所有键已转换为数字并保存。")
