import json
import re
if __name__ == "__main__":
    # 输入文件路径和输出文件路径
    input_file_path = 'COLIEE_content.json'

    sentence_output_file_path = 'COLIEE_sentence.json'
    all_ids_output_file_path = 'COLIEE_all_ids.json'

    # 创建一个新的字典来存储转换后的数据
    new_data = {}
    relation_data = {}
    negtive_case = {}
    all_ids = []

    # 逐行读取JSON数据
    with open(input_file_path, 'r', encoding='utf-8') as f:
        coliee_data = json.load(f)
    
    for k, v in coliee_data.items():
        all_ids.append(k)
        sentences = re.split(r'\[\d+\]', v)  # 按照 [数字] 切分
        sentences = [sentence.strip() for sentence in sentences if sentence.strip()]  # 去除每段的前后空白字符，并过滤空的段落
        
        new_data[k] = sentences

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

    print("处理完成，所有键已转换为数字并保存。")
