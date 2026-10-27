import json
import numpy as np
import jieba
from tqdm import tqdm
from rank_bm25 import BM25Okapi
from sklearn.metrics.pairwise import cosine_similarity


def build_bm25_topk(input_json, id_list, top_k):
    results = {}

    with open(input_json, 'r', encoding='utf-8') as f:
        docs_dict = json.load(f)

    # 保证 id_list 是整数列表，构造映射用的 key 列表（字符串）
    id_list = list(id_list)
    filtered_keys = [str(i) for i in id_list if str(i) in docs_dict]
    filtered_texts = [docs_dict[k] for k in filtered_keys]

    # 分词与 BM25 构建
    tokenized_corpus = [list(jieba.cut(text)) for text in tqdm(filtered_texts, desc="Tokenizing")]
    bm25 = BM25Okapi(tokenized_corpus)

    for i, query_tokens in enumerate(tokenized_corpus):
        print(i)
        query_id = int(filtered_keys[i])
        scores = bm25.get_scores(query_tokens)
        scores[i] = -1  # 排除自己

        topk_indices = np.argpartition(-scores, range(min(top_k * 5, len(scores))))[:top_k * 5]
        topk_sorted = topk_indices[np.argsort(-scores[topk_indices])]
        topk_ids = [int(filtered_keys[j]) for j in topk_sorted[:top_k]]

        results[query_id] = topk_ids

    return results


def build_emb_topk(input_emb, id_list, top_k):
    results = {}

    data_emb = np.load(input_emb)  # shape: (N, D)

    # 根据 id_list 过滤向量
    id_list = list(id_list)  # 确保是 list 类型
    filtered_emb = data_emb[id_list]  # shape: (M, D)

    # 计算相似度矩阵
    cos_sim = cosine_similarity(filtered_emb)
    np.fill_diagonal(cos_sim, -np.inf)  # 忽略自身

    # 对每一行取 top-k 相似样本（用原始 ID 表示）
    for i, sim_row in enumerate(cos_sim):
        topk_indices = np.argpartition(-sim_row, top_k)[:top_k]
        topk_sorted = topk_indices[np.argsort(-sim_row[topk_indices])]

        # 用 id_list 映射回原始 ID
        query_id = id_list[i]
        topk_ids = [id_list[j] for j in topk_sorted]

        results[query_id] = topk_ids

    return results

def save_json(data, path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    with open('/data/students/zzj/ddl/dataset/COLIEE/train/train_candidate_ids.json', 'r') as f:
        id_list_train = json.load(f)
    with open('/data/students/zzj/ddl/dataset/COLIEE/test/test_candidate_ids.json', 'r') as f:
        id_list_test = json.load(f)

    method = "emb"  # 可选 "bm25" 或 "emb"
    top_k = 1100
    
    if method == 'bm25':
        input_json_path = "COLIEE_content.json"                      # 原始文本路径
        output_topk_path_train = "feature/bm25_topk_train.json"      
        output_topk_path_test = "feature/bm25_topk_test.json"      
        print("构建 BM25 TopK 相似文档...")

        topk_dict_train = build_bm25_topk(input_json_path, id_list_train, top_k)
        topk_dict_test = build_bm25_topk(input_json_path, id_list_test, top_k)
        save_json(topk_dict_train, output_topk_path_train)
        save_json(topk_dict_test, output_topk_path_train)
        print(f"BM25 Top{top_k} 保存至 {output_topk_path_train} 和 {output_topk_path_test}")

    elif method == 'emb':
        input_emb_path = '/data/students/zzj/AAAI2026/final/COLIEE_white_emb.npy'
        output_topk_path_train = "feature/COLIEE_similar_topk_train.json"
        output_topk_path_test = "feature/COLIEE_similar_topk_test.json"
        print("构建 Embedding TopK 相似文档...")

        topk_dict_train = build_emb_topk(input_emb_path, id_list_train, top_k)
        topk_dict_test = build_emb_topk(input_emb_path, id_list_test, top_k)

        save_json(topk_dict_train, output_topk_path_train)
        save_json(topk_dict_test, output_topk_path_test)
        print(f"Embedding Top{top_k} 保存至 {output_topk_path_train}和{output_topk_path_test}")
