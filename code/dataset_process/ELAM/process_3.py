import json
import numpy as np
import jieba
from rank_bm25 import BM25Okapi
from sklearn.metrics.pairwise import cosine_similarity


def build_bm25_topk(input_json, top_k):
    with open(input_json, 'r', encoding='utf-8') as f:
        docs_dict = json.load(f)

    keys = list(docs_dict.keys())
    texts = list(docs_dict.values())

    tokenized_corpus = [list(jieba.cut(doc)) for doc in texts]
    bm25 = BM25Okapi(tokenized_corpus)

    results = {}
    for idx, key in enumerate(keys):
        query_tokens = tokenized_corpus[idx]
        scores = bm25.get_scores(query_tokens)
        scores[idx] = -1  # 排除自己

        top_indices = np.argpartition(-scores, range(top_k))[:top_k]
        top_indices = top_indices[np.argsort(-scores[top_indices])]

        related_ids = [int(keys[i]) for i in top_indices]
        results[key] = related_ids

    return results

def build_emb_topk(input_emb, top_k):
    results = {}
    
    # 读取 .npy 文件
    data_emb = np.load(input_emb)  # 直接加载为 numpy 数组

    # 计算余弦相似度
    cos_sim = cosine_similarity(data_emb)
    np.fill_diagonal(cos_sim, -np.inf)  # 排除自身

    # 对每一行取 top_k 相似节点索引
    for idx, row in enumerate(cos_sim):
        topk_indices = np.argpartition(-row, top_k)[:top_k]  # 获取 top_k 结果
        topk_sorted = topk_indices[np.argsort(-row[topk_indices])]  # 排序索引
        topk_keys = topk_sorted.tolist()

        results[idx] = topk_keys  # 保存 top_k 相似项的索引

    return results

def save_json(data, path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    method = "emb"  # 可选 "bm25" 或 "emb"
    top_k = 500
    
    if method == 'bm25':
        input_json_path = "ELAM_content.json"                      # 原始文本路径
        output_topk_path = "feature/bm25_topk.json"               # 保存 BM25 相似 TopK

        print("构建 BM25 TopK 相似文档...")
        topk_dict = build_bm25_topk(input_json_path, top_k)
        save_json(topk_dict, output_topk_path)
        print(f"BM25 Top{top_k} 保存至 {output_topk_path}")

    elif method == 'emb':
        # input_emb_path = 'ELAM_emb.npy'
        input_emb_path = '/data/students/zzj/AAAI2026/final/ELAM_white_emb.npy'
        output_topk_path = "feature/ELAM_similar_topk.json"
        print("构建 Embedding TopK 相似文档...")
        topk_dict = build_emb_topk(input_emb_path, top_k)
        save_json(topk_dict, output_topk_path)
        print(f"Embedding Top{top_k} 保存至 {output_topk_path}")


