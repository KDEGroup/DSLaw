import json
import jieba
import math
import os
import numpy as np
import torch
from collections import defaultdict
from transformers import AutoTokenizer, AutoModel

# ==================== 分段模块 ====================

def chunk_text(text, window_size=200, overlap=10):
    chunks = []
    start = 0
    length = len(text)

    while start < length:
        end = min(start + window_size, length)
        segment = text[start:end]

        last_period = segment.rfind('。')
        if last_period != -1:
            end = start + last_period + 1
            chunk = text[start:end]
            chunks.append(chunk)
            start = end
        else:
            chunk = text[start:end]
            chunks.append(chunk)
            if end == length:
                start = end
            else:
                start = end - overlap if end - overlap > start else end

    return chunks

def chunk_json_file_and_get_min_len(input_path, output_path, window_size=200, overlap=10):
    with open(input_path, 'r', encoding='utf-8') as f:
        json_dict = json.load(f)

    chunked_dict = {}
    min_len = None

    for key, text in json_dict.items():
        chunks = chunk_text(text, window_size, overlap)
        chunked_dict[key] = chunks
        if min_len is None or len(chunks) < min_len:
            min_len = len(chunks)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(chunked_dict, f, ensure_ascii=False, indent=2)

    return min_len

# ==================== 打分模块 ====================

def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_json(data, path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def load_important_words(path):
    words = set()
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            word = line.strip().strip('"').strip("'")
            if word:
                words.add(word)
    print(f"加载重要词汇共{len(words)}个，前10个示例：{list(words)[:10]}")
    return words

def compute_df(sentence_dict):
    df = defaultdict(int)
    total_docs = 0
    for sentences in sentence_dict.values():
        for sent in sentences:
            total_docs += 1
            tokens = set(jieba.lcut(sent))
            for token in tokens:
                df[token] += 1
    return df, total_docs

def compute_score1(tokens, important_words):
    return sum(1 for token in tokens if token in important_words)

def compute_score2(tokens, idf):
    if not tokens:
        return 0.0
    tf = defaultdict(int)
    for token in tokens:
        tf[token] += 1
    length = len(tokens)
    score = 0.0
    for token, freq in tf.items():
        token_idf = idf.get(token, 0.0)
        score += (freq / length) * token_idf
    return score

def score_sentences(sentence_path, words_path, output_path, alpha=1, beta=1, add_keyword_summary=True, deduplicate_keywords=True):
    sentence_dict = load_json(sentence_path)
    important_words = load_important_words(words_path)

    df, total_docs = compute_df(sentence_dict)
    idf = {token: math.log(total_docs / (freq + 1)) for token, freq in df.items()}

    scored_dict = {}
    for key, sentences in sentence_dict.items():
        matched_keywords = []
        scored_sentences = []

        for sent in sentences:
            tokens = jieba.lcut(sent)

            # 收集关键词
            matched = [token for token in tokens if token in important_words]
            matched_keywords.extend(matched)

            # 打分
            score1 = compute_score1(tokens, important_words)
            score2 = compute_score2(tokens, idf)
            score3 = alpha * score1 + beta * score2

            scored_sentences.append({
                "sentence": sent,
                "score1": score1,
                "score2": score2,
                "score3": score3
            })
        scored_sentences = sorted(scored_sentences, key=lambda x: x['score3'], reverse=True)

        if add_keyword_summary:
            if deduplicate_keywords:
                matched_keywords = sorted(set(matched_keywords))
            keyword_sentence = " ".join(matched_keywords)
            summary_item = {
                "sentence": keyword_sentence,
                "score1": len(matched_keywords),
                "score2": 0.0,
                "score3": 0.0
            }
            scored_sentences.insert(0, summary_item)
        scored_dict[key] = scored_sentences

    save_json(scored_dict, output_path)
    print(f"评分完成，{'已添加关键词句子，' if add_keyword_summary else ''}结果已保存到：{output_path}")


# ==================== TopK提取模块 ====================

def extract_first_k_values(input_path, output_path, k=3):
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    result = {}
    for key, dict_list in data.items():
        values = []
        for i in range(k):
            if i < len(dict_list):
                first_key = next(iter(dict_list[i]))
                values.append(dict_list[i][first_key])
            else:
                values.append("")  # 不足k项填充空字符串
        result[key] = values

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"已提取前 {k} 项（不足填空），保存到：{output_path}")

# # ==================== 编码模块 ====================

# def mean_pooling(model_output, attention_mask):
#     token_embeddings = model_output[0]  # last_hidden_state
#     input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
#     sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
#     sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
#     return sum_embeddings / sum_mask

# def encode_sentences(model, input_path, output_npy_path, use_cls=True, device='cpu'):
#     tokenizer = AutoTokenizer.from_pretrained(model)
#     model = AutoModel.from_pretrained(model).to(device)
#     model.eval()

#     with open(input_path, 'r', encoding='utf-8') as f:
#         data = json.load(f)

#     keys = list(data.keys())
#     k = 0
#     emb_list = []

#     with torch.no_grad():
#         for key in keys:
#             sentences = data[key]
#             if k == 0:
#                 k = len(sentences)
#             emb_for_key = []
#             for sent in sentences:
#                 if not sent.strip():
#                     emb = np.zeros(768)  # XLM-R hidden size is 768
#                 else:
#                     encoded_input = tokenizer(sent, return_tensors='pt', truncation=True, max_length=512).to(device)
#                     model_output = model(**encoded_input)
#                     if use_cls:
#                         emb_tensor = model_output.last_hidden_state[:, 0, :]  # CLS token
#                     else:
#                         emb_tensor = mean_pooling(model_output, encoded_input['attention_mask'])
#                     emb = emb_tensor[0].cpu().numpy()
#                 emb_for_key.append(emb)
#             emb_list.append(emb_for_key)

#     emb_array = np.array(emb_list)  # shape (key_num, k, emb_dim)
#     np.save(output_npy_path, emb_array)
#     print(f"编码完成，保存为.npy文件，形状: {emb_array.shape}")


# ==================== 主流程 ====================

if __name__ == "__main__":

    dir_path = 'feature'

    # Step 1: Chunk
    input_json_path = "ELAM_content.json"
    chunked_output_path = os.path.join(dir_path, "ELAM_chunk.json")

    min_chunks = chunk_json_file_and_get_min_len(input_json_path, chunked_output_path)
    print(f"所有 key 中 chunk 列表最少元素数量是: {min_chunks}")

    # Step 2: Score
    alpha = 1
    beta = 1
    add_keyword_summary = True  # 是否添加关键词句子
    deduplicate_keywords = True  # 是否去重关键词
    important_words_path = "../LAW/important_words_zh.json"
    scored_output_path = os.path.join(dir_path, "ELAM_sentence_scored.json")
    score_sentences(chunked_output_path, important_words_path, scored_output_path, alpha, beta, add_keyword_summary, deduplicate_keywords)

    # Step 3: Extract TopK
    k = 3  # 提取前k个句子
    if add_keyword_summary:
        k = k + 1
    topk_output_path = os.path.join(dir_path, "ELAM_topk.json")
    extract_first_k_values(scored_output_path, topk_output_path, k)
