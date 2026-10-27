import json
import os
import math
import numpy as np
import torch
import re
from collections import defaultdict
from transformers import AutoTokenizer, AutoModel
from nltk.tokenize import word_tokenize  # 使用nltk进行英文分词

# ==================== 分段模块 ====================
def chunk_text(text, window_size=500, overlap=25):

    def find_closest_space(text, window_size):
        closest_space = -1
        min_diff = float('inf')  # 初始化为无穷大，用于存储最小差值

        # 遍历文本，查找所有小于 window_size 且为空格的字符
        for i in range(window_size):
            if text[i] == ' ':  # 如果遇到空格
                diff = abs(i - window_size)  # 计算当前空格位置与 window_size 的差距
                if diff < min_diff:
                    min_diff = diff
                    closest_space = i  # 更新最接近的空格位置

        return closest_space
    
    chunks = []
    length = len(text)

    # 正则表达式：匹配形如 [数字] 的分隔符
    pattern = r'\[\d+\]'

    # 查找所有 [数字] 位置
    markers = [match.start() for match in re.finditer(pattern, text)]

    # 如果没有找到任何分隔符，直接返回整个文本
    if not markers:
        raise ValueError("没有分隔符")

    current_chunk = ""

    # 循环遍历每个 [数字] 标记
    for i, marker in enumerate(markers):
        # 获取当前分隔符的结束位置，查找该标记对应的结束位置 ']'
        end = text.find(']', marker) + 1
        # segment = text[marker:end]  # 当前分隔符 [数字] 及其后内容
        next_marker = markers[i + 1] if i + 1 < len(markers) else length  # 下一个分隔符的位置
        
        # 获取当前分隔符和下一个分隔符之间的文本
        segment_with_next = text[marker:end] + text[end:next_marker].strip()

        # 如果合并后的长度不超过 window_size，合并成一个句子
        if len(current_chunk + " " + segment_with_next) <= window_size:
            current_chunk += " " + segment_with_next
        else:
            if current_chunk:  # 如果已经有内容，先把当前片段加入结果
                chunks.append(current_chunk.strip())
            current_chunk = segment_with_next  # 否则从当前分隔符开始新的片段
            while len(current_chunk) > window_size:
                idx = find_closest_space(current_chunk, window_size)
                temp = current_chunk[:idx]
                chunks.append(temp)
                current_chunk = current_chunk[idx:]


    # 添加最后剩余的内容
    if current_chunk:
        chunks.append(current_chunk.strip())

    # 删除每个分隔符 [数字]（比如 [1]， [2]，等等）
    chunks = [re.sub(r'\[\d+\]', '', chunk).strip() for chunk in chunks]

    return chunks

def chunk_json_file_and_get_min_len(input_path, output_path, window_size=800, overlap=40):
    with open(input_path, 'r', encoding='utf-8') as f:
        json_dict = json.load(f)

    chunked_dict = {}
    min_len = None
    idx = -1

    for key, text in json_dict.items():
        chunks = chunk_text(text, window_size, overlap)
        chunked_dict[key] = chunks
        if min_len is None or len(chunks) < min_len:
            min_len = len(chunks)
            idx = key

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(chunked_dict, f, ensure_ascii=False, indent=2)
    print(f"key {idx} has min_len")
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
    print(f"Loaded {len(words)} important words, first 10 examples: {list(words)[:10]}")
    return words

def compute_df(sentence_dict):
    df = defaultdict(int)
    total_docs = 0
    for sentences in sentence_dict.values():
        for sent in sentences:
            total_docs += 1
            tokens = set(word_tokenize(sent))  # 使用 nltk.word_tokenize 进行英文分词
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
        print(key)
        matched_keywords = []
        scored_sentences = []

        for sent in sentences:
            tokens = word_tokenize(sent)  # 使用nltk进行英文分词

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
    print(f"Scoring completed, {'with keyword sentence added, ' if add_keyword_summary else ''}results saved to: {output_path}")

# ==================== TopK提取模块 ====================

def extract_first_k_values(input_path, output_path, k=3):
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    result = {}
    for key, dict_list in data.items():
        print(key)
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

    print(f"Extracted top {k} items (empty for less than k), saved to: {output_path}")

# ==================== 主流程 ====================

if __name__ == "__main__":

    dir_path = 'feature'

    # Step 1: Chunk
    input_json_path = "COLIEE_content.json"
    chunked_output_path = os.path.join(dir_path, "COLIEE_chunk.json")

    min_chunks = chunk_json_file_and_get_min_len(input_json_path, chunked_output_path)
    print(f"Minimum chunk count across all keys: {min_chunks}")

    # Step 2: Score
    alpha = 1
    beta = 1
    add_keyword_summary = True  # Whether to add keyword sentence
    deduplicate_keywords = True  # Whether to deduplicate keywords
    important_words_path = "../LAW/important_words_en.json"
    scored_output_path = os.path.join(dir_path, "COLIEE_sentence_scored.json")
    score_sentences(chunked_output_path, important_words_path, scored_output_path, alpha, beta, add_keyword_summary, deduplicate_keywords)

    # Step 3: Extract TopK
    k = 9  # Extract top-k sentences
    if add_keyword_summary:
        k = k + 1
    topk_output_path = os.path.join(dir_path, "COLIEE_topk.json")
    extract_first_k_values(scored_output_path, topk_output_path, k)
