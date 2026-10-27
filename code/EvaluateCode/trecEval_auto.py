# Default configuration
DEFAULT_CONFIG = {
    # !This is just an example and needs to be modified to the actual path!
    "result_file": "codaAndData/Datasets/ELAM/test/test_candidate.json",           # Path to result.trec file
    "relevance_file": "codaAndData/Datasets/ELAM/test/test_candidate.json",     # Path to relevance.trec file
    "output_dir": "./outputDir",                 # Output directory
    "model_name": "BM25",                # Model name
    "dataset_name": "ELAM"         # Dataset name
}

# The evaluation results will be output as a JSON file in the following format:
# res{
#     model:<model name used>                           # Model name
#     dataset:<dataset name used>                       # Dataset name
#     R@k :[<value for k=1>,<k=5>,<k=10>,<k=20>,<k=30>]    # Recall@k metrics array
#     NDCG@k :[<k=1>,<k=5>,<k=10>,<k=20>,<k=30>]  # Normalized Discounted Cumulative Gain@k array
#     p@k  :[<k=1>,<k=5>,<k=10>,<k=20>,<k=30>]    # Precision@k metrics array
#     ACC   :<corresponding value>                              # Accuracy: proportion of top-1 matches
#     MRR  :<corresponding value>                               # Mean Reciprocal Rank
#     MAP  :<corresponding value>                               # Mean Average Precision
#     mi-F1:<corresponding value>                               # Micro-averaged F1 score
#     ma-F1:<corresponding value>                               # Macro-averaged F1 score
# }
#
# Usage example:
# python trecEval_auto_en.py \
#   --result_file ./result.trec \
#   --relevance_file ./relevance.trec \
#   --output_dir ./output \
#   --model_name "BM25 \
#   --dataset_name "COLIEE2025"

# <START DEL>
import argparse
import json
import os
import sys
import numpy as np
from collections import defaultdict

def read_trec_result(file_path):
    results = defaultdict(list)
    with open(file_path, 'r') as f:
        for line in f:
            query_id, _, doc_id, rank, score, _ = line.strip().split()
            results[query_id].append((doc_id, int(rank), float(score)))
    for query_id in results:
        results[query_id].sort(key=lambda x: x[1])
    return results

def read_trec_relevance(file_path):
    relevance = defaultdict(list)
    with open(file_path, 'r') as f:
        for line in f:
            query_id, _, doc_id, rel_rank = line.strip().split()
            relevance[query_id].append((doc_id, int(rel_rank)))
    return relevance

def compute_recall_at_k(predicted, relevant, k):
    if not relevant:
        return 0.0
    top_k_docs = set(doc_id for doc_id, _, _ in predicted[:k])
    relevant_docs = set(doc_id for doc_id, _ in relevant)
    retrieved_relevant = len(top_k_docs & relevant_docs)
    return retrieved_relevant / len(relevant_docs)

def compute_precision_at_k(predicted, relevant, k):
    if k == 0:
        return 0.0
    top_k_docs = set(doc_id for doc_id, _, _ in predicted[:k])
    relevant_docs = set(doc_id for doc_id, _ in relevant)
    retrieved_relevant = len(top_k_docs & relevant_docs)
    return retrieved_relevant / k

def compute_ndcg_at_k(predicted, relevant, k):
    if not relevant:
        return 0.0
    rel_dict = {doc_id: rel_rank for doc_id, rel_rank in relevant}
    dcg = 0.0
    for i, (doc_id, _, _) in enumerate(predicted[:k], 1):
        if doc_id in rel_dict:
            gain = 1.0 / rel_dict[doc_id] if rel_dict[doc_id] > 0 else 0.0
            dcg += gain / np.log2(i + 1)
    sorted_rel = sorted([1.0 / r if r > 0 else 0.0 for _, r in relevant], reverse=True)
    idcg = sum(g / np.log2(i + 2) for i, g in enumerate(sorted_rel[:k]))
    return dcg / idcg if idcg > 0 else 0.0

def save_results(results, output_dir, model_name, dataset_name):
    output = {
        "model": model_name,
        "dataset": dataset_name,
        "R@k": results["R@k"],
        "NDCG@k": results["NDCG@k"],
        "p@k": results["p@k"],
        "ACC": results["ACC"],
        "MRR": results["MRR"],
        "MAP": results["MAP"],
        "mi-F1": results["mi-F1"],
        "ma-F1": results["ma-F1"]
    }
    os.makedirs(output_dir, exist_ok=True)
    output_filename = f"{dataset_name}_{model_name}_metric.json"
    output_path = os.path.join(output_dir, output_filename)
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"Results saved to: {output_path}")

def evaluate(result_file, relevance_file, output_dir, model_name, dataset_name):
    print("="*60)
    print("Starting retrieval evaluation...")
    print(f"Result file: {result_file}")
    print(f"Ground truth file: {relevance_file}")
    print(f"Output directory: {output_dir}")
    print(f"Model name: {model_name}")
    print(f"Dataset name: {dataset_name}")
    print("="*60)
    
    print("\nCalculating metrics...")
    
    results = read_trec_result(result_file)
    relevance = read_trec_relevance(relevance_file)
    
    ks = [1, 5, 10, 20, 30]
    metrics = {
        "R@k": [],
        "NDCG@k": [],
        "p@k": [],
        "ACC": 0.0,
        "MRR": 0.0,
        "MAP": 0.0,
        "mi-F1": 0.0,
        "ma-F1": 0.0
    }
    
    total_queries = 0
    acc_count = 0
    reciprocal_ranks = []
    avg_precisions = []
    micro_tp, micro_fp, micro_fn = 0, 0, 0
    macro_f1_scores = []
    
    for query_id in results:
        if query_id not in relevance:
            continue
        total_queries += 1
        pred = results[query_id]
        rel = relevance[query_id]
        
        for k in ks:
            metrics["R@k"].append(compute_recall_at_k(pred, rel, k))
            metrics["p@k"].append(compute_precision_at_k(pred, rel, k))
            metrics["NDCG@k"].append(compute_ndcg_at_k(pred, rel, k))
        
        pred_top = pred[0][0] if pred else None
        rel_top = min(rel, key=lambda x: x[1])[0] if rel else None
        if pred_top == rel_top:
            acc_count += 1
        
        rel_docs = set(doc_id for doc_id, _ in rel)
        for i, (doc_id, _, _) in enumerate(pred, 1):
            if doc_id in rel_docs:
                reciprocal_ranks.append(1.0 / i)
                break
        else:
            reciprocal_ranks.append(0.0)
        
        retrieved = [doc_id for doc_id, _, _ in pred]
        relevant = set(doc_id for doc_id, _ in rel)
        precisions = []
        rel_count = 0
        for i, doc_id in enumerate(retrieved, 1):
            if doc_id in relevant:
                rel_count += 1
                precisions.append(rel_count / i)
        avg_precisions.append(sum(precisions) / len(relevant) if precisions else 0.0)
        
        pred_set = set(doc_id for doc_id, _, _ in pred[:max(ks)])
        rel_set = set(doc_id for doc_id, _ in rel)
        tp = len(pred_set & rel_set)
        fp = len(pred_set - rel_set)
        fn = len(rel_set - pred_set)
        micro_tp += tp
        micro_fp += fp
        micro_fn += fn
        if tp > 0:
            prec = tp / (tp + fp)
            rec = tp / (tp + fn)
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
            macro_f1_scores.append(f1)
    
    if total_queries == 0:
        print("Warning: No valid queries found! All metrics will be set to 0.")
        for metric in ["R@k", "p@k", "NDCG@k"]:
            metrics[metric] = [0.0] * len(ks)
    else:
        metrics["ACC"] = acc_count / total_queries
        metrics["MRR"] = sum(reciprocal_ranks) / total_queries
        metrics["MAP"] = sum(avg_precisions) / total_queries
        if micro_tp > 0:
            micro_prec = micro_tp / (micro_tp + micro_fp)
            micro_rec = micro_tp / (micro_tp + micro_fn)
            metrics["mi-F1"] = 2 * micro_prec * micro_rec / (micro_prec + micro_rec) if (micro_prec + micro_rec) > 0 else 0.0
        metrics["ma-F1"] = sum(macro_f1_scores) / len(macro_f1_scores) if macro_f1_scores else 0.0
        
        for metric in ["R@k", "p@k", "NDCG@k"]:
            values = metrics[metric]
            metrics[metric] = [sum(values[i::len(ks)]) / total_queries for i in range(len(ks))]
    
    save_results(metrics, output_dir, model_name, dataset_name)
    
    print("\nEvaluation complete")
    print("="*60)
    
    return metrics

def main():
    parser = argparse.ArgumentParser(description="Evaluate TREC retrieval results")
    parser.add_argument(
        "--result_file",
        default=DEFAULT_CONFIG["result_file"],
        help=f"Path to result.trec file (default: {DEFAULT_CONFIG['result_file']})"
    )
    parser.add_argument(
        "--relevance_file",
        default=DEFAULT_CONFIG["relevance_file"],
        help=f"Path to relevance.trec file (default: {DEFAULT_CONFIG['relevance_file']})"
    )
    parser.add_argument(
        "--output_dir",
        default=DEFAULT_CONFIG["output_dir"],
        help=f"Output directory for result files (default: {DEFAULT_CONFIG['output_dir']})"
    )
    parser.add_argument(
        "--model_name",
        default=DEFAULT_CONFIG["model_name"],
        help=f"Model name (default: {DEFAULT_CONFIG['model_name']})"
    )
    parser.add_argument(
        "--dataset_name",
        default=DEFAULT_CONFIG["dataset_name"],
        help=f"Dataset name (default: {DEFAULT_CONFIG['dataset_name']})"
    )
    args = parser.parse_args()
    
    print("Current configuration:")
    print(f"  Result file: {args.result_file}")
    print(f"  Ground truth file: {args.relevance_file}")
    print(f"  Output directory: {args.output_dir}")
    print(f"  Model name: {args.model_name}")
    print(f"  Dataset name: {args.dataset_name}")
    print()
    
    evaluate(args.result_file, args.relevance_file, args.output_dir, args.model_name, args.dataset_name)

if __name__ == "__main__":
    main()