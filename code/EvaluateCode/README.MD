Run ``trecEval_auto.py`` to complete the evaluation and generate the metrics file.

The TREC format (Text REtrieval Conference format) is a standard data format in the field of information retrieval, originally designed by the National Institute of Standards and Technology (NIST) for the TREC evaluation. It is widely used for evaluating and training information retrieval systems, especially in tasks such as question answering, retrieval models, and legal text retrieval. In our evaluation, we follow this convention by adopting the standard TREC format.(.trec文件)

The required parameters are as follows:
```
1. Result file in TREC format  
2. Ground truth (label) file, also in TREC format  
3. A output dir
4.(Optional) Model name or dataset name

```

Example：
```
python trecEval_auto.py \
   --result_file ./result.trec \
   --relevance_file ./relevance.trec \
   --output_dir ./output \
   --model_name "BM25" \
   --dataset_name "COLIEE2025"
```