For the ELAM dataset, we provide both the dataset files and the corresponding split files.

For the COLIEE2025 and LeCaRDv2 datasets, due to licensing restrictions (as of the time of our publication, COLIEE2025 requires an application for access) and storage limitations, we do not include the full versions. Please download them from their official websites.

We provide the JSON files for the test set splits, and the remaining data should be considered as the training set.


The Datasets directory contains three legal document retrieval datasets, each adopting a standard train/test split structure. The ELAM dataset provides a complete set of files, including candidate document pools (candidate.json) and related document relationships (relative.json) for establishing positive and negative sample pairs; the COLIEE dataset contains unique labels.json tag files for detailed label annotation; the LeCaRDv2 dataset includes query.json query content files (large files) and candidate.json candidate document ID lists using numeric ID format. All datasets share gt.trec ground truth files (for evaluation metric calculation) and ids.json query ID files, but have different additional file types based on specific task requirements, reflecting differences in data format, ID encoding methods, and task characteristics across different datasets.

Directory Tree Structure：
```
Datasets/
├── README.md
├── ELAM/
│   ├── readme.md
│   ├── train/
│   │   ├── train_gt.trec
│   │   ├── train_ids.json
│   │   ├── train_candidate.json
│   │   └── train_relative.json
│   └── test/
│       ├── test_gt.trec
│       ├── test_ids.json
│       ├── test_candidate.json
│       └── test_relative.json
├── COLIEE/
│   ├── train/
│   │   ├── train_gt.trec
│   │   ├── train_ids.json
│   │   └── train_labels.json
│   └── test/
│       ├── test_gt.trec
│       ├── test_ids.json
│       └── test_labels.json
└── LeCaRDv2/
    ├── train/
    │   ├── train_gt.trec
    │   ├── train_ids.json
    │   ├── train_query.json
    │   └── train_candidate.json
    └── test/
        ├── test_gt.trec
        ├── test_ids.json
        ├── test_query.json
        └── test_candidate.json
```