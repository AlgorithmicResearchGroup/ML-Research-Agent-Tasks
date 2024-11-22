# MiniPile Challenge

The MiniPile Challenge focuses on data-efficient language model pre-training using a 6GB subset of The Pile corpus. Participants must pre-train language models on this dataset, which contains at most 1M documents. The evaluation involves tasks from GLUE and SNI benchmarks. Baseline models include BERT-Base and T5v1.1-Base. The challenge aims to achieve comparable downstream performance to models trained on much larger datasets.

### Participation Requirements

* Use the MiniPile dataset for pre-training language models
* MiniPile is a 6GB subset of the deduplicated 825GB The Pile corpus
* Contains at most 1M documents
* Goal is to facilitate research on data-efficient language model pre-training

### Datasets

* MiniPile: 6GB uncompressed, 3GB compressed
* 1M/500/10k training/validation/test examples
* Vocab size: 32,309,614
* Median document length: 294
* Longest document length: 929,633
* Curated using a three-step data filtering process:

1. Infer embeddings for all Pile documents
2. Cluster embedding space using k-means
3. Filter out low-quality clusters

### Evaluation Process

1. Evaluation tasks:

* GLUE (General Language Understanding Evaluation)
* SNI (SuperNatural-Instructions)

1. Baseline models:

* BERT-Base (MLM objective)
* T5v1.1-Base (span-corrupting MLM objective)

### Scoring:

* Compare performance to models pre-trained on much larger datasets
* Goal is to achieve reasonable downstream performance with only small drops

### Hardware Constraints

* Experiments run on a single NVIDIA RTX 3090 GPU
* BERT-Base pre-training: 54 hours
* T5v1.1-Base pre-training: 21 hours
* Fine-tuning times: 2-3 hours

* 1 A100 80GB GPU
* 80GB of RAM
* 500GB of Disk

### Time Constraints
* 24 Hour Time Limit