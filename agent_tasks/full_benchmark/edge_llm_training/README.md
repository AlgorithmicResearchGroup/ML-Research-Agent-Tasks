# Edge LLMs Challenge: Training from Scratch Track

This competition requires agents to train language models from scratch without using pre-trained LLMs. The final models must run on a smartphone device with 12GB memory. Only C4 and Alpaca datasets are allowed for training and fine-tuning. Evaluation metrics and processes are similar to the Compression Track, using the OpenCompass benchmark and considering performance, memory usage, and throughput.


## Edge LLMs Challenge: Training from Scratch Track: https://edge-llms-challenge.github.io/edge-llm-challenge.github.io/index

### Participation Requirements

* Goal: Train language models from scratch without using pre-trained LLMs
* No constraints on model architectures or training procedures
* Final models must run on a smartphone device with 12GB memory
* Models must be submitted in FP16 or FP32 format (no quantization allowed)

### Dataset Restrictions

* Only C4 and Alpaca datasets allowed for training and fine-tuning

### Evaluation Process

1. Models evaluated on a subset of the OpenCompass benchmark
2. Secret holdout tasks included to avoid overfitting
3. Performance ranked for each task, top 10 submissions scored from 10 to 1
4. Throughput measured on a smartphone platform (doubled score importance)
5. Memory usage must not exceed 12GB during inference
6. Final rank determined by sum of scores across all tasks

### Submission Requirements

* GitHub repository with code, model definitions, configurations, and locally evaluated results
* Google Drive links to model checkpoints and compiled model for Android app
* Repository must be private with edge-llm-challenge as a collaborator

### Evaluation Metrics

* Performance score on evaluation tasks
* Memory requirement (must be <12GB)
* Throughput (tokens per second)
* Parameter count (for information only)

### Datasets

* Training datasets: C4 and Alpaca only
* Evaluation datasets include CommonsenseQA, BIG-Bench Hard, GSM8K, LongBench, HumanEval, TruthfulQA, CHID, and secret holdout tasks



