# Edge LLMs Challenge: Compression Track

In this competition, agents are tasked with developing compression methods for pre-trained LLMs (Phi-2, Llama-3-8B, and Qwen-7B) to run on smartphone devices with 12 GB DRAM. Models must be submitted in FP16 or FP32 format without quantization. Evaluation uses a subset of the OpenCompass benchmark, including secret holdout tasks. The challenge considers performance, memory usage, and throughput, with final rankings based on scores across all tasks.

### Participation Requirements

* Goal: Develop compression methods for pre-trained LLMs to run on smartphone devices
* Target models: Phi-2, Llama-3-8B, and Qwen-7B
* Compressed models must run on a smartphone device with 12 GB DRAM
* Models must be submitted in FP16 or FP32 format (no quantization allowed)


### Datasets

* Evaluation datasets include CommonsenseQA, BIG-Bench Hard, GSM8K, LongBench, HumanEval, TruthfulQA, CHID, and secret holdout tasks

### Evaluation Process

1. Models evaluated on a subset of the OpenCompass benchmark
2. Secret holdout tasks included to avoid overfitting
3. Performance ranked for each task, top 10 submissions scored from 10 to 1
4. Throughput measured on a smartphone platform (doubled score importance)
5. Memory usage must not exceed 12GB during inference
6. Final rank determined by sum of scores across all tasks

### Scoring

* Performance score on evaluation tasks
* Memory requirement (must be <12GB)
* Throughput (tokens per second)
* Parameter count (for information only)


### Hardware Constraints

* 1 A100 80GB GPU
* 80GB of RAM
* 500GB of Disk

### Time Constraints
* 24 Hour Time Limit


