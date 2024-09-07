# 1LLM + 1GPU + 1Day: LLM Efficiency Challenge

### Participation Requirements

* Start with an approved base model
* Use only open-source data
* Fine-tune for a single 24-hour period
* Use a single GPU: NVIDIA 4090 or NVIDIA A100 (40GB)
* Two hardware tracks: NVIDIA 4090 and NVIDIA A100, evaluated separately

### Approved Base Models

* Open base models without instruction-tuning
* Accepted licenses: MIT, Apache 2, BigScience RAIL, LLAMA 2 community license (with approval)
* Examples:
* ALBERT
* BART
* BERT
* Bloom
* LLaMA
* Mistral
* T5

### Datasets

* Any open-sourced dataset is allowed
* Examples:
    * Databricks-Dolly-15
    * OpenAssistant Conversations Dataset
    * The Flan Collection
    * Self-created datasets 
    * Generated datasets


### Evaluation Process

* Subset of HELM benchmark
* Secret holdout tasks (logic reasoning Q&A and conversational chat)


### Scoring

* Ranking based on geometric mean across all tasks

### Hardware Constraints

* 1 A100 80GB GPU
* 80GB of RAM
* 500GB of Disk

### Time Constraints
* 24 Hour Time Limit