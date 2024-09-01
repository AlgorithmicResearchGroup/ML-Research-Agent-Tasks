1LLM + 1GPU + 1Day: LLM Efficiency Challenge: Detailed Rules and Guidelines: https://llm-efficiency-challenge.github.io/challenge

Participation Requirements

* Start with an approved base model
* Use only open-source data
* Fine-tune for a single 24-hour period
* Use a single GPU: NVIDIA 4090 or NVIDIA A100 (40GB)
* Two hardware tracks: NVIDIA 4090 and NVIDIA A100, evaluated separately

Approved Base Models

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

Datasets

* Any open-sourced dataset is allowed
* Examples:
    * Databricks-Dolly-15
    * OpenAssistant Conversations Dataset
    * The Flan Collection
    * Self-created datasets 
    * Generated datasets

Evaluation Process

1. First stage:

* Subset of HELM benchmark
* Secret holdout tasks (logic reasoning Q&A and conversational chat)
* Ranking based on geometric mean across all tasks

1. Second stage (after October 25th, 2023):

* Top 3 teams in each hardware category submit code and data for reproduction
* Organizers replicate the entire process within 24 hours using a single GPU
* Winners selected based on reproducibility and performance

Hardware Constraints

* 128GB of RAM
* 500GB of Disk
* Maximum 2 hours runtime on sample configuration files


# Directions: 

1. **Choose approved models and datasets**
   - Select from the list on the official challenge website

2. **Start with a sample submission**
   - Use one of the provided samples in the repository

3. **Test locally**
   - Ensure your submission runs on the GPU you have access to

4. **Create your `Dockerfile`**
   - Include all necessary code and dependencies
   - Expose an HTTP server with `/process` and `/tokenize` endpoints
   - Configure to download model weights during build or runtime
   - Do not include weights directly in the repository

5. **Evaluate using HELM**
   - Follow instructions in `helm.md` to test locally

9. **Follow the OpenAPI specification**
    - Use the `openapi.json` file for API endpoint implementation
