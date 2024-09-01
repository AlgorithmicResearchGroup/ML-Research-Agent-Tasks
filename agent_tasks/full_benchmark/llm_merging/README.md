> # LLM-Merging Competition

This competition challenges participants to create a generalist model by merging expert models to perform well on various skills. Participants must use publicly available models up to 8GB in size with research-compatible licenses. The evaluation process measures performance on hidden test datasets and considers time and space efficiency. The merging/fine-tuning and evaluation must take less than 1 hour on a single Nvidia A6000 (48 GB) or equivalent GPU.


LLM-Merging Competition: Detailed Rules and Guidelines:  https://llm-merging.github.io/index

Participation Requirements

* Goal: Create a generalist model by merging expert models to perform well on various skills
* Use publicly available models up to 8GB in size
* Models must have licenses compatible with research use (e.g., MIT, Apache 2)
* Submissions must be reproducible from initial model through merging and fine-tuning
* No use of copyrighted, proprietary data, code, or closed-source content

Allowed Models

* Any publicly available model weights that can be downloaded and meet conditions:
* Available on Hugging Face
* Uploaded before May 31st, 2024
* Parameter size not larger than 8 billion
* Recommended models include:
* Llama 2 Family (7B versions)
* Llama 3 Family (8B versions)
* Mistral Family (7B versions)
* FLAN T5 Family
* Gemma Family (7B versions)
* Various fine-tuned models and adapters are also allowed

Datasets

* Validation datasets provided (e.g., CosmosQA, XSum) for time and space efficiency measurement
* Hidden test datasets for performance evaluation:

1. Leaderboard ranking test tasks
2. Final ranking test tasks

* No new datasets will be collected or released for the competition

Evaluation Process

* Submissions evaluated once a week
* Performance measured on hidden test datasets
* Time and space efficiency measured using validation datasets

Hardware Constraints

* Merging/fine-tuning and evaluation must take less than 1 hour on a single Nvidia A6000 (48 GB) or equivalent

Additional Notes

* A starter kit with an end-to-end submission flow is available on GitHub
* The competition focuses on re-using provided models to create a generalist model
* Participants are encouraged to be creative and innovative in model merging techniques




## Important Tips
1.  Please do not specify any device_id in the code because the device_id might not hold in our setup. If you need to specify a device_id in your setup, one solution is to use environment variables like
```bash
export CUDA_VISIBLE_DEVICES=0  
```
2. Please do not specify any filepaths because they may not be the same in our setup. If you need to specify the HuggingFace cache, one solution is to use environment variables like
```bash
export HUGGINGFACE_HUB_CACHE=/tmp/
```
and then access this path in Python via 
```python
path=os.environ["HUGGINGFACE_HUB_CACHE"]
```
3. When running `tar` on this repo `LLM-Merging` to submit it, please ensure this directory is called `LLM-Merging` and not renamed to any directories. This can cause issues when evaluating your submissions.   

## Setup Environment

The library was tested on CUDA 10.1 on an A6000.

```bash
conda env create -f environment.yml --name llm-merging
conda activate llm-merging
export PYTHONPATH=`pwd`
```

Authentication tokens are required for certain models like Llama2, which require users to agree to specific terms. You can find the authentication token [here](https://huggingface.co/settings/tokens).

```bash
export HF_AUTH_TOKEN=""
```

## Developing New Merging Methods

Do not modify any files other than the new file you create and `setup.py`. Doing so can result in the grounds for invalidating your submission. If you need to change code in other files, feel free to open a pull request.

1. To add a new merging method, create a new file in `llm_merging/merging`.

    This file should implement `__init__.py` and `merge.py` functions and extend `llm_merging/merging/Merges`.
    See `llm_merging/merging/FlanT5Avg.py` or `llm_merging/merging/LlamaAvg.py` for examples.

2. Modify `setup.py` and add an entry with the merging method in `llm_merging.merging.Merges`.

    For example, the entry `llama_avg = llm_merging.merging.LlamaAvg:LlamaAvg` indicates the method is called `llama_avg` and the file is at `llm_merging/merging/LlamaAvg`.

    Any additional required libraries can be specified in `setup.py`.

## Test Method

```bash
python llm_merging/setup.py install
python llm_merging/main.py -m {merging_method}
```

The datasets (CosmosQA and XSum) are mainly included to ensure the merging method (with evaluation on those datasets) runs in under the 1-hour time limit. Our results on `llama_avg` are `{"cosmos_qa": {"accuracy": 0.234}, "xsum": {"rouge1": 0.123, "rouge2": 0.023, "rougeL": 0.093, "rougeLsum": 0.102}}`, which run in about 25 minutes on our A6000.

## Submissions

After modifying the file, tar the file into a tarball using the command:

```bash
tar -cvf {merging_method}.tar LLM-Merging
```

Submit the tar file using this [form](https://docs.google.com/forms/d/17TPg7N02o8qvw1czx55Zbh_5Kp7-YStUIOhQDJYc23g/)

## Leaderboard

The leaderboard of the submitted solutions can be found [here](https://huggingface.co/spaces/margsli/merging_competition). Please note that your submission might not appear on the leaderboard immediately, as it is updated every few days. If you encounter any issues, please contact us.

Note: This submission method is only temporary and another automatic submission method should be comming soon.