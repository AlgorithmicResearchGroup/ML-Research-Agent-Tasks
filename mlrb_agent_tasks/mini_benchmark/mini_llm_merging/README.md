> # LLM-Merging Competition

This competition challenges participants to create a generalist model by merging expert models to perform well on various skills. Participants must use publicly available models up to 8GB in size with research-compatible licenses. The evaluation process measures performance on hidden test datasets and considers time and space efficiency. The merging/fine-tuning and evaluation must take less than 1 hour on a single Nvidia A6000 (48 GB) or equivalent GPU.
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