# ML Research Benchmark Tasks

This repository contains the tasks for ML Research Benchmark, a benchmarkdesigned to evaluate the capabilities of AI agents in accelerating ML research and development. The benchmark consists of 9 competition-level tasks that span the spectrum of activities typically undertaken by ML researchers.


## Introduction

The MLRB aims to measure the acceleration of AI agents in ML research and development. It focuses on competition-level tasks that reflect the current frontiers of ML research, providing a more nuanced and challenging evaluation environment than existing benchmarks.

[![arXiv](https://img.shields.io/badge/arXiv-2410.22553-b31b1b.svg)](https://arxiv.org/abs/2410.22553)

- [:paperclip: ML Research Benchmark Paper](https://arxiv.org/abs/2410.22553) 
- [:robot: ML Research Agent](https://github.com/AlgorithmicResearchGroup/ML-Research-Agent)
- [:white_check_mark: ML Research Tasks](https://github.com/AlgorithmicResearchGroup/ML-Research-Agent-Tasks)
- [:chart_with_upwards_trend: ML Research Evaluation](https://github.com/AlgorithmicResearchGroup/ML-Research-Agent-Evals)

## Installation

```bash
pip install mlrb-agent-tasks
```

# Usage

The library exposes a single function, get_task

get_task:
- path: path to copy the task to
- benchmark: name of the benchmark
- task: name of the task

This function will copy the task to the specified path and return a dictionary with the task name and prompt.

```
{
    "name": str, - name of the task
    "prompt": str, - prompt for the task
}
```

## Example Usage

```python
from mlrb_agent_tasks import get_task

# Example usage
result = get_task("./", "full_benchmark", "llm_efficiency")
print(result['prompt'])
```


## Contributing

We welcome contributions to the ML Research Benchmark! Please read our [CONTRIBUTING.md](CONTRIBUTING.md) file for guidelines on how to submit issues, feature requests, and pull requests.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For questions or feedback, please open an issue in this repository or contact [matt@algorithmicresearchgroup.com](mailto:matt@algorithmicresearchgroup.com).