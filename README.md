# AI Competition Benchmark

This repository contains the tasks for AI Competition Benchmark, a benchmarkdesigned to evaluate the capabilities of AI agents in accelerating AI research and development. The benchmark consists of 9 competition-level tasks that span the spectrum of activities typically undertaken by AI researchers.

## Introduction

The AI Competition Benchmark aims to measure the acceleration of AI agents in AI research and development. It focuses on competition-level tasks that reflect the current frontiers of machine learning research, providing a more nuanced and challenging evaluation environment than existing benchmarks.

## Installation

```bash
pip install agent-tasks
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
from agent_tasks.run import get_task

# Example usage
result = get_task("./", "mini_benchmark", "mini_baby_lm")
print(result['prompt'])
```


## Contributing

We welcome contributions to the AI Competition Benchmark! Please read our [CONTRIBUTING.md](CONTRIBUTING.md) file for guidelines on how to submit issues, feature requests, and pull requests.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For questions or feedback, please open an issue in this repository or contact [matt@algorithmicresearchgroup.com](mailto:matt@algorithmicresearchgroup.com).