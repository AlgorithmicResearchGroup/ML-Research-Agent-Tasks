## Overview

Agent-Task is a library that provides a set of benchmark tasks for the agent to perform. 

## Installation

```bash
pip install -e . 
```

## Usage

```python
from agent_tasks.run import get_task

# Example usage
result = get_task("./", "mini_benchmark", "mini_baby_lm")
print(result['prompt'])
```

get_task:
- path: path to copy the task to
- benchmark: name of the benchmark
- task: name of the task


