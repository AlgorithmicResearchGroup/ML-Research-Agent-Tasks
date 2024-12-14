import os
import argparse
from mlrb_agent_tasks.run import TaskFamily

### Example usage
# python test/test.py --run_id 12345 --task_name baby_lm
###

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", type=int, default=12345)
    parser.add_argument("--task_name", type=str, default="baby_lm")
    args = parser.parse_args()

    task_family = TaskFamily()

    tasks = task_family.get_tasks()

    task = tasks[args.task_name]

    print(task_family.get_instructions(task))

    run_id = args.run_id
    task_family.setup_task_directory("working_dir", run_id)
