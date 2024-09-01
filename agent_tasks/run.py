import os
import shutil
from pathlib import Path
from typing import Dict, Any
import subprocess
from jinja2 import Environment, FileSystemLoader, TemplateNotFound
from agent_tasks.prompts import retreive_tasks

def get_task(directory: str, benchmark: str, task_name: str) -> Dict[str, Any]:
    """
    Copy files from a task in a benchmark to a specified directory, execute it, and return the result.

    Args:
    directory (str): Target directory path
    benchmark (str): Benchmark type ('mini_benchmark' or 'full_benchmark')
    task_name (str): Task name

    Returns:
    Dict[str, Any]: A dictionary containing the task result and any additional information
    """
    # Get the current directory (where run.py is located)
    current_dir = Path(__file__).parent

    # Construct paths
    benchmark_dir = current_dir / benchmark
    task_dir = benchmark_dir / task_name
    target_dir = Path(directory)

    # Check if the benchmark directory exists
    if not benchmark_dir.exists():
        return {"error": f"Benchmark directory '{benchmark}' does not exist."}

    # Check if the task directory exists
    if not task_dir.exists():
        return {"error": f"Task '{task_name}' does not exist in the {benchmark}."}

    # Create the target directory if it doesn't exist
    target_dir.mkdir(parents=True, exist_ok=True)

    # Copy only the files within the task directory to the target directory
    try:
        for item in task_dir.iterdir():
            if item.is_file():
                shutil.copy2(str(item), str(target_dir))
        print(f"Successfully copied files from '{task_name}' in {benchmark} to {directory}")
    except Exception as e:
        return {"error": f"Error copying task files: {e}"}

    # Change to the target directory
    os.chdir(str(target_dir))

    # Load and render the template
    try:
        # First, try to load the template from the current directory
        env = Environment(loader=FileSystemLoader("./"))
        template = env.get_template("prompt_template.j2")
    except TemplateNotFound:
        # If not found, try to load from the parent directory (agent_tasks)
        try:
            env = Environment(loader=FileSystemLoader(str(current_dir)))
            template = env.get_template("prompt_template.j2")
        except TemplateNotFound:
            return {
                "error": "Template not found",
                "details": f"Looked for prompt_template.j2 in:\n1. {os.getcwd()}\n2. {current_dir}"
            }

    try:
        rendered_prompt = template.render(task=task_name)
    except Exception as e:
        return {"error": f"Error rendering template: {str(e)}"}

    # Execute the task
    model_size = 'x-small'
    tasks = [
        {
            "name": task["name"],
            "prompt": template.render(task),
        }
        for task in retreive_tasks(model_size)
    ]
    tasks = {t["name"]: t for t in tasks}
    task = tasks[task_name]

    return task 

# Example usage (can be removed if not needed)
if __name__ == "__main__":
    result = get_task("/path/to/target/directory", "mini_benchmark", "task_name")