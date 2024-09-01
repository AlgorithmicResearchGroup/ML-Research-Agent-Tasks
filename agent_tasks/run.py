import os
import shutil
from pathlib import Path
from typing import Dict, Any
import subprocess
from jinja2 import Environment, FileSystemLoader, TemplateNotFound
from agent_tasks.prompts import retreive_tasks

def get_task(directory: str, benchmark: str, task_name: str) -> Dict[str, Any]:
    """
    Copy a task from a benchmark to a specified directory, execute it, and return the result.

    Args:
    directory (str): Target directory path
    benchmark (str): Benchmark type ('mini_benchmark' or 'full_benchmark')
    task (str): Task name

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

    # Copy the task directory to the target directory
    try:
        target_task_dir = target_dir / task_name
        shutil.copytree(str(task_dir), str(target_task_dir))
        print(f"Successfully copied '{task_name}' from {benchmark} to {directory}")
    except Exception as e:
        return {"error": f"Error copying task: {e}"}

    # Change to the target task directory
    os.chdir(str(target_task_dir))

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

    # Execute the task (this is a placeholder - replace with actual task execution)
    # try:
        # This is where you would actually execute the task
        # For now, we'll just return the rendered prompt
        #result = {"rendered_prompt": rendered_prompt}
        
    model_size = 'x-small'
    tasks = [
        {
            "name": task["name"],
            "prompt": template.render(task),
        }
        for task in retreive_tasks(model_size)
    ]
    tasks = {t["name"]: t for t in tasks}
    
    #task = tasks[task_name]
    #print(tasks)
    #result = next(t for t in tasks if 'mini_baby_lm' in t)
    #result = [t for t in tasks if "new2" in t][0]
    task = tasks[task_name]
        
        
    # except Exception as e:
    #     return {"error": f"Error executing task: {e}"}

    return task 

# Example usage (can be removed if not needed)
if __name__ == "__main__":
    result = get_task("/path/to/target/directory", "mini_benchmark", "task_name")