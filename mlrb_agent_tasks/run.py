from typing import TypedDict, Dict, Any
from pathlib import Path
import shutil
import os
from jinja2 import Environment, FileSystemLoader, TemplateNotFound


class Task(TypedDict):
    name: str
    full_name: str
    task: str
    dataset: str
    dataset_shape: str
    models: str
    compute: str
    time_limit: str
    result: str
    evaluation: str
    rules: str
    additional_info: str


class TaskFamily:
    standard_version = "0.5.0"

    @staticmethod
    def get_instructions(t: Task) -> str:
        # Load and render the template
        current_dir = Path(__file__).parent
        env = Environment(loader=FileSystemLoader(str(current_dir)))
        
        try:
            template = env.get_template("prompt_template.j2")
            return template.render(
                task=t.get("task", ""),
                dataset=t.get("dataset", ""),
                dataset_shape=t.get("dataset_shape", ""),
                models=t.get("models", ""),
                compute=t.get("compute", ""),
                time_limit=t.get("time_limit", ""),
                result=t.get("result", ""),
                evaluation=t.get("evaluation", ""),
                rules=t.get("rules", ""),
                additional_info=t.get("additional_info", "")
            )
        except TemplateNotFound:
            return "Error: Template not found"
        except Exception as e:
            return f"Error rendering template: {str(e)}"

    @staticmethod
    def install() -> None:
        # Add any necessary installation steps
        pass

    @staticmethod
    def get_tasks() -> Dict[str, Task]:
        from mlrb_agent_tasks.prompts import task_templates
        return {t["name"]: t for t in task_templates}

    @staticmethod
    def score(t: Task, submission: str) -> float | None:
        # Implement scoring logic
        # Return None if the submission is invalid
        # Return a float score otherwise
        return None

    @staticmethod
    def setup_task_directory(directory: str, run_id: int) -> Dict[str, Any]:
        """Helper method to set up task directory
        
        Args:
            directory (str): Base directory name
            run_id (int): Run ID that will be used as a subdirectory name
        
        Returns:
            Dict[str, Any]: Status of the operation
        """
        try:
            # Create the base directory if it doesn't exist
            base_dir = Path(directory)
            base_dir.mkdir(parents=True, exist_ok=True)
            
            # Create the run-specific subdirectory
            run_dir = base_dir / str(run_id)
            run_dir.mkdir(parents=True, exist_ok=True)
            
            return {"success": True, "path": str(run_dir)}
        except Exception as e:
            return {"error": f"Error creating directory structure: {e}"}
