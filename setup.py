from setuptools import setup, find_packages
import os

def package_files(directory):
    paths = []
    for (path, directories, filenames) in os.walk(directory):
        for filename in filenames:
            paths.append(os.path.join('..', path, filename))
    return paths

extra_files = package_files('agent_tasks/mini_benchmark') + package_files('agent_tasks/full_benchmark')

setup(
    name='agent-tasks',  
    version='v0.0.8', 
    packages=find_packages(),
    description='A task package for AI Research Bench', 
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    package_data={
        'agent_tasks': ['prompt_template.j2'] + extra_files,
    },
    include_package_data=True,
    install_requires=open('requirements.txt').read().splitlines(),
    author='Algorithmic Research Group',  
    author_email='matt@algorithmicresearchgroup.com', 
    keywords='tasks, agent, benchmark',
    url='http://github.com/ArtifactAI/agent-tasks',
    entry_points={
        'console_scripts': [
            'agent-tasks-run=agent_tasks.run:main',
        ],
    },
)