# Budgeted Model Training | ICCV 2023 RCV Workshop:

This challenge aims to build the best classifier for ImageNet100 within strict constraints. Participants cannot use pre-trained models or depend on external resources. The evaluation process includes multiple phases, with the final evaluation using a private ImageNet100 dataset. Hardware constraints include training on a P100 GPU with 4 CPU cores, a 9-hour time limit, and a 6GB GPU memory limit. The goal is to maximize accuracy while adhering to these resource constraints.


## Budgeted Model Training | ICCV 2023 RCV Workshop: Detailed Rules and Guidelines: https://www.kaggle.com/competitions/budgeted-model-training-iccv-2023-rcv-workshop/discussion?sort=undefined 

### Participation Requirements

* Goal: Build the best classifier for ImageNet100 within given constraints
* No use of pre-trained models
* Code should not depend on external pre-trained models or internet access
* Submit Python scripts and a .yaml file for the run environment
* Main script file should be named main.py and take one command line argument —data_dir
* Output a submission.csv in the current working directory

### Dataset

* ImageNet100 (100 classes subset of ImageNet)
* Final evaluation will use a different set of 100 classes than those revealed initially

### Evaluation Process

1. Phase I:

* Standard classification problem
* Maximize accuracy on the test set
* Hosted on Kaggle
* Leaderboard ranking based on accuracy

1. Phase II:

* For solutions with accuracy higher than baseline in Phase I
* Offline evaluation
* Re-run on private data with memory and time constraints
* Results viewable on Phase II Leaderboard
* Each team can submit two best solutions

1. Final Evaluation:

* Train on private ImageNet100 dataset
* Score on corresponding test set determines model performance

### Hardware Constraints

* Training on P100 GPU with 4 CPU cores
* Maximum 9 hours of training time
* GPU memory consumption should not exceed 6 GB at any point
* Final evaluation machine configuration:
* GPU: Quadro RTX 8000
* GPU Memory Limit: 6GB
* Total Time Limit: 9 Hours
* CPU Cores: 4
* RAM: 32GB

