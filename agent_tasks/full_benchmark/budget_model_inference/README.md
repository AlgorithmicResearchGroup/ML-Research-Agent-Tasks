# Budgeted Inference Challenge | ICCV 2023 RCV Workshop

The objective is to classify UltraMNIST digits (3-5 digits per image) with limited GPU memory and inference time. Participants must predict the sum of digits (range: 0-27) in each image. The evaluation metric (TWE) combines accuracy and inference time. The challenge uses a V100 GPU with a 16GB memory constraint for inference. The final evaluation configuration includes a Quadro RTX 8000 GPU with specific memory and CPU core limitations.


## Budgeted Inference Challenge | ICCV 2023 RCV Workshop: Detailed Rules and Guidelines: https://www.kaggle.com/competitions/budgeted-model-inference-iccv-2023-rcv-workshop/ 

### Participation Requirements

* Goal: Classify UltraMNIST digits with limited GPU memory and inference time
* Task: Predict the sum of 3-5 digits per image (sum range: 0-27)
* Submit Python scripts, one model weights file, and a .yml file for the run environment
* Main script should be named main.py and take one command line argument —data_dir
* Output a submission.csv in the current directory

### Dataset

* UltraMNIST: Adapted variant with images containing 3-5 digits per image
* Digits extracted from the original MNIST dataset

### Evaluation Process

1. Phase I:

* Standard classification problem
* Maximize accuracy on the test set
* Hosted on Kaggle
* Leaderboard ranking based on accuracy

1. Phase II:

* For solutions with accuracy higher than baseline in Phase I
* Offline evaluation
* Evaluated on a separate private test set
* Results viewable on Phase II Leaderboard
* Each team can submit two best solutions

1. Final Evaluation:

* Inference speed and accuracy on private test set determine final score
* Metric: TWE = (accuracy^2) / inference_time (in minutes)

### Hardware Constraints

* Inference on V100 GPU with 16 GB memory constraint
* Final evaluation machine configuration:
* GPU: Quadro RTX 8000
* GPU Memory Limit: 16GB
* CPU Cores: 4
* RAM: 32GB

