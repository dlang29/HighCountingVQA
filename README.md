# VQA Counting Task with Higher Object Numbers

## Overview
This repository contains the code and resources for the project on Visual Question Answering (VQA) focusing on the counting task with higher object numbers. The project leverages advanced VQA models to handle the challenges of counting a large number of objects in images.

## Team Members
- Ege Aktemur
- David Lang
- Lars Stockum
- Ronik Tempski
- Vibhanshu Singh Sindhu

## Project Structure
The repository is organized as follows:

- `config.py`: Configuration file containing model and training parameters.
- `train.py`: Script for training the VQA models.
- `evaluation.py`: Script for evaluating the trained models.
- `plots.py`: Script for generating plots and visualizations of the results.
- `dataset.py`: Script defining the dataset and data loader.
- `data/`: Directory containing the datasets and related files.

## Datasets
The project utilizes the following datasets:
- **TallyQA**: A dataset focused on object counting, which combines:
    - **COCO**: Contains diverse contextual images.
    - **Visual Genome**: Provides dense image annotations.

The images used in the dataset are derived from COCO and Visual Genome. All the images can be downloaded from the publicly available datasets below:
- [Visual Genome dataset](https://homes.cs.washington.edu/~ranjay/visualgenome/api.html)
- [COCO train/val images](https://visualqa.org/download.html)

These images should be placed into `/teamspace/studios/this_studio/data/`.

## Models
The project compares several state-of-the-art VQA models, including:
- **BLIP**: A Vision Transformer combined with BERT.
- **Pali-Gemma-3b**: Integrates SigLIP vision and PaLI-3 framework.

## Development Platform
The project was developed using the Platform as a Service (PaaS) Lightning AI, which provided advanced capabilities for model training and deployment:
Selecting necessary GPU compute power whenever needed, storing all the data sets and additionally giving a setup which already was runable. Therefore no requirements.txt provided.

## Setup and Installation
Recommended: Use Lightning AI
1. Select and clone the repository:
   ```bash
   git clone https://github.com/dlang29/HighCountingVQA.git
   ```

2. Ensure you have access to the datasets and prepare them that they are available in `/teamspace/studios/this_studio/data/`
3. update the paths in `config.py`.

## Configuration
The `config.py` file contains the necessary configuration parameters for the project. Key configurations include:
- `MODEL_ID`: Identifier for the pre-trained model.
- `DEVICE`: Device to run the computations (CPU or GPU).
- `DATA_ROOT`: Root directory for the datasets.
- `EPOCHS`, `BATCH_SIZE`, `LR`: Training parameters.

## Training
To train the models, run the `train.py` script:
```bash
python train.py
```
This script initializes the model, loads the dataset, and starts the training process with specified parameters.

## Evaluation
To evaluate the trained models, run the `evaluation.py` script:
```bash
python evaluation.py
```
This script evaluates the model's performance on the test set and saves the results.

## Plots and Visualization
To generate plots and visualize the results, run the `plots.py` script:
```bash
python plots.py
```
This script creates various plots, including data distributions, model accuracies, mean absolute errors, NaN counts, and confusion matrices.
