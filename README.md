# Dense Dots

Dense Dots is a PyTorch-based implementation of neural networks designed for particle classification in high-energy physics experiments. This project aims to check the performance of Lorentz invariant inputs for distinguishing signal events from background events using particle 4-momentum data.

## Features

- Custom dataset loader for the Top Quark Tagging Reference Dataset:
  Kasieczka, G., Plehn, T., Thompson, J., & Russel, M. (2019). Top Quark Tagging Reference Dataset (v0 (2018_03_27)) [Data set]. Zenodo. https://doi.org/10.5281/zenodo.2603256
- Two neural network architectures:
  1. Dense Neural Network (DenseNN)
  2. Deep Sets Architecture (DeepSets)
- Training loop with validation and best model saving
- Evaluation metrics including accuracy, AUC, and ROC curve
- Background rejection vs. signal efficiency plotting
- Logging of training progress and results
- Modular code structure for easy maintenance and modification
- Command-line argument parsing for hyperparameter tuning and model selection

## Requirements

- PyTorch>=2.2.0
- NumPy>=1.26.4
- scikit-learn>=1.2.1
- h5py>=3.7.0
- matplotlib>=3.7.0

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/PradyunHebbar/dense_dots.git
   cd dense_dots
   ```

2. Install the required packages:
   ```
   pip install torch numpy scikit-learn h5py matplotlib
   ```

## Project Structure

- `dataloader.py`: Contains the `ParticleDataset` class for loading and preprocessing data
- `neuralnet.py`: Defines the `DenseNN` and `DeepSets` neural network architectures
- `train.py`: Implements the training loop and model saving
- `evaluate.py`: Contains functions for model evaluation and metrics calculation
- `run.py`: Main script to run the entire pipeline with command-line argument parsing
- `plotting.py`: Script for generating ROC curves and other plots

## Usage

1. Prepare your data:
   - Ensure you have 'train.h5', 'valid.h5', and 'test.h5' files in the specified directory.
   - Each file should contain 'Pmu' (particle 4-momentum) and 'is_signal' (binary classification) datasets.

2. Run the training and evaluation script with desired hyperparameters and model architecture:
   ```
   python run.py --batch_size 4096 --learning_rate 0.001 --num_epochs 30 --model_type densenn --train_path path/to/train.h5 --valid_path path/to/valid.h5 --test_path path/to/test.h5
   ```

   Available options for `--model_type` are:
   - `densenn`: Dense Neural Network
   - `deepsets`: Deep Sets Architecture

   You can adjust the hyperparameters as needed. Run `python run.py --help` to see all available options.

3. Check the 'logs' folder for results:
   - 'best_model_{model_type}.pth': The saved model with the best validation accuracy
   - 'best_metrics_{model_type}.txt': Best validation accuracy, test accuracy, and AUC score
   - 'roc_auc_score_{model_type}.txt': ROC AUC score
   - 'roc_curve_data.csv': False Positive Rate, True Positive Rate, and thresholds for ROC curve

4. Generate plots:
   ```
   python plotting.py
   ```
   This will create:
   - 'roc_curve.png': ROC curve plot
   - 'background_rejection_vs_signal_efficiency.png': Plot of background rejection vs. signal efficiency

## Model Architectures

### Dense Neural Network (DenseNN)
A traditional fully-connected neural network that processes the entire 200x200 dot product matrix as a flattened input.

### Deep Sets Architecture (DeepSets)
A permutation-invariant architecture that processes each row of the 200x200 dot product matrix independently before aggregating the results. This architecture is particularly suited for particle physics data where the order of particles should not affect the classification.

## Customization

You can modify the following files to customize the project:

- `neuralnet.py`: Adjust the neural network architectures in the `DenseNN` and `DeepSets` classes
- `train.py`: Modify the training process in the `train_model` function
- `evaluate.py`: Change evaluation metrics or add new ones in the `evaluate_model` function
- `run.py`: Adjust default hyperparameters or add new command-line arguments

## Contributing
Pradyun Hebbar pradyun.hebbar@gmail.com

## Acknowledgments

- This project was inspired by particle physics classification challenges in high-energy experiments.
- Thanks to the PyTorch and scikit-learn communities for their excellent libraries.
