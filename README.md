# ParticleNet: Dense Neural Network for Particle Classification

ParticleNet is a PyTorch-based implementation of a dense neural network designed for particle classification in high-energy physics experiments. This project aims to distinguish signal events from background events using particle 4-momentum data.

## Features

- Custom dataset loader for HDF5 files containing particle data
- Dense neural network architecture with configurable layers
- Training loop with validation and best model saving
- Comprehensive evaluation metrics including accuracy, AUC, and ROC curve
- Background rejection vs. signal efficiency plotting
- Detailed logging of training progress and results

## Requirements

- Python 3.7+
- PyTorch
- NumPy
- scikit-learn
- h5py
- matplotlib

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/ParticleNet.git
   cd ParticleNet
   ```

2. Install the required packages:
   ```
   pip install torch numpy scikit-learn h5py matplotlib
   ```

## Usage

1. Prepare your data:
   - Ensure you have 'train.h5', 'valid.h5', and 'test.h5' files in the project directory.
   - Each file should contain 'Pmu' (particle 4-momentum) and 'is_signal' (binary classification) datasets.

2. Run the training and evaluation script:
   ```
   python run.py
   ```

3. Check the 'logs' folder for results:
   - 'best_model.pth': The saved model with the best validation accuracy
   - 'best_metrics.txt': Best validation accuracy, test accuracy, and AUC score
   - 'roc_auc_score.txt': ROC AUC score
   - 'epoch_loss.json': Training loss for each epoch
   - 'roc_curve_data.csv': False Positive Rate, True Positive Rate, and thresholds for ROC curve
   - 'background_rejection_vs_signal_efficiency.png': Plot of background rejection vs. signal efficiency

## Customization

You can modify the following in `run.py`:

- Neural network architecture in the `DenseNN` class
- Hyperparameters such as learning rate, batch size, and number of epochs in the `main` function
- Training and evaluation process in the `train_model` and `main` functions

## Contributing

Contributions to ParticleNet are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- This project was inspired by particle physics classification challenges in high-energy experiments.
- Thanks to the PyTorch and scikit-learn communities for their excellent libraries.

