import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
import os

from src.dataloader import ParticleDataset
from src.neuralnet import DenseNN
from src.train import train_model
from src.evaluate import evaluate_model

def main(args):
    # Device configuration
    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cuda')

    # Load datasets
    print("Loading datasets...")
    train_dataset = ParticleDataset(args.train_path)
    valid_dataset = ParticleDataset(args.valid_path)
    test_dataset = ParticleDataset(args.test_path)
    print("Datasets loaded successfully.")
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
    
    # Initialize model
    input_dim = 200*200  # Dot product matrix is now (N, N)
    model = DenseNN(input_dim).to(device)
    
    # Loss and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    
    # Train the model
    print("Starting training...")
    best_accuracy = train_model(model, train_loader, valid_loader, criterion, optimizer, args.num_epochs, device)
    print("Training completed.")
    
    # Create logs folder
    os.makedirs('./logs', exist_ok=True)
    
    # Evaluation
    print("Evaluating on test set...")
    model.load_state_dict(torch.load('best_model.pth'))
    test_accuracy, test_auc = evaluate_model(model, test_loader, device)
    
    print(f'Test Accuracy: {test_accuracy:.4f}')
    print(f'Test AUC: {test_auc:.4f}')

    
   

    # Move the best model to the logs folder
    os.rename('best_model.pth', './logs/best_model.pth')

    # Save best accuracy and AUC score
    with open('./logs/best_metrics.txt', 'w') as f:
        f.write(f'Best Accuracy: {best_accuracy:.4f}\n')
        f.write(f'Test AUC: {test_auc:.4f}\n')

    # Save ROC AUC score
    with open('./logs/roc_auc_score.txt', 'w') as f:
        f.write(f'ROC AUC Score: {test_auc:.4f}\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train and evaluate particle classification model')
    parser.add_argument('--train_path', type=str, default='../../Datasets/Greg_Top_tag_data/train.h5', help='Path to training dataset')
    parser.add_argument('--valid_path', type=str, default='../../Datasets/Greg_Top_tag_data/valid.h5', help='Path to validation dataset')
    parser.add_argument('--test_path', type=str, default='../../Datasets/Greg_Top_tag_data/test.h5', help='Path to test dataset')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for optimizer')
    parser.add_argument('--num_epochs', type=int, default=20, help='Number of epochs for training')
    
    args = parser.parse_args()
    main(args)