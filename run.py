import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np
from sklearn.metrics import accuracy_score
import os
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

# Updated Custom Dataset
class ParticleDataset(Dataset):
    def __init__(self, file_path):
        self.file = h5py.File(file_path, 'r')
        self.pmu = torch.tensor(self.file['Pmu'][:], dtype=torch.float32)
        self.is_signal = torch.tensor(self.file['is_signal'][:], dtype=torch.float32)
        self.minkowski_metric = torch.tensor([1, -1, -1, -1], dtype=torch.float32)
        
    def __len__(self):
        return len(self.pmu)
    
    def __getitem__(self, idx):
        pmu = self.pmu[idx]
        is_signal = self.is_signal[idx]
        
        # Calculate dot product correctly with Minkowski metric
        pmu_with_metric = pmu * self.minkowski_metric
        dot_product = torch.matmul(pmu, pmu_with_metric.t())
        
        return dot_product, is_signal

# Neural Network Model (unchanged)
class DenseNN(nn.Module):
    def __init__(self, input_dim):
        super(DenseNN, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(input_dim, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.fc4 = nn.Linear(128, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x):
        x = self.flatten(x)
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = self.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = self.relu(self.bn3(self.fc3(x)))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc4(x))
        return x

# Training function (unchanged)
def train_model(model, train_loader, valid_loader, criterion, optimizer, num_epochs, device):
    best_accuracy = 0.0
    for epoch in range(num_epochs):
        model.train()
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels.unsqueeze(1))
            loss.backward()
            optimizer.step()
            
            if batch_idx % 100 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}')
        
        # Validation
        model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for inputs, labels in valid_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                preds = (outputs > 0.5).float()
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        accuracy = accuracy_score(all_labels, all_preds)
        print(f'Validation Accuracy: {accuracy:.4f}')
        
        # Save best model
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), 'best_model.pth')
            print(f'Best model saved with accuracy: {best_accuracy:.4f}')
            
    return best_accuracy

# Main execution (unchanged)
def main():
    # Hyperparameters
    batch_size = 64
    learning_rate = 0.001
    num_epochs = 20
    
    # Device configuration
    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cuda')

    # Load datasets
    print("Loading datasets...")
    path="../../Datasets/Greg_Top_tag_data/"
    train_dataset = ParticleDataset(path+'train.h5')
    valid_dataset = ParticleDataset(path+'valid.h5')
    test_dataset = ParticleDataset(path+'test.h5')
    print("Datasets loaded successfully.")
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    # Initialize model
    input_dim = 200*200  # Dot product matrix is now (128, 128)
    model = DenseNN(input_dim).to(device)
    
    # Loss and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Train the model
    print("Starting training...")
    best_accuracy = train_model(model, train_loader, valid_loader, criterion, optimizer, num_epochs, device)
    print("Training completed.")
    
    # Evaluation
    print("Evaluating on test set...")
    model.load_state_dict(torch.load('best_model.pth'))
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            all_preds.extend(outputs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    all_preds = np.array(all_preds).flatten()
    all_labels = np.array(all_labels)

    # Calculate and save ROC curve
    fpr, tpr, thresholds = roc_curve(all_labels, all_preds)
    roc_data = np.column_stack((fpr, tpr, thresholds))
    np.savetxt('roc_curve_data.csv', roc_data, delimiter=',', header='fpr,tpr,thresholds', comments='')
    
    #Calculate AUC and Accuracy
    test_accuracy = accuracy_score(all_labels, (all_preds > 0.5).astype(int))
    test_auc = roc_auc_score(all_labels, all_preds)
    print(f'Test Accuracy: {test_accuracy:.4f}')
    print(f'Test AUC: {test_auc:.4f}')

    # Create logs folder
    os.makedirs('./logs', exist_ok=True)

    # Save best model (already done in train_model function)
    # Just move the best model to the logs folder
    os.rename('best_model.pth', './logs/best_model.pth')
    
    os.rename('roc_curve_data.csv', './logs/roc_curve_data.csv')

    # Save best accuracy and AUC score
    with open('./logs/best_metrics.txt', 'w') as f:
        f.write(f'Best Accuracy: {best_accuracy:.4f}\n')
        f.write(f'Test AUC: {test_auc:.4f}\n')

    # Save ROC AUC score
    with open('./logs/roc_auc_score.txt', 'w') as f:
        f.write(f'ROC AUC Score: {test_auc:.4f}\n')
        

if __name__ == '__main__':
    main()
