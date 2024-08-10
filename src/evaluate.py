import torch
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve

def evaluate_model(model, test_loader, device):
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
    np.savetxt('logs/roc_curve_data.csv', roc_data, delimiter=',', header='fpr,tpr,thresholds', comments='')
    
    # Calculate AUC and Accuracy
    test_accuracy = accuracy_score(all_labels, (all_preds > 0.5).astype(int))
    test_auc = roc_auc_score(all_labels, all_preds)
    
    return test_accuracy, test_auc