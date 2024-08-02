import numpy as np
import matplotlib.pyplot as plt

# Load the ROC curve data
roc_data = np.loadtxt('../logs/roc_curve_data.csv', delimiter=',', skiprows=1)
fpr, tpr, _ = roc_data.T  # Transpose to get columns

# Create the ROC curve plot
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', label='Dense_dots ROC curve')
plt.plot([0, 1], [0, 1], color='red', linestyle='--', label='Random Classifier')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.savefig('./roc_curve.png')
plt.close()

plt.figure(figsize=(10, 8))
plt.plot(tpr, 1/fpr, label='Model Performance')
plt.xlabel(r'Signal Efficiency ($\epsilon_s$)')
plt.ylabel(r'Background Rejection (1/$\epsilon_b$)')
plt.title('Background Rejection vs Signal Efficiency')
plt.xlim([0, 1])
plt.ylim([1, 10000])  # Set a reasonable upper limit
plt.yscale('log')
plt.grid(True)
plt.legend(loc='lower left')
plt.savefig('./bgr_rej_vs_sig_eff.png')
plt.close()