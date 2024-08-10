import torch
from torch.utils.data import Dataset
import h5py

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